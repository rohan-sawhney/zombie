#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_scene_loader.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

class Scene {
public:
	fcpw::BoundingBox<2> bbox;
	std::vector<Vector2> vertices;
	std::vector<std::vector<size_t>> segments;

	const bool isWatertight;
	const bool isDoubleSided;

	zombie::GeometricQueries<2> queries;
	zombie::PDE<float, 2> pde;

	Scene(const json &config):
		  isWatertight(getOptional<bool>(config, "isWatertight", true)),
		  isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
		  queries(isWatertight)
	{
		const std::string boundaryFile = getRequired<std::string>(config, "boundary");
		const std::string isNeumannFile = getRequired<std::string>(config, "isNeumann");
		const std::string dirichletBoundaryValueFile = getRequired<std::string>(config, "dirichletBoundaryValue");
		const std::string neumannBoundaryValueFile = getRequired<std::string>(config, "neumannBoundaryValue");
		const std::string sourceValueFile = getRequired<std::string>(config, "sourceValue");
		bool normalize = getOptional<bool>(config, "normalizeDomain", true);
		bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
		absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

		isNeumann = std::make_shared<Image<1>>(isNeumannFile);
		dirichletBoundaryValue = std::make_shared<Image<1>>(dirichletBoundaryValueFile);
		neumannBoundaryValue = std::make_shared<Image<1>>(neumannBoundaryValueFile);
		sourceValue = std::make_shared<Image<1>>(sourceValueFile);

		loadOBJ(boundaryFile, normalize, flipOrientation);
		separateBoundaries();
		populateGeometricQueries();
		setPDE();
	}

	bool onNeumannBoundary(Vector2 x) const {
		Vector2 uv = (x - bbox.pMin) / bbox.extent().maxCoeff();
		return isNeumann->get(uv)[0] > 0;
	}

	bool ignoreCandidateSilhouette(float dihedralAngle, int index) const {
		// ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
		// NOTE: for complex scenes with both open and closed meshes, the primitive index argument
		// (of an adjacent line segment/triangle in the scene) can be used to determine whether a
		// vertex/edge should be ignored as a candidate for silhouette tests.
		return isDoubleSided ? false : dihedralAngle < 1e-3f;
	}

	float getSolveRegionVolume() const {
		if (isDoubleSided) return (bbox.pMax - bbox.pMin).prod();
		float solveRegionVolume = 0.0f;
		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		if (dirichletAggregate != nullptr) solveRegionVolume += dirichletAggregate->signedVolume();
		if (neumannAggregate != nullptr) solveRegionVolume += neumannAggregate->signedVolume();
		return std::fabs(solveRegionVolume);
	}


private:
	void loadOBJ(const std::string &filename, bool normalize, bool flipOrientation) {
		std::ifstream obj(filename);
		if (!obj) {
			std::cerr << "Error opening file: " << filename << std::endl;
			abort();
		}

		std::string line;
		while (std::getline(obj, line)) {
			std::istringstream ss(line);
			std::string token;
			ss >> token;
			if (token == "v") {
				float x, y;
				ss >> x >> y;
				vertices.emplace_back(Vector2(x, y));
			} else if (token == "l") {
				size_t i, j;
				ss >> i >> j;
				if (flipOrientation) {
					segments.emplace_back(std::vector<size_t>({j - 1, i - 1}));
				} else {
					segments.emplace_back(std::vector<size_t>({i - 1, j - 1}));
				}
			}
		}
		obj.close();

		if (normalize) {
			Vector2 cm(0, 0);
			for (Vector2 v : vertices) cm += v;
			cm /= vertices.size();
			float radius = 0.0f;
			for (Vector2& v : vertices) {
				v -= cm;
				radius = std::max(radius, v.norm());
			}
			for (Vector2& v : vertices) v /= radius;
		}

		bbox = zombie::computeBoundingBox(vertices, true, 1.0);
	}

	void separateBoundaries() {
		std::vector<size_t> indices(2, -1);
		size_t vDirichlet = 0, vNeumann = 0;
		std::unordered_map<size_t, size_t> dirichletIndexMap, neumannIndexMap;
		for (int i = 0; i < segments.size(); i++) {
			Vector2 pMid = 0.5f * (vertices[segments[i][0]] + vertices[segments[i][1]]);
			if (onNeumannBoundary(pMid)) {
				for (int j = 0; j < 2; j++) {
					size_t vIndex = segments[i][j];
					if (neumannIndexMap.find(vIndex) == neumannIndexMap.end()) {
						const Vector2& p = vertices[vIndex];
						neumannVertices.emplace_back(p);
						neumannIndexMap[vIndex] = vNeumann++;
					}
					indices[j] = neumannIndexMap[vIndex];
				}
				neumannSegments.emplace_back(indices);
			} else {
				for (int j = 0; j < 2; j++) {
					size_t vIndex = segments[i][j];
					if (dirichletIndexMap.find(vIndex) == dirichletIndexMap.end()) {
						const Vector2& p = vertices[vIndex];
						dirichletVertices.emplace_back(p);
						dirichletIndexMap[vIndex] = vDirichlet++;
					}
					indices[j] = dirichletIndexMap[vIndex];
				}
				dirichletSegments.emplace_back(indices);
			}
		}

		std::function<bool(float, int)> ignoreCandidateSilhouette = [this](float dihedralAngle, int index) -> bool {
			return this->ignoreCandidateSilhouette(dihedralAngle, index);
		};
		dirichletSceneLoader = new zombie::FcpwSceneLoader<2>(dirichletVertices, dirichletSegments);
		neumannSceneLoader = new zombie::FcpwSceneLoader<2>(neumannVertices, neumannSegments,
															ignoreCandidateSilhouette, true);
	}

	void populateGeometricQueries() {
		neumannSamplingTraversalWeight = [this](float r2) -> float {
			float r = std::max(std::sqrt(r2), 1e-2f);
			return std::fabs(this->harmonicGreensFn.evaluate(r));
		};

		const fcpw::Aggregate<3> *dirichletAggregate = dirichletSceneLoader->getSceneAggregate();
		const fcpw::Aggregate<3> *neumannAggregate = neumannSceneLoader->getSceneAggregate();
		zombie::populateGeometricQueries<2>(queries, bbox, dirichletAggregate, neumannAggregate,
											neumannSamplingTraversalWeight);
	}

	void setPDE() {
		float maxLength = this->bbox.extent().maxCoeff();
		pde.dirichlet = [this, maxLength](const Vector2& x) -> float {
			Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return this->dirichletBoundaryValue->get(uv)[0];
		};
		pde.neumann = [this, maxLength](const Vector2& x) -> float {
			Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return this->neumannBoundaryValue->get(uv)[0];
		};
		pde.dirichletDoubleSided = [this, maxLength](const Vector2& x, bool _) -> float {
			Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return this->dirichletBoundaryValue->get(uv)[0];
		};
		pde.neumannDoubleSided = [this, maxLength](const Vector2& x, bool _) -> float {
			Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return this->neumannBoundaryValue->get(uv)[0];
		};
		pde.source = [this, maxLength](const Vector2& x) -> float {
			Vector2 uv = (x - this->bbox.pMin) / maxLength;
			return this->sourceValue->get(uv)[0];
		};
		pde.absorption = absorptionCoeff;
	}

	std::vector<Vector2> dirichletVertices;
	std::vector<Vector2> neumannVertices;

	std::vector<std::vector<size_t>> dirichletSegments;
	std::vector<std::vector<size_t>> neumannSegments;

	zombie::FcpwSceneLoader<2>* dirichletSceneLoader;
	zombie::FcpwSceneLoader<2>* neumannSceneLoader;

	std::shared_ptr<Image<1>> isNeumann;
	std::shared_ptr<Image<1>> dirichletBoundaryValue;
	std::shared_ptr<Image<1>> neumannBoundaryValue;
	std::shared_ptr<Image<1>> sourceValue;
	float absorptionCoeff;

	zombie::HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
	std::function<float(float)> neumannSamplingTraversalWeight;
};

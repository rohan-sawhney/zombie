#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_boundary_handler.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

class Scene {
public:
    std::pair<Vector2, Vector2> bbox;
    std::vector<Vector2> vertices;
    std::vector<std::vector<size_t>> segments;

    const bool isWatertight;
    const bool isDoubleSided;

    zombie::GeometricQueries<2> queries;
    zombie::PDE<float, 2> pde;

    Scene(const json &config):
          isWatertight(getOptional<bool>(config, "isWatertight", true)),
          isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
          queries(isWatertight) {
        const std::string boundaryFile = getRequired<std::string>(config, "boundary");
        const std::string isReflectingFile = getRequired<std::string>(config, "isNeumann");
        const std::string absorbingBoundaryValueFile = getRequired<std::string>(config, "dirichletBoundaryValue");
        const std::string reflectingBoundaryValueFile = getRequired<std::string>(config, "neumannBoundaryValue");
        const std::string sourceValueFile = getRequired<std::string>(config, "sourceValue");
        bool normalize = getOptional<bool>(config, "normalizeDomain", true);
        bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
        bool useReflectingRobinBoundaries = getOptional<bool>(config, "useReflectingRobinBoundaries", false);
        absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);
        robinCoeff = getOptional<float>(config, "robinCoeff", std::numeric_limits<float>::lowest());

        isReflecting = std::make_shared<Image<1>>(isReflectingFile);
        absorbingBoundaryValue = std::make_shared<Image<1>>(absorbingBoundaryValueFile);
        reflectingBoundaryValue = std::make_shared<Image<1>>(reflectingBoundaryValueFile);
        sourceValue = std::make_shared<Image<1>>(sourceValueFile);

        loadOBJ(boundaryFile, normalize, flipOrientation);
        buildAccelerationStructures(useReflectingRobinBoundaries);
        populateGeometricQueries(useReflectingRobinBoundaries);
        setPDE(useReflectingRobinBoundaries);
    }

    bool onReflectingBoundary(Vector2 x) const {
        const Vector2& bMin = bbox.first;
        const Vector2& bMax = bbox.second;
        Vector2 uv = (x - bMin) / (bMax - bMin).maxCoeff();
        return isReflecting->get(uv)[0] > 0;
    }

    bool ignoreCandidateSilhouette(float dihedralAngle, int index) const {
        // ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
        // NOTE: for complex scenes with both open and closed meshes, the primitive index argument
        // (of an adjacent line segment/triangle in the scene) can be used to determine whether a
        // vertex/edge should be ignored as a candidate for silhouette tests.
        return isDoubleSided ? false : dihedralAngle < 1e-3f;
    }

    float getSolveRegionVolume() const {
        if (isDoubleSided) return (bbox.second - bbox.first).prod();
        return std::fabs(queries.computeSignedDomainVolume());
    }

private:
    void loadOBJ(const std::string &filename, bool normalize, bool flipOrientation) {
        zombie::loadBoundaryMesh<2>(filename, vertices, segments);
        if (normalize) zombie::normalize<2>(vertices);
        if (flipOrientation) zombie::flipOrientation(segments);
        bbox = zombie::computeBoundingBox<2>(vertices, true, 1.0);
    }

    void buildAccelerationStructures(bool useReflectingRobinBoundaries) {
        std::vector<size_t> indices(2, -1);
        size_t nAbsorbingBoundaryVerts = 0, nReflectingBoundaryVerts = 0;
        std::unordered_map<size_t, size_t> absorbingBoundaryIndexMap, reflectingBoundaryIndexMap;
        for (int i = 0; i < segments.size(); i++) {
            Vector2 pMid = 0.5f * (vertices[segments[i][0]] + vertices[segments[i][1]]);
            if (onReflectingBoundary(pMid)) {
                for (int j = 0; j < 2; j++) {
                    size_t vIndex = segments[i][j];
                    if (reflectingBoundaryIndexMap.find(vIndex) == reflectingBoundaryIndexMap.end()) {
                        const Vector2& p = vertices[vIndex];
                        reflectingBoundaryVertices.emplace_back(p);
                        reflectingBoundaryIndexMap[vIndex] = nReflectingBoundaryVerts++;
                    }
                    indices[j] = reflectingBoundaryIndexMap[vIndex];
                }
                reflectingBoundarySegments.emplace_back(indices);
            } else {
                for (int j = 0; j < 2; j++) {
                    size_t vIndex = segments[i][j];
                    if (absorbingBoundaryIndexMap.find(vIndex) == absorbingBoundaryIndexMap.end()) {
                        const Vector2& p = vertices[vIndex];
                        absorbingBoundaryVertices.emplace_back(p);
                        absorbingBoundaryIndexMap[vIndex] = nAbsorbingBoundaryVerts++;
                    }
                    indices[j] = absorbingBoundaryIndexMap[vIndex];
                }
                absorbingBoundarySegments.emplace_back(indices);
            }
        }

        absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryVertices, absorbingBoundarySegments);
        std::function<bool(float, int)> ignoreCandidateSilhouette = [this](float dihedralAngle, int index) -> bool {
            return this->ignoreCandidateSilhouette(dihedralAngle, index);
        };
        if (useReflectingRobinBoundaries) {
            std::vector<float> minRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
            std::vector<float> maxRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
            reflectingRobinBoundaryHandler.buildAccelerationStructure(reflectingBoundaryVertices,
                                                                      reflectingBoundarySegments,
                                                                      ignoreCandidateSilhouette, false,
                                                                      minRobinCoeffValues, maxRobinCoeffValues);

        } else {
            reflectingNeumannBoundaryHandler.buildAccelerationStructure(reflectingBoundaryVertices,
                                                                        reflectingBoundarySegments,
                                                                        ignoreCandidateSilhouette, true);
        }
    }

    void populateGeometricQueries(bool useReflectingRobinBoundaries) {
        branchTraversalWeight = [this](float r2) -> float {
            float r = std::max(std::sqrt(r2), 1e-2f);
            return std::fabs(this->harmonicGreensFn.evaluate(r));
        };

        if (useReflectingRobinBoundaries) {
            zombie::populateGeometricQueries<2, true>(absorbingBoundaryHandler,
                                                      reflectingRobinBoundaryHandler,
                                                      branchTraversalWeight, bbox, queries);

        } else {
            zombie::populateGeometricQueries<2, false>(absorbingBoundaryHandler,
                                                       reflectingNeumannBoundaryHandler,
                                                       branchTraversalWeight, bbox, queries);
        }
    }

    void setPDE(bool useReflectingRobinBoundaries) {
        const Vector2& bMin = this->bbox.first;
        const Vector2& bMax = this->bbox.second;
        float maxLength = (bMax - bMin).maxCoeff();
        pde.source = [this, &bMin, maxLength](const Vector2& x) -> float {
            Vector2 uv = (x - bMin) / maxLength;
            return this->sourceValue->get(uv)[0];
        };
        pde.dirichlet = [this, &bMin, maxLength](const Vector2& x) -> float {
            Vector2 uv = (x - bMin) / maxLength;
            return this->absorbingBoundaryValue->get(uv)[0];
        };
        pde.dirichletDoubleSided = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
            Vector2 uv = (x - bMin) / maxLength;
            return this->absorbingBoundaryValue->get(uv)[0];
        };
        if (useReflectingRobinBoundaries) {
            pde.robin = [this, &bMin, maxLength](const Vector2& x) -> float {
                Vector2 uv = (x - bMin) / maxLength;
                return this->reflectingBoundaryValue->get(uv)[0];
            };
            pde.robinCoeff = [this](const Vector2& x) -> float {
                return this->robinCoeff;
            };
            pde.robinDoubleSided = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
                Vector2 uv = (x - bMin) / maxLength;
                return this->reflectingBoundaryValue->get(uv)[0];
            };
            pde.robinCoeffDoubleSided = [this](const Vector2& x, bool boundaryNormalAligned) -> float {
                return this->robinCoeff;
            };

        } else {
            pde.neumann = [this, &bMin, maxLength](const Vector2& x) -> float {
                Vector2 uv = (x - bMin) / maxLength;
                return this->reflectingBoundaryValue->get(uv)[0];
            };
            pde.neumannDoubleSided = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
                Vector2 uv = (x - bMin) / maxLength;
                return this->reflectingBoundaryValue->get(uv)[0];
            };
        }
        pde.absorption = absorptionCoeff;
    }

    std::vector<Vector2> absorbingBoundaryVertices;
    std::vector<Vector2> reflectingBoundaryVertices;
    std::vector<std::vector<size_t>> absorbingBoundarySegments;
    std::vector<std::vector<size_t>> reflectingBoundarySegments;

    zombie::FcpwBoundaryHandler<2, false> absorbingBoundaryHandler;
    zombie::FcpwBoundaryHandler<2, false> reflectingNeumannBoundaryHandler;
    zombie::FcpwBoundaryHandler<2, true> reflectingRobinBoundaryHandler;

    std::shared_ptr<Image<1>> isReflecting;
    std::shared_ptr<Image<1>> absorbingBoundaryValue;
    std::shared_ptr<Image<1>> reflectingBoundaryValue;
    std::shared_ptr<Image<1>> sourceValue;
    float absorptionCoeff, robinCoeff;

    zombie::HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
    std::function<float(float)> branchTraversalWeight;
};

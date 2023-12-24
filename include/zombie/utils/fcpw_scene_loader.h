#pragma once

#include <zombie/core/geometric_queries.h>
#include <cmath>
#include <fcpw/fcpw.h>
#include <fcpw/utilities/scene_loader.h>

#define RAY_OFFSET 1e-6f

namespace zombie {

template <int DIM>
void loadSurfaceMesh(const std::string& filename,
					 std::vector<Vector<DIM>>& meshPositions,
					 std::vector<std::vector<size_t>>& meshIndices)
{
	// do nothing
}

template <>
void loadSurfaceMesh<2>(const std::string& filename,
						std::vector<Vector2>& meshPositions,
						std::vector<std::vector<size_t>>& meshIndices)
{
	// load file
	fcpw::PolygonSoup<3> soup;
	fcpw::loadLineSegmentSoupFromOBJFile(filename, soup);

	// collect mesh positions and indices
	meshPositions.clear();
	meshIndices.clear();
	int V = (int)soup.positions.size();
	int L = (int)soup.indices.size()/2;

	for (int l = 0; l < L; l++) {
		size_t i = soup.indices[2*l + 0];
		size_t j = soup.indices[2*l + 1];

		meshIndices.emplace_back(std::vector<size_t>{i, j});
	}

	for (int v = 0; v < V; v++) {
		meshPositions.emplace_back(soup.positions[v].head(2));
	}
}

template <>
void loadSurfaceMesh<3>(const std::string& filename,
						std::vector<Vector3>& meshPositions,
						std::vector<std::vector<size_t>>& meshIndices)
{
	// load file
	fcpw::PolygonSoup<3> soup;
	fcpw::loadTriangleSoupFromOBJFile(filename, soup);

	// collect mesh positions and indices
	meshPositions.clear();
	meshIndices.clear();
	int V = (int)soup.positions.size();
	int T = (int)soup.indices.size()/3;

	for (int t = 0; t < T; t++) {
		size_t i = soup.indices[3*t + 0];
		size_t j = soup.indices[3*t + 1];
		size_t k = soup.indices[3*t + 2];

		meshIndices.emplace_back(std::vector<size_t>{i, j, k});
	}

	for (int v = 0; v < V; v++) {
		meshPositions.emplace_back(soup.positions[v]);
	}
}

template <int DIM>
fcpw::BoundingBox<DIM> computeBoundingBox(const std::vector<Vector<DIM>>& positions,
										  bool makeSquare, float scale)
{
	fcpw::BoundingBox<DIM> bbox;
	for (size_t i = 0; i < positions.size(); i++) {
		bbox.expandToInclude(positions[i]*scale);
	}

	if (makeSquare) {
		Vector<DIM> center = bbox.centroid();
		Vector<DIM> extent = bbox.extent();
		float maxCoeff = 0.5f*extent.maxCoeff();
		bbox.pMin = center - Vector<DIM>::Constant(maxCoeff);
		bbox.pMax = center + Vector<DIM>::Constant(maxCoeff);
	}

	return bbox;
}

template <int DIM>
class FcpwSceneLoader {
public:
	// constructor
	FcpwSceneLoader(const std::vector<Vector<DIM>>& positions,
					const std::vector<std::vector<size_t>>& indices,
					std::function<bool(float, int)> ignoreCandidateSilhouette={},
					bool computeSilhouettes=false, bool buildBvh=true,
					bool computeWeightedNormals=false) {
		std::cerr << "FcpwSceneLoader() not implemented for DIM: " << DIM << std::endl;
		exit(EXIT_FAILURE);
	}

	// returns scene aggregate
	const fcpw::Aggregate<3>* getSceneAggregate() {
		std::cerr << "FcpwSceneLoader::getSceneAggregate() not implemented for DIM: " << DIM << std::endl;
		exit(EXIT_FAILURE);

		return nullptr;
	}
};

template <>
class FcpwSceneLoader<2> {
public:
	// constructor
	FcpwSceneLoader(const std::vector<Vector2>& positions,
					const std::vector<std::vector<size_t>>& indices,
					std::function<bool(float, int)> ignoreCandidateSilhouette={},
					bool computeSilhouettes=false, bool buildBvh=true,
					bool computeWeightedNormals=false) {
		if (positions.size() > 0) {
			// scene geometry is made up of line segments
			std::vector<std::vector<fcpw::PrimitiveType>> objectTypes(
				1, std::vector<fcpw::PrimitiveType>{fcpw::PrimitiveType::LineSegment});
			scene.setObjectTypes(objectTypes);

			// set the vertex and line segment count
			int V = (int)positions.size();
			int L = (int)indices.size();
			scene.setObjectVertexCount(V, 0);
			scene.setObjectLineSegmentCount(L, 0);

			// specify the vertex positions
			for (int i = 0; i < V; i++) {
				Vector3 position = Vector3::Zero();
				position.head(2) = positions[i];

				scene.setObjectVertex(position, i, 0);
			}

			// specify the line segment indices
			for (int i = 0; i < L; i++) {
				int index[2] = {(int)indices[i][0], (int)indices[i][1]};
				scene.setObjectLineSegment(index, i, 0);
			}

			// compute normals
			scene.computeObjectNormals(0, computeWeightedNormals);

			// compute silhouettes
			if (computeSilhouettes) {
				scene.computeSilhouettes(ignoreCandidateSilhouette);
			}

			// build aggregate
			fcpw::AggregateType aggregateType = buildBvh ?
												fcpw::AggregateType::Bvh_OverlapSurfaceArea :
												fcpw::AggregateType::Baseline;
			scene.build(aggregateType, true, true, true);
		}
	}

	// returns scene aggregate
	const fcpw::Aggregate<3>* getSceneAggregate() {
		fcpw::SceneData<3> *sceneData = scene.getSceneData();
		return sceneData->soups.size() > 0 ? sceneData->aggregate.get() : nullptr;
	}

	// member
	fcpw::Scene<3> scene;
};

template <>
class FcpwSceneLoader<3> {
public:
	// constructor
	FcpwSceneLoader(const std::vector<Vector3>& positions,
					const std::vector<std::vector<size_t>>& indices,
					std::function<bool(float, int)> ignoreCandidateSilhouette={},
					bool computeSilhouettes=false, bool buildBvh=true,
					bool computeWeightedNormals=false) {
		if (positions.size() > 0) {
			// scene geometry is made up of triangles
			std::vector<std::vector<fcpw::PrimitiveType>> objectTypes(
				1, std::vector<fcpw::PrimitiveType>{fcpw::PrimitiveType::Triangle});
			scene.setObjectTypes(objectTypes);

			// set the vertex and triangle count
			int V = (int)positions.size();
			int T = (int)indices.size();
			scene.setObjectVertexCount(V, 0);
			scene.setObjectTriangleCount(T, 0);

			// specify the vertex positions
			for (int i = 0; i < V; i++) {
				scene.setObjectVertex(positions[i], i, 0);
			}

			// specify the triangle indices
			for (int i = 0; i < T; i++) {
				int index[3] = {(int)indices[i][0], (int)indices[i][1], (int)indices[i][2]};
				scene.setObjectTriangle(index, i, 0);
			}

			// compute normals
			scene.computeObjectNormals(0, computeWeightedNormals);

			// compute silhouettes
			if (computeSilhouettes) {
				scene.computeSilhouettes(ignoreCandidateSilhouette);
			}

			// build aggregate
			fcpw::AggregateType aggregateType = buildBvh ?
												fcpw::AggregateType::Bvh_OverlapSurfaceArea :
												fcpw::AggregateType::Baseline;
			scene.build(aggregateType, true, true, true);
		}
	}

	// returns scene aggregate
	const fcpw::Aggregate<3>* getSceneAggregate() {
		fcpw::SceneData<3> *sceneData = scene.getSceneData();
		return sceneData->soups.size() > 0 ? sceneData->aggregate.get() : nullptr;
	}

	// member
	fcpw::Scene<3> scene;
};

inline float int_as_float(int a)
{
	union {int a; float b;} u;
	u.a = a;

	return u.b;
}

inline int float_as_int(float a)
{
	union {float a; int b;} u;
	u.a = a;

	return u.b;
}

template <int DIM>
inline Vector<DIM> offsetPointAlongDirection(const Vector<DIM>& p, const Vector<DIM>& n)
{
	return p + RAY_OFFSET*n;
}

template <>
inline Vector2 offsetPointAlongDirection<2>(const Vector2& p, const Vector2& n)
{
	// source: https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf
	const float origin = 1.0f/32.0f;
	const float floatScale = 1.0f/65536.0f;
	const float intScale = 256.0f;

	Eigen::Vector2i nOffset(n(0)*intScale, n(1)*intScale);
	Eigen::Vector2f pOffset(int_as_float(float_as_int(p(0)) + (p(0) < 0 ? -nOffset(0) : nOffset(0))),
							int_as_float(float_as_int(p(1)) + (p(1) < 0 ? -nOffset(1) : nOffset(1))));

	return Eigen::Vector2f(std::fabs(p(0)) < origin ? p(0) + floatScale*n(0) : pOffset(0),
						   std::fabs(p(1)) < origin ? p(1) + floatScale*n(1) : pOffset(1));
}

template <>
inline Vector3 offsetPointAlongDirection<3>(const Vector3& p, const Vector3& n)
{
	// source: https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf
	const float origin = 1.0f/32.0f;
	const float floatScale = 1.0f/65536.0f;
	const float intScale = 256.0f;

	Eigen::Vector3i nOffset(n(0)*intScale, n(1)*intScale, n(2)*intScale);
	Eigen::Vector3f pOffset(int_as_float(float_as_int(p(0)) + (p(0) < 0 ? -nOffset(0) : nOffset(0))),
							int_as_float(float_as_int(p(1)) + (p(1) < 0 ? -nOffset(1) : nOffset(1))),
							int_as_float(float_as_int(p(2)) + (p(2) < 0 ? -nOffset(2) : nOffset(2))));

	return Eigen::Vector3f(std::fabs(p(0)) < origin ? p(0) + floatScale*n(0) : pOffset(0),
						   std::fabs(p(1)) < origin ? p(1) + floatScale*n(1) : pOffset(1),
						   std::fabs(p(2)) < origin ? p(2) + floatScale*n(2) : pOffset(2));
}

template <int DIM>
void populateGeometricQueries(GeometricQueries<DIM>& geometricQueries,
							  const fcpw::BoundingBox<DIM>& boundingBox,
							  const fcpw::Aggregate<3> *dirichletAggregate,
							  const fcpw::Aggregate<3> *neumannAggregate,
							  const std::function<float(float)>& neumannSamplingTraversalWeight)
{
	geometricQueries.computeDistToDirichlet = [dirichletAggregate, &boundingBox](
											   const Vector<DIM>& x, bool computeSignedDistance) -> float {
		if (dirichletAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
			dirichletAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

			return computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;
		}

		float d2Min, d2Max;
		boundingBox.computeSquaredDistance(x, d2Min, d2Max);
		return std::sqrt(d2Max);
	};
	geometricQueries.computeDistToNeumann = [neumannAggregate](const Vector<DIM>& x,
															   bool computeSignedDistance) -> float {
		if (neumannAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
			neumannAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

			return computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;
		}

		return fcpw::maxFloat;
	};
	geometricQueries.computeDistToBoundary = [&geometricQueries](const Vector<DIM>& x,
																 bool computeSignedDistance,
																 bool& closerToDirichlet) -> float {
		float d1 = geometricQueries.computeDistToDirichlet(x, computeSignedDistance);
		float d2 = geometricQueries.computeDistToNeumann(x, computeSignedDistance);

		if (std::fabs(d1) < std::fabs(d2)) {
			closerToDirichlet = true;
			return d1;
		}

		closerToDirichlet = false;
		return d2;
	};
	geometricQueries.projectToDirichlet = [dirichletAggregate](Vector<DIM>& x, Vector<DIM>& normal,
															   float& distance, bool computeSignedDistance) -> bool {
		if (dirichletAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
			dirichletAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

			x = interaction.p.head(DIM);
			normal = interaction.n.head(DIM);
			distance = computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;

			return true;
		}

		distance = 0.0f;
		return false;
	};
	geometricQueries.projectToNeumann = [neumannAggregate](Vector<DIM>& x, Vector<DIM>& normal,
														   float& distance, bool computeSignedDistance) -> bool {
		if (neumannAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> sphere(queryPt, fcpw::maxFloat);
			neumannAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

			x = interaction.p.head(DIM);
			normal = interaction.n.head(DIM);
			distance = computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;

			return true;
		}

		distance = 0.0f;
		return false;
	};
	geometricQueries.projectToBoundary = [&geometricQueries, dirichletAggregate, neumannAggregate](
										  Vector<DIM>& x, Vector<DIM>& normal, float& distance,
										  bool& projectToDirichlet, bool computeSignedDistance) -> bool {
		if (dirichletAggregate != nullptr && neumannAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			fcpw::Interaction<3> interactionDirichlet;
			fcpw::BoundingSphere<3> sphereDirichlet(queryPt, fcpw::maxFloat);
			dirichletAggregate->findClosestPoint(sphereDirichlet, interactionDirichlet, computeSignedDistance);

			fcpw::Interaction<3> interactionNeumann;
			fcpw::BoundingSphere<3> sphereNeumann(queryPt, fcpw::maxFloat);
			neumannAggregate->findClosestPoint(sphereNeumann, interactionNeumann, computeSignedDistance);

			if (interactionDirichlet.d < interactionNeumann.d) {
				x = interactionDirichlet.p.head(DIM);
				normal = interactionDirichlet.n.head(DIM);
				distance = computeSignedDistance ? interactionDirichlet.signedDistance(queryPt) : interactionDirichlet.d;
				projectToDirichlet = true;

			} else {
				x = interactionNeumann.p.head(DIM);
				normal = interactionNeumann.n.head(DIM);
				distance = computeSignedDistance ? interactionNeumann.signedDistance(queryPt) : interactionNeumann.d;
				projectToDirichlet = false;
			}

			return true;
		}

		if (geometricQueries.projectToDirichlet(x, normal, distance, computeSignedDistance)) {
			projectToDirichlet = true;
			return true;

		} else if (geometricQueries.projectToNeumann(x, normal, distance, computeSignedDistance)) {
			projectToDirichlet = false;
			return true;
		}

		return false;
	};
	geometricQueries.offsetPointAlongDirection = [](const Vector<DIM>& x,
													const Vector<DIM>& dir) -> Vector<DIM> {
		return offsetPointAlongDirection<DIM>(x, dir);
	};
	geometricQueries.intersectWithDirichlet = [&geometricQueries, dirichletAggregate](
											   const Vector<DIM>& origin, const Vector<DIM>& normal,
											   const Vector<DIM>& dir, float tMax, bool onDirichletBoundary,
											   IntersectionPoint<DIM>& intersectionPt) -> bool {
		if (dirichletAggregate != nullptr) {
			Vector3 queryOrigin = Vector3::Zero();
			Vector3 queryDir = Vector3::Zero();

			queryOrigin.head(DIM) = onDirichletBoundary ?
									geometricQueries.offsetPointAlongDirection(origin, -normal) :
									origin;
			queryDir.head(DIM) = dir;

			fcpw::Ray<3> queryRay(queryOrigin, queryDir, tMax);
			std::vector<fcpw::Interaction<3>> queryInteractions;
			int nHits = dirichletAggregate->intersect(queryRay, queryInteractions, false, false);
			if (nHits < 1) return false;

			intersectionPt.pt = queryInteractions[0].p.head(DIM);
			intersectionPt.normal = queryInteractions[0].n.head(DIM);
			intersectionPt.dist = queryInteractions[0].d;

			return true;
		}

		return false;
	};
	geometricQueries.intersectWithNeumann = [&geometricQueries, neumannAggregate](
											 const Vector<DIM>& origin, const Vector<DIM>& normal,
											 const Vector<DIM>& dir, float tMax, bool onNeumannBoundary,
											 IntersectionPoint<DIM>& intersectionPt) -> bool {
		if (neumannAggregate != nullptr) {
			Vector3 queryOrigin = Vector3::Zero();
			Vector3 queryDir = Vector3::Zero();

			queryOrigin.head(DIM) = onNeumannBoundary ?
									geometricQueries.offsetPointAlongDirection(origin, -normal) :
									origin;
			queryDir.head(DIM) = dir;

			fcpw::Ray<3> queryRay(queryOrigin, queryDir, tMax);
			std::vector<fcpw::Interaction<3>> queryInteractions;
			int nHits = neumannAggregate->intersect(queryRay, queryInteractions, false, false);
			if (nHits < 1) return false;

			intersectionPt.pt = queryInteractions[0].p.head(DIM);
			intersectionPt.normal = queryInteractions[0].n.head(DIM);
			intersectionPt.dist = queryInteractions[0].d;

			return true;
		}

		return false;
	};
	geometricQueries.intersectsWithNeumann = [&geometricQueries, neumannAggregate](
											  const Vector<DIM>& xi, const Vector<DIM>& xj,
											  const Vector<DIM>& ni, const Vector<DIM>& nj,
											  bool offseti, bool offsetj) -> bool {
		if (neumannAggregate != nullptr) {
			Vector3 pt1 = Vector3::Zero();
			Vector3 pt2 = Vector3::Zero();
			pt1.head(DIM) = offseti ? geometricQueries.offsetPointAlongDirection(xi, -ni) : xi;
			pt2.head(DIM) = offsetj ? geometricQueries.offsetPointAlongDirection(xj, -nj) : xj;

			return !neumannAggregate->hasLineOfSight(pt1, pt2);
		}

		return false;
	};
	geometricQueries.intersectWithBoundary = [&geometricQueries](
											  const Vector<DIM>& origin, const Vector<DIM>& normal,
											  const Vector<DIM>& dir, float tMax, bool onDirichletBoundary,
											  bool onNeumannBoundary, IntersectionPoint<DIM>& intersectionPt,
											  bool& hitDirichlet) -> bool {
		IntersectionPoint<DIM> dirichletIntersectionPt;
		bool intersectedDirichlet = geometricQueries.intersectWithDirichlet(origin, normal, dir, tMax,
																			onDirichletBoundary,
																			dirichletIntersectionPt);

		IntersectionPoint<DIM> neumannIntersectionPt;
		bool intersectedNeumann = geometricQueries.intersectWithNeumann(origin, normal, dir, tMax,
																		onNeumannBoundary,
																		neumannIntersectionPt);

		if (intersectedDirichlet || intersectedNeumann) {
			if (intersectedDirichlet && intersectedNeumann) {
				if (dirichletIntersectionPt.dist < neumannIntersectionPt.dist) {
					intersectionPt = dirichletIntersectionPt;
					hitDirichlet = true;

				} else {
					intersectionPt = neumannIntersectionPt;
					hitDirichlet = false;
				}

			} else if (intersectedDirichlet) {
				intersectionPt = dirichletIntersectionPt;
				hitDirichlet = true;

			} else if (intersectedNeumann) {
				intersectionPt = neumannIntersectionPt;
				hitDirichlet = false;
			}

			return true;
		}

		return false;
	};
	geometricQueries.intersectWithBoundaryAllHits = [&geometricQueries, dirichletAggregate, neumannAggregate](
													 const Vector<DIM>& origin, const Vector<DIM>& normal,
													 const Vector<DIM>& dir, float tMax,
													 bool onDirichletBoundary, bool onNeumannBoundary,
													 std::vector<IntersectionPoint<DIM>>& intersectionPts,
													 std::vector<bool>& hitDirichlet) -> int {
		// clear buffers
		int nIntersections = 0;
		intersectionPts.clear();
		hitDirichlet.clear();

		if (dirichletAggregate != nullptr) {
			// initialize query
			Vector3 queryOrigin = Vector3::Zero();
			Vector3 queryDir = Vector3::Zero();
			queryOrigin.head(DIM) = onDirichletBoundary ?
									geometricQueries.offsetPointAlongDirection(origin, -normal) :
									origin;
			queryDir.head(DIM) = dir;

			// intersect dirichlet boundary
			fcpw::Ray<3> queryRay(queryOrigin, queryDir, tMax);
			std::vector<fcpw::Interaction<3>> queryInteractions;
			int nHits = dirichletAggregate->intersect(queryRay, queryInteractions, false, true);
			nIntersections += nHits;

			for (int i = 0; i < nHits; i++) {
				hitDirichlet.emplace_back(true);
				intersectionPts.emplace_back(IntersectionPoint<DIM>(queryInteractions[i].p.head(DIM),
																	queryInteractions[i].n.head(DIM),
																	queryInteractions[i].d));
			}
		}

		if (neumannAggregate != nullptr) {
			// initialize query
			Vector3 queryOrigin = Vector3::Zero();
			Vector3 queryDir = Vector3::Zero();
			queryOrigin.head(DIM) = onNeumannBoundary ?
									geometricQueries.offsetPointAlongDirection(origin, -normal) :
									origin;
			queryDir.head(DIM) = dir;

			// intersect neumann boundary
			fcpw::Ray<3> queryRay(queryOrigin, queryDir, tMax);
			std::vector<fcpw::Interaction<3>> queryInteractions;
			int nHits = neumannAggregate->intersect(queryRay, queryInteractions, false, true);
			nIntersections += nHits;

			for (int i = 0; i < nHits; i++) {
				hitDirichlet.emplace_back(false);
				intersectionPts.emplace_back(IntersectionPoint<DIM>(queryInteractions[i].p.head(DIM),
																	queryInteractions[i].n.head(DIM),
																	queryInteractions[i].d));
			}
		}

		return nIntersections;
	};
	geometricQueries.sampleNeumann = [neumannAggregate, &neumannSamplingTraversalWeight](
									  const Vector<DIM>& x, float radius, float *randNums,
									  BoundarySample<DIM>& neumannSample) -> bool {
		if (neumannAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			fcpw::BoundingSphere<3> querySphere(queryPt, radius*radius);
			std::vector<fcpw::Interaction<3>> queryInteractions;
			int nHits = neumannAggregate->intersectStochastic(querySphere, queryInteractions, randNums,
															  neumannSamplingTraversalWeight);
			if (nHits < 1) return false;

			neumannSample.pt = queryInteractions[0].p.head(DIM);
			neumannSample.normal = queryInteractions[0].n.head(DIM);
			neumannSample.pdf = queryInteractions[0].d;

			return true;
		}

		return false;
	};
	geometricQueries.computeStarRadius = [neumannAggregate](const Vector<DIM>& x, float minRadius,
															float maxRadius, float silhouettePrecision,
															bool flipNormalOrientation) -> float {
		if (minRadius > maxRadius) return maxRadius;
		if (neumannAggregate != nullptr) {
			Vector3 queryPt = Vector3::Zero();
			queryPt.head(DIM) = x;

			bool flipNormals = true; // FCPW's internal convention requires normals to be flipped
			if (flipNormalOrientation) flipNormals = !flipNormals;

			float squaredSphereRadius = maxRadius < fcpw::maxFloat ? maxRadius*maxRadius : fcpw::maxFloat;
			fcpw::Interaction<3> interaction;
			fcpw::BoundingSphere<3> querySphere(queryPt, squaredSphereRadius);
			bool found = neumannAggregate->findClosestSilhouettePoint(querySphere, interaction, flipNormals,
																	  minRadius*minRadius, silhouettePrecision);
			if (found) return std::max(interaction.d, minRadius);
		}

		return std::max(maxRadius, minRadius);
	};
	geometricQueries.insideDomain = [&geometricQueries](const Vector<DIM>& x) -> bool {
		if (!geometricQueries.domainIsWatertight) return true;
		float d1 = geometricQueries.computeDistToDirichlet(x, true);
		float d2 = geometricQueries.computeDistToNeumann(x, true);

		return std::fabs(d1) < std::fabs(d2) ? d1 < 0.0f : d2 < 0.0f;
	};
	geometricQueries.outsideBoundingDomain = [&boundingBox](const Vector<DIM>& x) -> bool {
		return !boundingBox.contains(x);
	};
}

} // zombie

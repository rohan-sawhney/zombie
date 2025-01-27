// This file provides utility functions to load 2D or 3D boundary meshes from OBJ files,
// normalize mesh positions to lie within a unit sphere, swap mesh indices to flip orientation,
// and compute the bounding box of a mesh. The FcpwBoundaryHandler class builds an acceleration
// structure to perform geometric queries against a mesh, while the 'populateGeometricQueriesForBoundary'
// functions populate the GeometricQueries structure using FcpwBoundaryHandler objects for the
// absorbing (Dirichlet) and reflecting (Neumann or Robin) boundaries.

#pragma once

#include <zombie/core/geometric_queries.h>
#include <zombie/core/distributions.h>
#include <cmath>
#include <fcpw/utilities/scene_loader.h>
#include <zombie/utils/reflectance_boundary_bvh/baseline.h>
#ifdef FCPW_USE_ENOKI
    #include <zombie/utils/reflectance_boundary_bvh/mbvh.h>
#else
    #include <zombie/utils/reflectance_boundary_bvh/bvh.h>
#endif
#include <zombie/utils/reflectance_boundary_bvh/robin_bounds.h>

namespace zombie {

// loads 2D or 3D boundary mesh from OBJ file
template <size_t DIM>
void loadBoundaryMesh(const std::string& objFile,
                      std::vector<Vector<DIM>>& positions,
                      std::vector<Vectori<DIM>>& indices);
template <size_t DIM>
void loadTexturedBoundaryMesh(const std::string& objFile,
                              std::vector<Vector<DIM>>& positions,
                              std::vector<Vector<DIM-1>>& textureCoordinates,
                              std::vector<Vectori<DIM>>& indices,
                              std::vector<Vectori<DIM>>& textureIndices);

// mesh utility functions
template <size_t DIM>
void normalize(std::vector<Vector<DIM>>& positions);

template <size_t DIM>
void flipOrientation(std::vector<Vectori<DIM>>& indices);

template <size_t DIM>
std::pair<Vector<DIM>, Vector<DIM>> computeBoundingBox(const std::vector<Vector<DIM>>& positions,
                                                       bool makeSquare, float scale);

template <size_t DIM>
void addBoundingBoxToBoundaryMesh(const Vector<DIM>& boundingBoxMin,
                                  const Vector<DIM>& boundingBoxMax,
                                  std::vector<Vector<DIM>>& positions,
                                  std::vector<Vectori<DIM>>& indices);

// partitions boundary mesh into absorbing and reflecting parts using primitive centroids---
// this assumes the boundary discretization is perfectly adapted to the boundary conditions,
// which isn't always a correct assumption
template <size_t DIM>
void partitionBoundaryMesh(std::function<bool(const Vector<DIM>&)> onReflectingBoundary,
                           const std::vector<Vector<DIM>>& positions,
                           const std::vector<Vectori<DIM>>& indices,
                           std::vector<Vector<DIM>>& absorbingPositions,
                           std::vector<Vectori<DIM>>& absorbingIndices,
                           std::vector<Vector<DIM>>& reflectingPositions,
                           std::vector<Vectori<DIM>>& reflectingIndices);

// Helper classes to build an acceleration structure to perform geometric queries such as
// ray intersections, closest points, etc. against a boundary mesh with Dirichlet, Neumann
// and Robin conditions.
template <size_t DIM>
class FcpwDirichletBoundaryHandler {
public:
    // constructor
    FcpwDirichletBoundaryHandler();

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. Uses a simple list of mesh faces for brute-force geometric
    // queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                    const std::vector<Vectori<DIM>>& indices,
                                    bool buildBvh=true, bool enableBvhVectorization=false);
};

template <size_t DIM>
class FcpwNeumannBoundaryHandler {
public:
    // constructor
    FcpwNeumannBoundaryHandler();

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. Uses a simple list of mesh faces for brute-force geometric
    // queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                    const std::vector<Vectori<DIM>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette,
                                    bool buildBvh=true, bool enableBvhVectorization=false);
};

template <size_t DIM>
class FcpwRobinBoundaryHandler {
public:
    // constructor
    FcpwRobinBoundaryHandler();

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions, indices, and min and max coefficients per mesh face. Uses a simple
    // list of mesh faces for brute-force geometric queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                    const std::vector<Vectori<DIM>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette,
                                    const std::vector<float>& minRobinCoeffValues,
                                    const std::vector<float>& maxRobinCoeffValues,
                                    bool buildBvh=true, bool enableBvhVectorization=false);

    // updates the Robin coefficients on the boundary mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues);
};

std::function<bool(float, int)> getIgnoreCandidateSilhouetteCallback(bool solveDoubleSided = false,
                                                                     float silhouettePrecision = 1e-3f);

// populates the GeometricQueries structure
template <size_t DIM>
void populateGeometricQueriesForDirichletBoundary(FcpwDirichletBoundaryHandler<DIM>& dirichletBoundaryHandler,
                                                  GeometricQueries<DIM>& geometricQueries);
template <size_t DIM>
void populateGeometricQueriesForNeumannBoundary(FcpwNeumannBoundaryHandler<DIM>& neumannBoundaryHandler,
                                                std::function<float(float)> branchTraversalWeight,
                                                GeometricQueries<DIM>& geometricQueries);
template <size_t DIM>
void populateGeometricQueriesForRobinBoundary(FcpwRobinBoundaryHandler<DIM>& robinBoundaryHandler,
                                              std::function<float(float)> branchTraversalWeight,
                                              GeometricQueries<DIM>& geometricQueries);

std::function<float(float)> getBranchTraversalWeightCallback(float minRadialDist = 1e-2f);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE: BVH refit

template <size_t DIM>
void loadBoundaryMesh(const std::string& objFile,
                      std::vector<Vector<DIM>>& positions,
                      std::vector<Vectori<DIM>>& indices)
{
    std::cerr << "loadBoundaryMesh: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
void loadBoundaryMesh<2>(const std::string& objFile,
                         std::vector<Vector2>& positions,
                         std::vector<Vector2i>& indices)
{
    // load file
    fcpw::PolygonSoup<2> soup;
    fcpw::loadLineSegmentSoupFromOBJFile(objFile, soup);

    // collect mesh positions and indices
    positions = soup.positions;
    indices.clear();
    int L = (int)soup.indices.size()/2;

    for (int l = 0; l < L; l++) {
        size_t i = soup.indices[2*l + 0];
        size_t j = soup.indices[2*l + 1];

        indices.emplace_back(Vector2i(i, j));
    }
}

template <>
void loadBoundaryMesh<3>(const std::string& objFile,
                         std::vector<Vector3>& positions,
                         std::vector<Vector3i>& indices)
{
    // load file
    fcpw::PolygonSoup<3> soup;
    fcpw::loadTriangleSoupFromOBJFile(objFile, soup);

    // collect mesh positions and indices
    positions = soup.positions;
    indices.clear();
    int T = (int)soup.indices.size()/3;

    for (int t = 0; t < T; t++) {
        size_t i = soup.indices[3*t + 0];
        size_t j = soup.indices[3*t + 1];
        size_t k = soup.indices[3*t + 2];

        indices.emplace_back(Vector3i(i, j, k));
    }
}

template <size_t DIM>
void loadTexturedBoundaryMesh(const std::string& objFile,
                              std::vector<Vector<DIM>>& positions,
                              std::vector<Vector<DIM-1>>& textureCoordinates,
                              std::vector<Vectori<DIM>>& indices,
                              std::vector<Vectori<DIM>>& textureIndices)
{
    std::cerr << "loadTexturedBoundaryMesh: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
void loadTexturedBoundaryMesh<3>(const std::string& objFile,
                                 std::vector<Vector3>& positions,
                                 std::vector<Vector2>& textureCoordinates,
                                 std::vector<Vector3i>& indices,
                                 std::vector<Vector3i>& textureIndices)
{
    // load file
    fcpw::PolygonSoup<3> soup;
    fcpw::loadTriangleSoupFromOBJFile(objFile, soup);

    // collect mesh positions, texture coordinates and indices
    positions = soup.positions;
    textureCoordinates = soup.textureCoordinates;
    indices.clear();
    textureIndices.clear();
    int T = (int)soup.indices.size()/3;

    for (int t = 0; t < T; t++) {
        size_t i = soup.indices[3*t + 0];
        size_t j = soup.indices[3*t + 1];
        size_t k = soup.indices[3*t + 2];
        size_t ti = soup.tIndices[3*t + 0];
        size_t tj = soup.tIndices[3*t + 1];
        size_t tk = soup.tIndices[3*t + 2];

        indices.emplace_back(Vector3i(i, j, k));
        textureIndices.emplace_back(Vector3i(ti, tj, tk));
    }
}

template <size_t DIM>
void normalize(std::vector<Vector<DIM>>& positions)
{
    int V = (int)positions.size();
    Vector<DIM> cm = Vector<DIM>::Zero();
    for (int i = 0; i < V; i++) {
        cm += positions[i];
    }

    cm /= V;
    float radius = 0.0f;
    for (int i = 0; i < V; i++) {
        positions[i] -= cm;
        radius = std::max(radius, positions[i].norm());
    }

    for (int i = 0; i < V; i++) {
        positions[i] /= radius;
    }
}

template <size_t DIM>
void flipOrientation(std::vector<Vectori<DIM>>& indices)
{
    for (int i = 0; i < (int)indices.size(); i++) {
        std::swap(indices[i][0], indices[i][1]);
    }
}

template <size_t DIM>
std::pair<Vector<DIM>, Vector<DIM>> computeBoundingBox(const std::vector<Vector<DIM>>& positions,
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

    return std::make_pair(bbox.pMin, bbox.pMax);
}

template <size_t DIM>
void buildBoundingBoxMesh(const Vector<DIM>& boundingBoxMin,
                          const Vector<DIM>& boundingBoxMax,
                          std::vector<Vector<DIM>>& positions,
                          std::vector<Vectori<DIM>>& indices)
{
    // do nothing
}

template <>
void buildBoundingBoxMesh<2>(const Vector2& boundingBoxMin,
                             const Vector2& boundingBoxMax,
                             std::vector<Vector2>& positions,
                             std::vector<Vector2i>& indices)
{
    positions.clear();
    positions.emplace_back(boundingBoxMin);
    positions.emplace_back(Vector2(boundingBoxMax(0), boundingBoxMin(1)));
    positions.emplace_back(boundingBoxMax);
    positions.emplace_back(Vector2(boundingBoxMin(0), boundingBoxMax(1)));

    indices.clear();
    indices.emplace_back(Vector2i(0, 1));
    indices.emplace_back(Vector2i(1, 2));
    indices.emplace_back(Vector2i(2, 3));
    indices.emplace_back(Vector2i(3, 0));
}

template <>
void buildBoundingBoxMesh<3>(const Vector3& boundingBoxMin,
                             const Vector3& boundingBoxMax,
                             std::vector<Vector3>& positions,
                             std::vector<Vector3i>& indices)
{
    positions.clear();
    positions.emplace_back(boundingBoxMin);
    positions.emplace_back(Vector3(boundingBoxMax(0), boundingBoxMin(1), boundingBoxMin(2)));
    positions.emplace_back(Vector3(boundingBoxMax(0), boundingBoxMax(1), boundingBoxMin(2)));
    positions.emplace_back(Vector3(boundingBoxMin(0), boundingBoxMax(1), boundingBoxMin(2)));
    positions.emplace_back(Vector3(boundingBoxMin(0), boundingBoxMin(1), boundingBoxMax(2)));
    positions.emplace_back(Vector3(boundingBoxMax(0), boundingBoxMin(1), boundingBoxMax(2)));
    positions.emplace_back(boundingBoxMax);
    positions.emplace_back(Vector3(boundingBoxMin(0), boundingBoxMax(1), boundingBoxMax(2)));

    indices.clear();
    indices.emplace_back(Vector3i(0, 2, 1));
    indices.emplace_back(Vector3i(0, 3, 2));
    indices.emplace_back(Vector3i(0, 5, 4));
    indices.emplace_back(Vector3i(0, 1, 5));
    indices.emplace_back(Vector3i(0, 7, 3));
    indices.emplace_back(Vector3i(0, 4, 7));
    indices.emplace_back(Vector3i(6, 7, 4));
    indices.emplace_back(Vector3i(6, 4, 5));
    indices.emplace_back(Vector3i(6, 5, 1));
    indices.emplace_back(Vector3i(6, 1, 2));
    indices.emplace_back(Vector3i(6, 2, 3));
    indices.emplace_back(Vector3i(6, 3, 7));
}

template <size_t DIM>
void addBoundingBoxToBoundaryMesh(const Vector<DIM>& boundingBoxMin,
                                  const Vector<DIM>& boundingBoxMax,
                                  std::vector<Vector<DIM>>& positions,
                                  std::vector<Vectori<DIM>>& indices)
{
    // build box
    std::vector<Vector<DIM>> boxPositions;
    std::vector<Vectori<DIM>> boxIndices;
    buildBoundingBoxMesh<DIM>(boundingBoxMin, boundingBoxMax, boxPositions, boxIndices);

    // append box positions and indices
    int V = (int)positions.size();
    for (int i = 0; i < (int)boxPositions.size(); i++) {
        positions.emplace_back(boxPositions[i]);
    }

    for (int i = 0; i < (int)boxIndices.size(); i++) {
        Vectori<DIM> boxIndex = boxIndices[i];
        for (int j = 0; j < DIM; j++) {
            boxIndex[j] += V;
        }

        indices.emplace_back(boxIndex);
    }
}

template <size_t DIM>
Vector<DIM> computePrimitiveMidpoint(const std::vector<Vector<DIM>>& positions,
                                     const std::vector<Vectori<DIM>>& indices,
                                     size_t primitiveIndex)
{
    Vector<DIM> pMid = Vector<DIM>::Zero();
    for (int j = 0; j < DIM; j++) {
        int vIndex = indices[primitiveIndex][j];
        const Vector<DIM>& p = positions[vIndex];

        pMid += p;
    }

    return pMid/DIM;
}

template <size_t DIM>
void partitionBoundaryMesh(std::function<bool(const Vector<DIM>&)> onReflectingBoundary,
                           const std::vector<Vector<DIM>>& positions,
                           const std::vector<Vectori<DIM>>& indices,
                           std::vector<Vector<DIM>>& absorbingPositions,
                           std::vector<Vectori<DIM>>& absorbingIndices,
                           std::vector<Vector<DIM>>& reflectingPositions,
                           std::vector<Vectori<DIM>>& reflectingIndices)
{
    Vectori<DIM> index = Vectori<DIM>::Constant(-1);
    std::unordered_map<size_t, size_t> absorbingBoundaryMap, reflectingBoundaryMap;
    absorbingPositions.clear();
    absorbingIndices.clear();
    reflectingPositions.clear();
    reflectingIndices.clear();

    for (int i = 0; i < (int)indices.size(); i++) {
        Vector<DIM> pMid = computePrimitiveMidpoint<DIM>(positions, indices, i);

        if (onReflectingBoundary(pMid)) {
            for (int j = 0; j < DIM; j++) {
                int vIndex = indices[i][j];
                const Vector<DIM>& p = positions[vIndex];

                if (reflectingBoundaryMap.find(vIndex) == reflectingBoundaryMap.end()) {
                    reflectingBoundaryMap[vIndex] = reflectingPositions.size();
                    reflectingPositions.emplace_back(p);
                }

                index[j] = reflectingBoundaryMap[vIndex];
            }

            reflectingIndices.emplace_back(index);

        } else {
            for (int j = 0; j < DIM; j++) {
                int vIndex = indices[i][j];
                const Vector<DIM>& p = positions[vIndex];

                if (absorbingBoundaryMap.find(vIndex) == absorbingBoundaryMap.end()) {
                    absorbingBoundaryMap[vIndex] = absorbingPositions.size();
                    absorbingPositions.emplace_back(p);
                }

                index[j] = absorbingBoundaryMap[vIndex];
            }

            absorbingIndices.emplace_back(index);
        }
    }
}

template <size_t DIM>
FcpwDirichletBoundaryHandler<DIM>::FcpwDirichletBoundaryHandler()
{
    std::cerr << "FcpwDirichletBoundaryHandler: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t DIM>
void FcpwDirichletBoundaryHandler<DIM>::buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                                                   const std::vector<Vectori<DIM>>& indices,
                                                                   bool buildBvh, bool enableBvhVectorization)
{
    std::cerr << "FcpwDirichletBoundaryHandler::buildAccelerationStructure: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
class FcpwDirichletBoundaryHandler<2> {
public:
    // constructor
    FcpwDirichletBoundaryHandler() {}

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. Uses a simple list of mesh faces for brute-force geometric
    // queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector2>& positions,
                                    const std::vector<Vector2i>& indices,
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
        if (positions.size() > 0) {
            // load positions and indices
            scene.setObjectCount(1);
            scene.setObjectVertices(positions, 0);
            scene.setObjectLineSegments(indices, 0);
            
            // build aggregate
            fcpw::AggregateType aggregateType = buildBvh ?
                                                fcpw::AggregateType::Bvh_SurfaceArea :
                                                fcpw::AggregateType::Baseline;
            scene.build(aggregateType, enableBvhVectorization, true, true);
        }
    }

    // member
    fcpw::Scene<2> scene;
};

template <>
class FcpwDirichletBoundaryHandler<3> {
public:
    // constructor
    FcpwDirichletBoundaryHandler() {}

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. Uses a simple list of mesh faces for brute-force geometric
    // queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector3>& positions,
                                    const std::vector<Vector3i>& indices,
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
        if (positions.size() > 0) {
            // load positions and indices
            scene.setObjectCount(1);
            scene.setObjectVertices(positions, 0);
            scene.setObjectTriangles(indices, 0);

            // build aggregate
            fcpw::AggregateType aggregateType = buildBvh ?
                                                fcpw::AggregateType::Bvh_SurfaceArea :
                                                fcpw::AggregateType::Baseline;
            scene.build(aggregateType, enableBvhVectorization, true, true);
        }
    }

    // member
    fcpw::Scene<3> scene;
};

template <size_t DIM>
FcpwNeumannBoundaryHandler<DIM>::FcpwNeumannBoundaryHandler()
{
    std::cerr << "FcpwNeumannBoundaryHandler: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t DIM>
void FcpwNeumannBoundaryHandler<DIM>::buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                                                 const std::vector<Vectori<DIM>>& indices,
                                                                 std::function<bool(float, int)> ignoreCandidateSilhouette,
                                                                 bool buildBvh, bool enableBvhVectorization)
{
    std::cerr << "FcpwNeumannBoundaryHandler::buildAccelerationStructure: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
class FcpwNeumannBoundaryHandler<2> {
public:
    // constructor
    FcpwNeumannBoundaryHandler() {}

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. Uses a simple list of mesh faces for brute-force geometric
    // queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector2>& positions,
                                    const std::vector<Vector2i>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette,
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
        if (positions.size() > 0) {
            // load positions and indices
            scene.setObjectCount(1);
            scene.setObjectVertices(positions, 0);
            scene.setObjectLineSegments(indices, 0);

            // compute silhouettes
            scene.computeSilhouettes(ignoreCandidateSilhouette);
            
            // build aggregate
            fcpw::AggregateType aggregateType = buildBvh ?
                                                fcpw::AggregateType::Bvh_SurfaceArea :
                                                fcpw::AggregateType::Baseline;
            scene.build(aggregateType, enableBvhVectorization, true, true);
        }
    }

    // member
    fcpw::Scene<2> scene;
};

template <>
class FcpwNeumannBoundaryHandler<3> {
public:
    // constructor
    FcpwNeumannBoundaryHandler() {}

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. Uses a simple list of mesh faces for brute-force geometric
    // queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector3>& positions,
                                    const std::vector<Vector3i>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette,
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
        if (positions.size() > 0) {
            // load positions and indices
            scene.setObjectCount(1);
            scene.setObjectVertices(positions, 0);
            scene.setObjectTriangles(indices, 0);

            // compute silhouettes
            scene.computeSilhouettes(ignoreCandidateSilhouette);
            
            // build aggregate
            fcpw::AggregateType aggregateType = buildBvh ?
                                                fcpw::AggregateType::Bvh_SurfaceArea :
                                                fcpw::AggregateType::Baseline;
            scene.build(aggregateType, enableBvhVectorization, true, true);
        }
    }

    // member
    fcpw::Scene<3> scene;
};

template <size_t DIM>
FcpwRobinBoundaryHandler<DIM>::FcpwRobinBoundaryHandler()
{
    std::cerr << "FcpwRobinBoundaryHandler: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t DIM>
void FcpwRobinBoundaryHandler<DIM>::buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                                               const std::vector<Vectori<DIM>>& indices,
                                                               std::function<bool(float, int)> ignoreCandidateSilhouette,
                                                               const std::vector<float>& minRobinCoeffValues,
                                                               const std::vector<float>& maxRobinCoeffValues,
                                                               bool buildBvh, bool enableBvhVectorization)
{
    std::cerr << "FcpwRobinBoundaryHandler::buildAccelerationStructure: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t DIM>
void FcpwRobinBoundaryHandler<DIM>::updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                                            const std::vector<float>& maxRobinCoeffValues)
{
    std::cerr << "FcpwRobinBoundaryHandler::updateRobinCoefficients: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
class FcpwRobinBoundaryHandler<2> {
public:
    // constructor
    FcpwRobinBoundaryHandler() {
        baseline = nullptr;
        bvh = nullptr;
#ifdef FCPW_USE_ENOKI
        mbvh = nullptr;
#endif
    }

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions, indices, and min and max coefficients per mesh face. Uses a simple
    // list of mesh faces for brute-force geometric queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector2>& positions,
                                    const std::vector<Vector2i>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette,
                                    const std::vector<float>& minRobinCoeffValues,
                                    const std::vector<float>& maxRobinCoeffValues,
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
        if (positions.size() > 0) {
            struct VertexFaceAdjacency {
                VertexFaceAdjacency(): adjacentFaceIndices{-1, -1} {}
                int adjacentFaceIndices[2];
            };

            // set the vertex and line segment count
            int V = (int)positions.size();
            int L = (int)indices.size();
            if (minRobinCoeffValues.size() != L || maxRobinCoeffValues.size() != L) {
                std::cerr << "FcpwRobinBoundaryHandler<2>::buildAccelerationStructure: invalid Robin coefficient sizes!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<VertexFaceAdjacency> vertexTable(V);
            soup.positions = positions;
            soup.indices.resize(2*L);
            lineSegments.resize(L);
            lineSegmentPtrs.resize(L, nullptr);

            // update soup and line segment indices
            for (int i = 0; i < L; i++) {
                ReflectanceLineSegment<PrimitiveBound>& lineSegment = lineSegments[i];
                lineSegmentPtrs[i] = &lineSegment;
                lineSegment.soup = &soup;
                lineSegment.setIndex(i);
                lineSegment.minReflectanceCoeff = minRobinCoeffValues[i];
                lineSegment.maxReflectanceCoeff = maxRobinCoeffValues[i];

                for (int j = 0; j < 2; j++) {
                    int vIndex = (int)indices[i][j];
                    VertexFaceAdjacency& v = vertexTable[vIndex];
                    v.adjacentFaceIndices[1 - j] = i;

                    soup.indices[2*i + j] = vIndex;
                    lineSegment.indices[j] = vIndex;
                }
            }

            // compute adjacent normals for line segment primitives
            for (int i = 0; i < L; i++) {
                ReflectanceLineSegment<PrimitiveBound>& lineSegment = lineSegments[i];
                Vector2 n0 = lineSegment.normal(true);

                for (int j = 0; j < 2; j++) {
                    int vIndex = (int)indices[i][j];
                    const VertexFaceAdjacency& v = vertexTable[vIndex];
                    int adjacentFaceIndex = v.adjacentFaceIndices[j];

                    if (adjacentFaceIndex != -1) {
                        lineSegment.hasAdjacentFace[j] = true;
                        lineSegment.n[j] = lineSegments[adjacentFaceIndex].normal(true);

                    } else {
                        lineSegment.hasAdjacentFace[j] = false;
                    }

                    if (ignoreCandidateSilhouette && lineSegment.hasAdjacentFace[j]) {
                        const Vector2& nj = lineSegment.n[j];
                        float det = n0[0]*nj[1] - n0[1]*nj[0];
                        float sign = j == 0 ? 1.0f : -1.0f;

                        lineSegment.ignoreAdjacentFace[j] = ignoreCandidateSilhouette(det*sign, lineSegment.getIndex());

                    } else {
                        lineSegment.ignoreAdjacentFace[j] = false;
                    }
                }
            }

            // build aggregate
            if (buildBvh) {
                if (enableBvhVectorization) {
#ifdef FCPW_USE_ENOKI
                    bvh = createReflectanceBvh<2, ReflectanceLineSegment<PrimitiveBound>, NodeBound>(soup, lineSegmentPtrs, silhouettePtrsStub,
                                                                                                     true, true, FCPW_SIMD_WIDTH);
                    mbvh = createVectorizedReflectanceBvh<2, ReflectanceLineSegment<PrimitiveBound>, WideNodeBound, NodeBound>(
                                                                                                     bvh.get(), lineSegmentPtrs,
                                                                                                     silhouettePtrsStub, true);
#else
                    bvh = createReflectanceBvh<2, ReflectanceLineSegment<PrimitiveBound>, NodeBound>(soup, lineSegmentPtrs, silhouettePtrsStub);
#endif
                } else {
                    bvh = createReflectanceBvh<2, ReflectanceLineSegment<PrimitiveBound>, NodeBound>(soup, lineSegmentPtrs, silhouettePtrsStub);
                }

            } else {
                baseline = createReflectanceBaseline<2, ReflectanceLineSegment<PrimitiveBound>>(lineSegmentPtrs, silhouettePtrsStub);
            }
        }
    }

    // updates the Robin coefficients on the boundary mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues) {
        if (baseline) {
            baseline->updateReflectanceCoefficients(minRobinCoeffValues, maxRobinCoeffValues);

#ifdef FCPW_USE_ENOKI
        } else if (mbvh) {
            mbvh->updateReflectanceCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
#endif
        } else if (bvh) {
            bvh->updateReflectanceCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
        }
    }

    // members
    typedef RobinLineSegmentBound PrimitiveBound;
    typedef RobinBvhNodeBound<2> NodeBound;
    std::unique_ptr<ReflectanceBaseline<2, ReflectanceLineSegment<PrimitiveBound>>> baseline;
    std::unique_ptr<ReflectanceBvh<2, ReflectanceBvhNode<2>, ReflectanceLineSegment<PrimitiveBound>, NodeBound>> bvh;
#ifdef FCPW_USE_ENOKI
    typedef RobinMbvhNodeBound<2> WideNodeBound;
    std::unique_ptr<ReflectanceMbvh<FCPW_SIMD_WIDTH, 2,
                                    ReflectanceLineSegment<PrimitiveBound>,
                                    ReflectanceMbvhNode<2>, WideNodeBound>> mbvh;
#endif
    PolygonSoup<2> soup;
    std::vector<ReflectanceLineSegment<PrimitiveBound>> lineSegments;
    std::vector<ReflectanceLineSegment<PrimitiveBound> *> lineSegmentPtrs;
    std::vector<fcpw::SilhouettePrimitive<2> *> silhouettePtrsStub;
};

template <>
class FcpwRobinBoundaryHandler<3> {
public:
    // constructor
    FcpwRobinBoundaryHandler() {
        baseline = nullptr;
        bvh = nullptr;
#ifdef FCPW_USE_ENOKI
        mbvh = nullptr;
#endif
    }

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions, indices, and min and max coefficients per mesh face. Uses a simple
    // list of mesh faces for brute-force geometric queries when buildBvh is false.
    void buildAccelerationStructure(const std::vector<Vector3>& positions,
                                    const std::vector<Vector3i>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette,
                                    const std::vector<float>& minRobinCoeffValues,
                                    const std::vector<float>& maxRobinCoeffValues,
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
        if (positions.size() > 0) {
            struct EdgeFaceAdjacency {
                EdgeFaceAdjacency(): adjacentFaceIndices{-1, -1} {}
                int adjacentFaceIndices[2];
            };

            // set the vertex and triangle count
            int V = (int)positions.size();
            int T = (int)indices.size();
            if (minRobinCoeffValues.size() != T || maxRobinCoeffValues.size() != T) {
                std::cerr << "FcpwRobinBoundaryHandler<3>::buildAccelerationStructure: invalid Robin coefficient sizes!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::map<std::pair<int, int>, EdgeFaceAdjacency> edgeTable;
            soup.positions = positions;
            soup.indices.resize(3*T);
            triangles.resize(T);
            trianglePtrs.resize(T, nullptr);

            // update soup and triangle indices
            for (int i = 0; i < T; i++) {
                ReflectanceTriangle<PrimitiveBound>& triangle = triangles[i];
                trianglePtrs[i] = &triangle;
                triangle.soup = &soup;
                triangle.setIndex(i);
                triangle.minReflectanceCoeff = minRobinCoeffValues[i];
                triangle.maxReflectanceCoeff = maxRobinCoeffValues[i];

                for (int j = 0; j < 3; j++) {
                    int k = (j + 1)%3;
                    int I = (int)indices[i][j];
                    int J = (int)indices[i][k];
                    bool performedSwap = false;
                    if (I > J) {
                        std::swap(I, J);
                        performedSwap = true;
                    }

                    std::pair<int, int> vIndices(I, J);
                    if (edgeTable.find(vIndices) == edgeTable.end()) {
                        EdgeFaceAdjacency e;
                        e.adjacentFaceIndices[performedSwap ? 1 : 0] = i;
                        edgeTable[vIndices] = e;

                    } else {
                        EdgeFaceAdjacency& e = edgeTable[vIndices];
                        e.adjacentFaceIndices[performedSwap ? 1 : 0] = i;
                    }

                    soup.indices[3*i + j] = (int)indices[i][j];
                    triangle.indices[j] = (int)indices[i][j];
                }
            }

            // compute adjacent normals for triangle primitives
            for (int i = 0; i < T; i++) {
                ReflectanceTriangle<PrimitiveBound>& triangle = triangles[i];
                Vector3 n0 = triangle.normal(true);

                for (int j = 0; j < 3; j++) {
                    int k = (j + 1)%3;
                    int I = (int)indices[i][j];
                    int J = (int)indices[i][k];
                    bool performedSwap = false;
                    if (I > J) {
                        std::swap(I, J);
                        performedSwap = true;
                    }

                    std::pair<int, int> vIndices(I, J);
                    const EdgeFaceAdjacency& e = edgeTable[vIndices];
                    int adjacentFaceIndex = e.adjacentFaceIndices[performedSwap ? 0 : 1];

                    if (adjacentFaceIndex != -1) {
                        triangle.hasAdjacentFace[j] = true;
                        triangle.n[j] = triangles[adjacentFaceIndex].normal(true);

                    } else {
                        triangle.hasAdjacentFace[j] = false;
                    }

                    if (ignoreCandidateSilhouette && triangle.hasAdjacentFace[j]) {
                        const Vector3& pa = soup.positions[indices[i][j]];
                        const Vector3& pb = soup.positions[indices[i][k]];
                        const Vector3& nj = triangle.n[j];
                        Vector3 edgeDir = (pb - pa).normalized();

                        float dihedralAngle = std::atan2(edgeDir.dot(nj.cross(n0)), n0.dot(nj));
                        triangle.ignoreAdjacentFace[j] = ignoreCandidateSilhouette(dihedralAngle, triangle.getIndex());

                    } else {
                        triangle.ignoreAdjacentFace[j] = false;
                    }
                }
            }

            // build aggregate
            if (buildBvh) {
                if (enableBvhVectorization) {
#ifdef FCPW_USE_ENOKI
                    bvh = createReflectanceBvh<3, ReflectanceTriangle<PrimitiveBound>, NodeBound>(soup, trianglePtrs, silhouettePtrsStub,
                                                                                                  true, true, FCPW_SIMD_WIDTH);
                    mbvh = createVectorizedReflectanceBvh<3, ReflectanceTriangle<PrimitiveBound>, WideNodeBound, NodeBound>(
                                                                                                  bvh.get(), trianglePtrs,
                                                                                                  silhouettePtrsStub, true);
#else
                    bvh = createReflectanceBvh<3, ReflectanceTriangle<PrimitiveBound>, NodeBound>(soup, trianglePtrs, silhouettePtrsStub);
#endif
                } else {
                    bvh = createReflectanceBvh<3, ReflectanceTriangle<PrimitiveBound>, NodeBound>(soup, trianglePtrs, silhouettePtrsStub);
                }

            } else {
                baseline = createReflectanceBaseline<3, ReflectanceTriangle<PrimitiveBound>>(trianglePtrs, silhouettePtrsStub);
            }
        }
    }

    // updates the Robin coefficients on the boundary mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues) {
        if (baseline) {
            baseline->updateReflectanceCoefficients(minRobinCoeffValues, maxRobinCoeffValues);

#ifdef FCPW_USE_ENOKI
        } else if (mbvh) {
            mbvh->updateReflectanceCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
#endif
        } else if (bvh) {
            bvh->updateReflectanceCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
        }
    }

    // members
    typedef RobinTriangleBound PrimitiveBound;
    typedef RobinBvhNodeBound<3> NodeBound;
    std::unique_ptr<ReflectanceBaseline<3, ReflectanceTriangle<PrimitiveBound>>> baseline;
    std::unique_ptr<ReflectanceBvh<3, ReflectanceBvhNode<3>, ReflectanceTriangle<PrimitiveBound>, NodeBound>> bvh;
#ifdef FCPW_USE_ENOKI
    typedef RobinMbvhNodeBound<3> WideNodeBound;
    std::unique_ptr<ReflectanceMbvh<FCPW_SIMD_WIDTH, 3,
                                    ReflectanceTriangle<PrimitiveBound>,
                                    ReflectanceMbvhNode<3>, WideNodeBound>> mbvh;
#endif
    PolygonSoup<3> soup;
    std::vector<ReflectanceTriangle<PrimitiveBound>> triangles;
    std::vector<ReflectanceTriangle<PrimitiveBound> *> trianglePtrs;
    std::vector<fcpw::SilhouettePrimitive<3> *> silhouettePtrsStub;
};

std::function<bool(float, int)> getIgnoreCandidateSilhouetteCallback(bool solveDoubleSided,
                                                                     float silhouettePrecision) {
    return [solveDoubleSided, silhouettePrecision](float dihedralAngle, int index) -> bool {
        // ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
        // NOTE: for complex scenes with both open and closed meshes, the primitive index argument
        // (of an adjacent line segment/triangle in the scene) can be used to determine whether a
        // vertex/edge should be ignored as a candidate for silhouette tests.
        return solveDoubleSided ? false : dihedralAngle < silhouettePrecision;
    };
}

template <size_t DIM>
void populateGeometricQueriesForDirichletBoundary(FcpwDirichletBoundaryHandler<DIM>& dirichletBoundaryHandler,
                                                  GeometricQueries<DIM>& geometricQueries)
{
    fcpw::Aggregate<DIM> *absorbingBoundaryAggregate = dirichletBoundaryHandler.scene.getSceneData()->aggregate.get();
    if (absorbingBoundaryAggregate) {
        geometricQueries.hasAbsorbingBoundary = true;
        geometricQueries.computeDistToAbsorbingBoundary = [absorbingBoundaryAggregate](
                                                          const Vector<DIM>& x, bool computeSignedDistance) -> float {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            absorbingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            return computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;
        };
        geometricQueries.projectToAbsorbingBoundary = [absorbingBoundaryAggregate](
                                                      Vector<DIM>& x, Vector<DIM>& normal,
                                                      float& distance, bool computeSignedDistance) -> bool {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            absorbingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            x = interaction.p;
            normal = interaction.n;
            distance = computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;

            return true;
        };
        geometricQueries.intersectAbsorbingBoundary = [&geometricQueries, absorbingBoundaryAggregate](
                                                      const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                      const Vector<DIM>& dir, float tMax, bool onAborbingBoundary,
                                                      IntersectionPoint<DIM>& intersectionPt) -> bool {
            Vector<DIM> queryOrigin = onAborbingBoundary ?
                                      geometricQueries.offsetPointAlongDirection(origin, -normal) :
                                      origin;
            Vector<DIM> queryDir = dir;
            fcpw::Ray<DIM> queryRay(queryOrigin, queryDir, tMax);
            fcpw::Interaction<DIM> queryInteraction;
            bool hit = absorbingBoundaryAggregate->intersect(queryRay, queryInteraction, false);
            if (!hit) return false;

            intersectionPt.pt = queryInteraction.p;
            intersectionPt.normal = queryInteraction.n;
            intersectionPt.dist = queryInteraction.d;

            return true;
        };
        geometricQueries.intersectAbsorbingBoundaryAllHits = [&geometricQueries, absorbingBoundaryAggregate](
                                                             const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                             const Vector<DIM>& dir, float tMax, bool onAborbingBoundary,
                                                             std::vector<IntersectionPoint<DIM>>& intersectionPts) -> int {
            Vector<DIM> queryOrigin = onAborbingBoundary ?
                                      geometricQueries.offsetPointAlongDirection(origin, -normal) :
                                      origin;
            Vector<DIM> queryDir = dir;
            fcpw::Ray<DIM> queryRay(queryOrigin, queryDir, tMax);
            std::vector<fcpw::Interaction<DIM>> queryInteractions;
            int nIntersections = absorbingBoundaryAggregate->intersect(queryRay, queryInteractions, false, true);

            intersectionPts.clear();
            for (int i = 0; i < nIntersections; i++) {
                intersectionPts.emplace_back(IntersectionPoint<DIM>(queryInteractions[i].p,
                                                                    queryInteractions[i].n,
                                                                    queryInteractions[i].d));
            }

            return nIntersections;
        };
        geometricQueries.computeAbsorbingBoundarySignedVolume = [absorbingBoundaryAggregate]() -> float {
            return absorbingBoundaryAggregate->signedVolume();
        };
    }
}

template <size_t DIM, typename ReflectingBoundaryAggregateType>
void populateGeometricQueriesForReflectingBoundary(const ReflectingBoundaryAggregateType *reflectingBoundaryAggregate,
                                                   std::function<float(float)> branchTraversalWeight,
                                                   GeometricQueries<DIM>& geometricQueries)
{
    if (reflectingBoundaryAggregate) {
        geometricQueries.hasReflectingBoundary = true;
        geometricQueries.computeDistToReflectingBoundary = [reflectingBoundaryAggregate](
                                                           const Vector<DIM>& x, bool computeSignedDistance) -> float {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            reflectingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            return computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;
        };
        geometricQueries.projectToReflectingBoundary = [reflectingBoundaryAggregate](
                                                       Vector<DIM>& x, Vector<DIM>& normal,
                                                       float& distance, bool computeSignedDistance) -> bool {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            reflectingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            x = interaction.p;
            normal = interaction.n;
            distance = computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;

            return true;
        };
        geometricQueries.intersectReflectingBoundary = [&geometricQueries, reflectingBoundaryAggregate](
                                                       const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                       const Vector<DIM>& dir, float tMax, bool onReflectingBoundary,
                                                       IntersectionPoint<DIM>& intersectionPt) -> bool {
            Vector<DIM> queryOrigin = onReflectingBoundary ?
                                      geometricQueries.offsetPointAlongDirection(origin, -normal) :
                                      origin;
            Vector<DIM> queryDir = dir;
            fcpw::Ray<DIM> queryRay(queryOrigin, queryDir, tMax);
            fcpw::Interaction<DIM> queryInteraction;
            bool hit = reflectingBoundaryAggregate->intersect(queryRay, queryInteraction, false);
            if (!hit) return false;

            intersectionPt.pt = queryInteraction.p;
            intersectionPt.normal = queryInteraction.n;
            intersectionPt.dist = queryInteraction.d;

            return true;
        };
        geometricQueries.intersectReflectingBoundaryAllHits = [&geometricQueries, reflectingBoundaryAggregate](
                                                              const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                              const Vector<DIM>& dir, float tMax, bool onReflectingBoundary,
                                                              std::vector<IntersectionPoint<DIM>>& intersectionPts) -> int {
            Vector<DIM> queryOrigin = onReflectingBoundary ?
                                      geometricQueries.offsetPointAlongDirection(origin, -normal) :
                                      origin;
            Vector<DIM> queryDir = dir;
            fcpw::Ray<DIM> queryRay(queryOrigin, queryDir, tMax);
            std::vector<fcpw::Interaction<DIM>> queryInteractions;
            int nIntersections = reflectingBoundaryAggregate->intersect(queryRay, queryInteractions, false, true);

            intersectionPts.clear();
            for (int i = 0; i < nIntersections; i++) {
                intersectionPts.emplace_back(IntersectionPoint<DIM>(queryInteractions[i].p,
                                                                    queryInteractions[i].n,
                                                                    queryInteractions[i].d));
            }

            return nIntersections;
        };
        geometricQueries.intersectsWithReflectingBoundary = [&geometricQueries, reflectingBoundaryAggregate](
                                                            const Vector<DIM>& xi, const Vector<DIM>& xj,
                                                            const Vector<DIM>& ni, const Vector<DIM>& nj,
                                                            bool offseti, bool offsetj) -> bool {
            Vector<DIM> pt1 = offseti ? geometricQueries.offsetPointAlongDirection(xi, -ni) : xi;
            Vector<DIM> pt2 = offsetj ? geometricQueries.offsetPointAlongDirection(xj, -nj) : xj;

            return !reflectingBoundaryAggregate->hasLineOfSight(pt1, pt2);
        };
        geometricQueries.sampleReflectingBoundary = [reflectingBoundaryAggregate, branchTraversalWeight](
                                                    const Vector<DIM>& x, float radius, const Vector<DIM>& randNums,
                                                    BoundarySample<DIM>& boundarySample) -> bool {
            Vector<DIM> queryPt = x;
            fcpw::BoundingSphere<DIM> querySphere(queryPt, radius*radius);
            fcpw::Interaction<DIM> queryInteraction;
            int nHits = reflectingBoundaryAggregate->intersect(querySphere, queryInteraction,
                                                               randNums, branchTraversalWeight);
            if (nHits < 1) return false;

            boundarySample.pt = queryInteraction.p;
            boundarySample.normal = queryInteraction.n;
            boundarySample.pdf = queryInteraction.d;

            return true;
        };
        geometricQueries.computeReflectingBoundarySignedVolume = [reflectingBoundaryAggregate]() -> float {
            return reflectingBoundaryAggregate->signedVolume();
        };
    }
}

template <size_t DIM, typename NeumannBoundaryAggregateType>
void populateStarRadiusQueryForNeumannBoundary(const NeumannBoundaryAggregateType *reflectingBoundaryAggregate,
                                               GeometricQueries<DIM>& geometricQueries)
{
    if (reflectingBoundaryAggregate) {
        geometricQueries.computeStarRadiusForReflectingBoundary = [reflectingBoundaryAggregate](
                                                                  const Vector<DIM>& x, float minRadius, float maxRadius,
                                                                  float silhouettePrecision, bool flipNormalOrientation) -> float {
            if (minRadius > maxRadius) return maxRadius;
            Vector<DIM> queryPt = x;
            bool flipNormals = true; // FCPW's internal convention requires normals to be flipped
            if (flipNormalOrientation) flipNormals = !flipNormals;
            float squaredSphereRadius = maxRadius < fcpw::maxFloat ? maxRadius*maxRadius : fcpw::maxFloat;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> querySphere(queryPt, squaredSphereRadius);
            bool found = reflectingBoundaryAggregate->findClosestSilhouettePoint(
                querySphere, interaction, flipNormals, minRadius*minRadius, silhouettePrecision);

            return found ? std::max(interaction.d, minRadius) : std::max(maxRadius, minRadius);
        };
    }
}

template <size_t DIM, typename RobinBoundaryAggregateType>
void populateStarRadiusQueryForRobinBoundary(const RobinBoundaryAggregateType *reflectingBoundaryAggregate,
                                             GeometricQueries<DIM>& geometricQueries)
{
    if (reflectingBoundaryAggregate) {
        geometricQueries.computeStarRadiusForReflectingBoundary = [reflectingBoundaryAggregate](
                                                                  const Vector<DIM>& x, float minRadius, float maxRadius,
                                                                  float silhouettePrecision, bool flipNormalOrientation) -> float {
            if (minRadius > maxRadius) return maxRadius;
            Vector<DIM> queryPt = x;
            bool flipNormals = true; // FCPW's internal convention requires normals to be flipped
            if (flipNormalOrientation) flipNormals = !flipNormals;
            float squaredSphereRadius = maxRadius < fcpw::maxFloat ? maxRadius*maxRadius : fcpw::maxFloat;
            fcpw::BoundingSphere<DIM> querySphere(queryPt, squaredSphereRadius);
            reflectingBoundaryAggregate->computeSquaredStarRadius(querySphere, flipNormals, silhouettePrecision);

            return std::max(std::sqrt(querySphere.r2), minRadius);
        };
    }
}

template <size_t DIM>
void populateGeometricQueriesForNeumannBoundary(FcpwNeumannBoundaryHandler<DIM>& neumannBoundaryHandler,
                                                std::function<float(float)> branchTraversalWeight,
                                                GeometricQueries<DIM>& geometricQueries)
{
    fcpw::Aggregate<DIM> *reflectingBoundaryAggregate =
        neumannBoundaryHandler.scene.getSceneData()->aggregate.get();
    populateGeometricQueriesForReflectingBoundary<DIM, fcpw::Aggregate<DIM>>(
        reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
    populateStarRadiusQueryForNeumannBoundary<DIM, fcpw::Aggregate<DIM>>(
        reflectingBoundaryAggregate, geometricQueries);
}

template <size_t DIM>
void populateGeometricQueriesForRobinBoundary(FcpwRobinBoundaryHandler<DIM>& robinBoundaryHandler,
                                              std::function<float(float)> branchTraversalWeight,
                                              GeometricQueries<DIM>& geometricQueries)
{
    std::cerr << "populateGeometricQueriesForRobinBoundary: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
void populateGeometricQueriesForRobinBoundary<2>(FcpwRobinBoundaryHandler<2>& robinBoundaryHandler,
                                                 std::function<float(float)> branchTraversalWeight,
                                                 GeometricQueries<2>& geometricQueries)
{
    using PrimitiveBound = FcpwRobinBoundaryHandler<2>::PrimitiveBound;
    if (robinBoundaryHandler.baseline) {
        using RobinAggregateType = ReflectanceBaseline<2, ReflectanceLineSegment<PrimitiveBound>>;
        RobinAggregateType *reflectingBoundaryAggregate = robinBoundaryHandler.baseline.get();
        populateGeometricQueriesForReflectingBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);

#ifdef FCPW_USE_ENOKI
    } else if (robinBoundaryHandler.mbvh) {
        using RobinAggregateType = ReflectanceMbvh<FCPW_SIMD_WIDTH, 2,
                                                   ReflectanceLineSegment<PrimitiveBound>,
                                                   ReflectanceMbvhNode<2>,
                                                   FcpwRobinBoundaryHandler<2>::WideNodeBound>;
        RobinAggregateType *reflectingBoundaryAggregate = robinBoundaryHandler.mbvh.get();
        populateGeometricQueriesForReflectingBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
#endif
    } else if (robinBoundaryHandler.bvh) {
        using RobinAggregateType = ReflectanceBvh<2, ReflectanceBvhNode<2>,
                                                  ReflectanceLineSegment<PrimitiveBound>,
                                                  FcpwRobinBoundaryHandler<2>::NodeBound>;
        RobinAggregateType *reflectingBoundaryAggregate = robinBoundaryHandler.bvh.get();
        populateGeometricQueriesForReflectingBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
    }
}

template <>
void populateGeometricQueriesForRobinBoundary<3>(FcpwRobinBoundaryHandler<3>& robinBoundaryHandler,
                                                 std::function<float(float)> branchTraversalWeight,
                                                 GeometricQueries<3>& geometricQueries)
{
    using PrimitiveBound = FcpwRobinBoundaryHandler<3>::PrimitiveBound;
    if (robinBoundaryHandler.baseline) {
        using RobinAggregateType = ReflectanceBaseline<3, ReflectanceTriangle<PrimitiveBound>>;
        RobinAggregateType *reflectingBoundaryAggregate = robinBoundaryHandler.baseline.get();
        populateGeometricQueriesForReflectingBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);

#ifdef FCPW_USE_ENOKI
    } else if (robinBoundaryHandler.mbvh) {
        using RobinAggregateType = ReflectanceMbvh<FCPW_SIMD_WIDTH, 3,
                                                   ReflectanceTriangle<PrimitiveBound>,
                                                   ReflectanceMbvhNode<3>,
                                                   FcpwRobinBoundaryHandler<3>::WideNodeBound>;
        RobinAggregateType *reflectingBoundaryAggregate = robinBoundaryHandler.mbvh.get();
        populateGeometricQueriesForReflectingBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
#endif
    } else if (robinBoundaryHandler.bvh) {
        using RobinAggregateType = ReflectanceBvh<3, ReflectanceBvhNode<3>,
                                                  ReflectanceTriangle<PrimitiveBound>,
                                                  FcpwRobinBoundaryHandler<3>::NodeBound>;
        RobinAggregateType *reflectingBoundaryAggregate = robinBoundaryHandler.bvh.get();
        populateGeometricQueriesForReflectingBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, branchTraversalWeight, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
    }
}

std::function<float(float)> getBranchTraversalWeightCallback(float minRadialDist)
{
    HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
    return [harmonicGreensFn, minRadialDist](float r2) -> float {
        float r = std::max(std::sqrt(r2), minRadialDist);
        return std::fabs(harmonicGreensFn.evaluate(r));
    };
}

} // zombie

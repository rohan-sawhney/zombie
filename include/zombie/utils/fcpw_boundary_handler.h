// This file provides utility functions to load 2D or 3D boundary meshes from OBJ files,
// normalize mesh positions to lie within a unit sphere, swap mesh indices to flip orientation,
// and compute the bounding box of a mesh. The FcpwBoundaryHandler class builds an acceleration
// structure to perform geometric queries against a mesh, while the 'populateGeometricQueries'
// function populates the GeometricQueries structure using FcpwBoundaryHandler objects for the
// absorbing (Dirichlet) and reflecting (Neumann or Robin) boundaries.

#pragma once

#include <zombie/core/geometric_queries.h>
#include <cmath>
#include <fcpw/utilities/scene_loader.h>
#include <zombie/utils/robin_boundary_bvh/baseline.h>
#ifdef FCPW_USE_ENOKI
    #include <zombie/utils/robin_boundary_bvh/mbvh.h>
#else
    #include <zombie/utils/robin_boundary_bvh/bvh.h>
#endif

#define RAY_OFFSET 1e-6f

namespace zombie {

// loads 2D or 3D boundary mesh from OBJ file
template <size_t DIM>
void loadBoundaryMesh(const std::string& objFile,
                      std::vector<Vector<DIM>>& positions,
                      std::vector<std::vector<size_t>>& indices);

// mesh utility functions
template <size_t DIM>
void normalize(std::vector<Vector<DIM>>& positions);

void flipOrientation(std::vector<std::vector<size_t>>& indices);

template <size_t DIM>
std::pair<Vector<DIM>, Vector<DIM>> computeBoundingBox(const std::vector<Vector<DIM>>& positions,
                                                       bool makeSquare, float scale);

// Helper class to build an acceleration structure to perform geometric queries such as
// ray intersection, closest point, etc. against a mesh. Also provides a utility function
// to update Robin coefficients after building the acceleration structure.
template <size_t DIM, bool useRobinConditions>
class FcpwBoundaryHandler {
public:
    // constructor
    FcpwBoundaryHandler();

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. For problems with Dirichlet or Robin boundary conditions,
    // set computeSilhouettes to false, while for problems with Neumann boundary conditions,
    // set computeSilhouettes to true. For Robin conditions, additionally provide min and max
    // Robin coefficients per mesh face. Setting buildBvh to false builds a simple list of
    // mesh faces instead of a BVH for brute force geometric queries.
    void buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                    const std::vector<std::vector<size_t>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette={},
                                    bool computeSilhouettes=false,
                                    const std::vector<float>& minRobinCoeffValues={},
                                    const std::vector<float>& maxRobinCoeffValues={},
                                    bool buildBvh=true, bool enableBvhVectorization=false);

    // updates the Robin coefficients for the mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues);
};

// populates the GeometricQueries structure
template <size_t DIM>
void populateGeometricQueries(FcpwBoundaryHandler<DIM, false>& absorbingBoundaryHandler,
                              const std::pair<Vector<DIM>, Vector<DIM>>& boundingBoxExtents,
                              GeometricQueries<DIM>& geometricQueries);

template <size_t DIM, bool useRobinConditions>
void populateGeometricQueries(FcpwBoundaryHandler<DIM, false>& absorbingBoundaryHandler,
                              FcpwBoundaryHandler<DIM, useRobinConditions>& reflectingBoundaryHandler,
                              const std::function<float(float)>& branchTraversalWeight,
                              const std::pair<Vector<DIM>, Vector<DIM>>& boundingBoxExtents,
                              GeometricQueries<DIM>& geometricQueries);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE: BVH refit

template <size_t DIM>
void loadBoundaryMesh(const std::string& objFile,
                      std::vector<Vector<DIM>>& positions,
                      std::vector<std::vector<size_t>>& indices)
{
    std::cerr << "loadBoundaryMesh: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);
}

template <>
void loadBoundaryMesh<2>(const std::string& objFile,
                         std::vector<Vector2>& positions,
                         std::vector<std::vector<size_t>>& indices)
{
    // load file
    fcpw::PolygonSoup<2> soup;
    fcpw::loadLineSegmentSoupFromOBJFile(objFile, soup);

    // collect mesh positions and indices
    positions.clear();
    indices.clear();
    int V = (int)soup.positions.size();
    int L = (int)soup.indices.size()/2;

    for (int l = 0; l < L; l++) {
        size_t i = soup.indices[2*l + 0];
        size_t j = soup.indices[2*l + 1];

        indices.emplace_back(std::vector<size_t>{i, j});
    }

    for (int v = 0; v < V; v++) {
        positions.emplace_back(soup.positions[v]);
    }
}

template <>
void loadBoundaryMesh<3>(const std::string& objFile,
                         std::vector<Vector3>& positions,
                         std::vector<std::vector<size_t>>& indices)
{
    // load file
    fcpw::PolygonSoup<3> soup;
    fcpw::loadTriangleSoupFromOBJFile(objFile, soup);

    // collect mesh positions and indices
    positions.clear();
    indices.clear();
    int V = (int)soup.positions.size();
    int T = (int)soup.indices.size()/3;

    for (int t = 0; t < T; t++) {
        size_t i = soup.indices[3*t + 0];
        size_t j = soup.indices[3*t + 1];
        size_t k = soup.indices[3*t + 2];

        indices.emplace_back(std::vector<size_t>{i, j, k});
    }

    for (int v = 0; v < V; v++) {
        positions.emplace_back(soup.positions[v]);
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

void flipOrientation(std::vector<std::vector<size_t>>& indices)
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

inline float intAsFloat(int a)
{
    union {int a; float b;} u;
    u.a = a;

    return u.b;
}

inline int floatAsInt(float a)
{
    union {float a; int b;} u;
    u.a = a;

    return u.b;
}

template <size_t DIM>
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
    Eigen::Vector2f pOffset(intAsFloat(floatAsInt(p(0)) + (p(0) < 0 ? -nOffset(0) : nOffset(0))),
                            intAsFloat(floatAsInt(p(1)) + (p(1) < 0 ? -nOffset(1) : nOffset(1))));

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
    Eigen::Vector3f pOffset(intAsFloat(floatAsInt(p(0)) + (p(0) < 0 ? -nOffset(0) : nOffset(0))),
                            intAsFloat(floatAsInt(p(1)) + (p(1) < 0 ? -nOffset(1) : nOffset(1))),
                            intAsFloat(floatAsInt(p(2)) + (p(2) < 0 ? -nOffset(2) : nOffset(2))));

    return Eigen::Vector3f(std::fabs(p(0)) < origin ? p(0) + floatScale*n(0) : pOffset(0),
                           std::fabs(p(1)) < origin ? p(1) + floatScale*n(1) : pOffset(1),
                           std::fabs(p(2)) < origin ? p(2) + floatScale*n(2) : pOffset(2));
}

template <size_t DIM, bool useRobinConditions>
FcpwBoundaryHandler<DIM, useRobinConditions>::FcpwBoundaryHandler()
{
    std::cerr << "FcpwBoundaryHandler: Unsupported dimension: " << DIM
              << ", useRobinConditions: " << useRobinConditions
              << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t DIM, bool useRobinConditions>
void FcpwBoundaryHandler<DIM, useRobinConditions>::buildAccelerationStructure(const std::vector<Vector<DIM>>& positions,
                                                                              const std::vector<std::vector<size_t>>& indices,
                                                                              std::function<bool(float, int)> ignoreCandidateSilhouette,
                                                                              bool computeSilhouettes,
                                                                              const std::vector<float>& minRobinCoeffValues,
                                                                              const std::vector<float>& maxRobinCoeffValues,
                                                                              bool buildBvh, bool enableBvhVectorization)
{
    std::cerr << "FcpwBoundaryHandler::buildAccelerationStructure: Unsupported dimension: " << DIM
              << ", useRobinConditions: " << useRobinConditions
              << std::endl;
    exit(EXIT_FAILURE);
}

template <size_t DIM, bool useRobinConditions>
void FcpwBoundaryHandler<DIM, useRobinConditions>::updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                                                           const std::vector<float>& maxRobinCoeffValues)
{
    std::cerr << "FcpwBoundaryHandler::updateRobinCoefficients: Unsupported dimension: " << DIM
              << ", useRobinConditions: " << useRobinConditions
              << std::endl;
    exit(EXIT_FAILURE);
}

template <>
class FcpwBoundaryHandler<2, false> {
public:
    // constructor
    FcpwBoundaryHandler() {}

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. For problems with Dirichlet or Robin boundary conditions,
    // set computeSilhouettes to false, while for problems with Neumann boundary conditions,
    // set computeSilhouettes to true. For Robin conditions, additionally provide min and max
    // Robin coefficients per mesh face. Setting buildBvh to false builds a simple list of
    // mesh faces instead of a BVH for brute force geometric queries.
    void buildAccelerationStructure(const std::vector<Vector2>& positions,
                                    const std::vector<std::vector<size_t>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette={},
                                    bool computeSilhouettes=false,
                                    const std::vector<float>& minRobinCoeffValues={},
                                    const std::vector<float>& maxRobinCoeffValues={},
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
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
                scene.setObjectVertex(positions[i], i, 0);
            }

            // specify the line segment indices
            for (int i = 0; i < L; i++) {
                fcpw::Vector2i index(indices[i][0], indices[i][1]);
                scene.setObjectLineSegment(index, i, 0);
            }

            // compute silhouettes
            if (computeSilhouettes) {
                scene.computeSilhouettes(ignoreCandidateSilhouette);
            }

            // build aggregate
            fcpw::AggregateType aggregateType = buildBvh ?
                                                fcpw::AggregateType::Bvh_SurfaceArea :
                                                fcpw::AggregateType::Baseline;
            scene.build(aggregateType, enableBvhVectorization, true, true);
        }
    }

    // updates the Robin coefficients for the mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues) {
        std::cerr << "FcpwBoundaryHandler<2, false>::updateRobinCoefficients: not supported!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // member
    fcpw::Scene<2> scene;
};

template <>
class FcpwBoundaryHandler<3, false> {
public:
    // constructor
    FcpwBoundaryHandler() {}

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. For problems with Dirichlet or Robin boundary conditions,
    // set computeSilhouettes to false, while for problems with Neumann boundary conditions,
    // set computeSilhouettes to true. For Robin conditions, additionally provide min and max
    // Robin coefficients per mesh face. Setting buildBvh to false builds a simple list of
    // mesh faces instead of a BVH for brute force geometric queries.
    void buildAccelerationStructure(const std::vector<Vector3>& positions,
                                    const std::vector<std::vector<size_t>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette={},
                                    bool computeSilhouettes=false,
                                    const std::vector<float>& minRobinCoeffValues={},
                                    const std::vector<float>& maxRobinCoeffValues={},
                                    bool buildBvh=true, bool enableBvhVectorization=false) {
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
                fcpw::Vector3i index(indices[i][0], indices[i][1], indices[i][2]);
                scene.setObjectTriangle(index, i, 0);
            }

            // compute silhouettes
            if (computeSilhouettes) {
                scene.computeSilhouettes(ignoreCandidateSilhouette);
            }

            // build aggregate
            fcpw::AggregateType aggregateType = buildBvh ?
                                                fcpw::AggregateType::Bvh_SurfaceArea :
                                                fcpw::AggregateType::Baseline;
            scene.build(aggregateType, enableBvhVectorization, true, true);
        }
    }

    // updates the Robin coefficients for the mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues) {
        std::cerr << "FcpwBoundaryHandler<3, false>::updateRobinCoefficients: not supported!" << std::endl;
        exit(EXIT_FAILURE);
    }

    // member
    fcpw::Scene<3> scene;
};

template <>
class FcpwBoundaryHandler<2, true> {
public:
    // constructor
    FcpwBoundaryHandler() {
        baseline = nullptr;
        bvh = nullptr;
#ifdef FCPW_USE_ENOKI
        mbvh = nullptr;
#endif
    }

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. For problems with Dirichlet or Robin boundary conditions,
    // set computeSilhouettes to false, while for problems with Neumann boundary conditions,
    // set computeSilhouettes to true. For Robin conditions, additionally provide min and max
    // Robin coefficients per mesh face. Setting buildBvh to false builds a simple list of
    // mesh faces instead of a BVH for brute force geometric queries.
    void buildAccelerationStructure(const std::vector<Vector2>& positions,
                                    const std::vector<std::vector<size_t>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette={},
                                    bool computeSilhouettes=false,
                                    const std::vector<float>& minRobinCoeffValues={},
                                    const std::vector<float>& maxRobinCoeffValues={},
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
                std::cerr << "FcpwBoundaryHandler<2, true>::buildAccelerationStructure: invalid Robin coefficient sizes!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::vector<VertexFaceAdjacency> vertexTable(V);
            soup.positions = positions;
            soup.indices.resize(2*L);
            lineSegments.resize(L);
            lineSegmentPtrs.resize(L, nullptr);

            // update soup and line segment indices
            for (int i = 0; i < L; i++) {
                RobinLineSegment& lineSegment = lineSegments[i];
                lineSegmentPtrs[i] = &lineSegment;
                lineSegment.soup = &soup;
                lineSegment.setIndex(i);
                lineSegment.minRobinCoeff = minRobinCoeffValues[i];
                lineSegment.maxRobinCoeff = maxRobinCoeffValues[i];

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
                RobinLineSegment& lineSegment = lineSegments[i];
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
                    bvh = initializeRobinBvh<2, RobinLineSegment>(soup, lineSegmentPtrs, silhouettePtrsStub,
                                                                  true, true, FCPW_SIMD_WIDTH);
                    mbvh = initializeVectorizedRobinBvh<2, RobinLineSegment>(bvh.get(), lineSegmentPtrs,
                                                                             silhouettePtrsStub, true);
#else
                    bvh = initializeRobinBvh<2, RobinLineSegment>(soup, lineSegmentPtrs, silhouettePtrsStub);
#endif
                } else {
                    bvh = initializeRobinBvh<2, RobinLineSegment>(soup, lineSegmentPtrs, silhouettePtrsStub);
                }

            } else {
                baseline = initializeRobinBaseline<2, RobinLineSegment>(lineSegmentPtrs, silhouettePtrsStub);
            }
        }
    }

    // updates the Robin coefficients for the mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues) {
        if (baseline != nullptr) {
            baseline->updateRobinCoefficients(minRobinCoeffValues, maxRobinCoeffValues);

#ifdef FCPW_USE_ENOKI
        } else if (mbvh != nullptr) {
            mbvh->updateRobinCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
#endif
        } else if (bvh != nullptr) {
            bvh->updateRobinCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
        }
    }

    // members
    std::unique_ptr<RobinBaseline<2, RobinLineSegment>> baseline;
    std::unique_ptr<RobinBvh<2, RobinBvhNode<2>, RobinLineSegment>> bvh;
#ifdef FCPW_USE_ENOKI
    std::unique_ptr<RobinMbvh<FCPW_SIMD_WIDTH, 2, RobinLineSegment, RobinMbvhNode<2>>> mbvh;
#endif
    PolygonSoup<2> soup;
    std::vector<RobinLineSegment> lineSegments;
    std::vector<RobinLineSegment *> lineSegmentPtrs;
    std::vector<fcpw::SilhouettePrimitive<2> *> silhouettePtrsStub;
};

template <>
class FcpwBoundaryHandler<3, true> {
public:
    // constructor
    FcpwBoundaryHandler() {
        baseline = nullptr;
        bvh = nullptr;
#ifdef FCPW_USE_ENOKI
        mbvh = nullptr;
#endif
    }

    // builds an FCPW acceleration structure (specifically a bounding volume hierarchy) from
    // a set of positions and indices. For problems with Dirichlet or Robin boundary conditions,
    // set computeSilhouettes to false, while for problems with Neumann boundary conditions,
    // set computeSilhouettes to true. For Robin conditions, additionally provide min and max
    // Robin coefficients per mesh face. Setting buildBvh to false builds a simple list of
    // mesh faces instead of a BVH for brute force geometric queries.
    void buildAccelerationStructure(const std::vector<Vector3>& positions,
                                    const std::vector<std::vector<size_t>>& indices,
                                    std::function<bool(float, int)> ignoreCandidateSilhouette={},
                                    bool computeSilhouettes=false,
                                    const std::vector<float>& minRobinCoeffValues={},
                                    const std::vector<float>& maxRobinCoeffValues={},
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
                std::cerr << "FcpwBoundaryHandler<3, true>::buildAccelerationStructure: invalid Robin coefficient sizes!" << std::endl;
                exit(EXIT_FAILURE);
            }

            std::map<std::pair<int, int>, EdgeFaceAdjacency> edgeTable;
            soup.positions = positions;
            soup.indices.resize(3*T);
            triangles.resize(T);
            trianglePtrs.resize(T, nullptr);

            // update soup and triangle indices
            for (int i = 0; i < T; i++) {
                RobinTriangle& triangle = triangles[i];
                trianglePtrs[i] = &triangle;
                triangle.soup = &soup;
                triangle.setIndex(i);
                triangle.minRobinCoeff = minRobinCoeffValues[i];
                triangle.maxRobinCoeff = maxRobinCoeffValues[i];

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
                RobinTriangle& triangle = triangles[i];
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
                    bvh = initializeRobinBvh<3, RobinTriangle>(soup, trianglePtrs, silhouettePtrsStub,
                                                               true, true, FCPW_SIMD_WIDTH);
                    mbvh = initializeVectorizedRobinBvh<3, RobinTriangle>(bvh.get(), trianglePtrs,
                                                                          silhouettePtrsStub, true);
#else
                    bvh = initializeRobinBvh<3, RobinTriangle>(soup, trianglePtrs, silhouettePtrsStub);
#endif
                } else {
                    bvh = initializeRobinBvh<3, RobinTriangle>(soup, trianglePtrs, silhouettePtrsStub);
                }

            } else {
                baseline = initializeRobinBaseline<3, RobinTriangle>(trianglePtrs, silhouettePtrsStub);
            }
        }
    }

    // updates the Robin coefficients for the mesh
    void updateRobinCoefficients(const std::vector<float>& minRobinCoeffValues,
                                 const std::vector<float>& maxRobinCoeffValues) {
        if (baseline != nullptr) {
            baseline->updateRobinCoefficients(minRobinCoeffValues, maxRobinCoeffValues);

#ifdef FCPW_USE_ENOKI
        } else if (mbvh != nullptr) {
            mbvh->updateRobinCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
#endif
        } else if (bvh != nullptr) {
            bvh->updateRobinCoefficients(minRobinCoeffValues, maxRobinCoeffValues);
        }
    }

    // members
    std::unique_ptr<RobinBaseline<3, RobinTriangle>> baseline;
    std::unique_ptr<RobinBvh<3, RobinBvhNode<3>, RobinTriangle>> bvh;
#ifdef FCPW_USE_ENOKI
    std::unique_ptr<RobinMbvh<FCPW_SIMD_WIDTH, 3, RobinTriangle, RobinMbvhNode<3>>> mbvh;
#endif
    PolygonSoup<3> soup;
    std::vector<RobinTriangle> triangles;
    std::vector<RobinTriangle *> trianglePtrs;
    std::vector<fcpw::SilhouettePrimitive<3> *> silhouettePtrsStub;
};

template <size_t DIM, typename AbsorbingBoundaryAggregateType, typename ReflectingBoundaryAggregateType>
void populateGeometricQueries(const AbsorbingBoundaryAggregateType *absorbingBoundaryAggregate,
                              const ReflectingBoundaryAggregateType *reflectingBoundaryAggregate,
                              const std::function<float(float)>& branchTraversalWeight,
                              const std::pair<Vector<DIM>, Vector<DIM>>& boundingBoxExtents,
                              GeometricQueries<DIM>& geometricQueries)
{
    fcpw::BoundingBox<DIM> boundingBox;
    boundingBox.expandToInclude(boundingBoxExtents.first);
    boundingBox.expandToInclude(boundingBoxExtents.second);

    geometricQueries.computeDistToAbsorbingBoundary = [absorbingBoundaryAggregate, boundingBox](
                                                       const Vector<DIM>& x, bool computeSignedDistance) -> float {
        if (absorbingBoundaryAggregate != nullptr) {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            absorbingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            return computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;
        }

        float d2Min, d2Max;
        boundingBox.computeSquaredDistance(x, d2Min, d2Max);
        return std::sqrt(d2Max);
    };
    geometricQueries.computeDistToReflectingBoundary = [reflectingBoundaryAggregate](
                                                        const Vector<DIM>& x,
                                                        bool computeSignedDistance) -> float {
        if (reflectingBoundaryAggregate != nullptr) {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            reflectingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            return computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;
        }

        return fcpw::maxFloat;
    };
    geometricQueries.computeDistToBoundary = [&geometricQueries](const Vector<DIM>& x,
                                                                 bool computeSignedDistance) -> float {
        float d1 = geometricQueries.computeDistToAbsorbingBoundary(x, computeSignedDistance);
        float d2 = geometricQueries.computeDistToReflectingBoundary(x, computeSignedDistance);

        return std::fabs(d1) < std::fabs(d2) ? d1 : d2;
    };
    geometricQueries.projectToAbsorbingBoundary = [absorbingBoundaryAggregate](
                                                   Vector<DIM>& x, Vector<DIM>& normal,
                                                   float& distance, bool computeSignedDistance) -> bool {
        if (absorbingBoundaryAggregate != nullptr) {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            absorbingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            x = interaction.p;
            normal = interaction.n;
            distance = computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;

            return true;
        }

        distance = 0.0f;
        return false;
    };
    geometricQueries.projectToReflectingBoundary = [reflectingBoundaryAggregate](
                                                    Vector<DIM>& x, Vector<DIM>& normal,
                                                    float& distance, bool computeSignedDistance) -> bool {
        if (reflectingBoundaryAggregate != nullptr) {
            Vector<DIM> queryPt = x;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> sphere(queryPt, fcpw::maxFloat);
            reflectingBoundaryAggregate->findClosestPoint(sphere, interaction, computeSignedDistance);

            x = interaction.p;
            normal = interaction.n;
            distance = computeSignedDistance ? interaction.signedDistance(queryPt) : interaction.d;

            return true;
        }

        distance = 0.0f;
        return false;
    };
    geometricQueries.projectToBoundary = [&geometricQueries](Vector<DIM>& x, Vector<DIM>& normal,
                                                             float& distance, bool computeSignedDistance) -> bool {
        distance = fcpw::maxFloat;
        bool didProject = false;

        Vector<DIM> absorbingBoundaryPt = x;
        Vector<DIM> absorbingBoundaryNormal;
        float distanceToAbsorbingBoundary;
        if (geometricQueries.projectToAbsorbingBoundary(absorbingBoundaryPt, absorbingBoundaryNormal,
                                                        distanceToAbsorbingBoundary, computeSignedDistance)) {
            x = absorbingBoundaryPt;
            normal = absorbingBoundaryNormal;
            distance = distanceToAbsorbingBoundary;
            didProject = true;
        }

        Vector<DIM> reflectingBoundaryPt = x;
        Vector<DIM> reflectingBoundaryNormal;
        float distanceToReflectingBoundary;
        if (geometricQueries.projectToReflectingBoundary(reflectingBoundaryPt, reflectingBoundaryNormal,
                                                         distanceToReflectingBoundary, computeSignedDistance)) {
            if (std::fabs(distanceToReflectingBoundary) < std::fabs(distance)) {
                x = reflectingBoundaryPt;
                normal = reflectingBoundaryNormal;
                distance = distanceToReflectingBoundary;
            }

            didProject = true;
        }

        if (!didProject) distance = 0.0f;
        return didProject;
    };
    geometricQueries.offsetPointAlongDirection = [](const Vector<DIM>& x,
                                                    const Vector<DIM>& dir) -> Vector<DIM> {
        return offsetPointAlongDirection<DIM>(x, dir);
    };
    geometricQueries.intersectAbsorbingBoundary = [&geometricQueries, absorbingBoundaryAggregate](
                                                   const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                   const Vector<DIM>& dir, float tMax, bool onAborbingBoundary,
                                                   IntersectionPoint<DIM>& intersectionPt) -> bool {
        if (absorbingBoundaryAggregate != nullptr) {
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
        }

        return false;
    };
    geometricQueries.intersectReflectingBoundary = [&geometricQueries, reflectingBoundaryAggregate](
                                                    const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                    const Vector<DIM>& dir, float tMax, bool onReflectingBoundary,
                                                    IntersectionPoint<DIM>& intersectionPt) -> bool {
        if (reflectingBoundaryAggregate != nullptr) {
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
        }

        return false;
    };
    geometricQueries.intersectBoundary = [&geometricQueries](
                                          const Vector<DIM>& origin, const Vector<DIM>& normal,
                                          const Vector<DIM>& dir, float tMax,
                                          bool onAborbingBoundary, bool onReflectingBoundary,
                                          IntersectionPoint<DIM>& intersectionPt) -> bool {
        IntersectionPoint<DIM> absorbingBoundaryIntersectionPt;
        bool intersectedAbsorbingBoundary = geometricQueries.intersectAbsorbingBoundary(
            origin, normal, dir, tMax, onAborbingBoundary, absorbingBoundaryIntersectionPt);

        IntersectionPoint<DIM> reflectingBoundaryIntersectionPt;
        bool intersectedReflectingBoundary = geometricQueries.intersectReflectingBoundary(
            origin, normal, dir, tMax, onReflectingBoundary, reflectingBoundaryIntersectionPt);

        if (intersectedAbsorbingBoundary && intersectedReflectingBoundary) {
            if (absorbingBoundaryIntersectionPt.dist < reflectingBoundaryIntersectionPt.dist) {
                intersectionPt = absorbingBoundaryIntersectionPt;

            } else {
                intersectionPt = reflectingBoundaryIntersectionPt;
            }

        } else if (intersectedAbsorbingBoundary) {
            intersectionPt = absorbingBoundaryIntersectionPt;

        } else if (intersectedReflectingBoundary) {
            intersectionPt = reflectingBoundaryIntersectionPt;
        }

        return intersectedAbsorbingBoundary || intersectedReflectingBoundary;
    };
    geometricQueries.intersectBoundaryAllHits = [&geometricQueries,
                                                 absorbingBoundaryAggregate, reflectingBoundaryAggregate](
                                                 const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                 const Vector<DIM>& dir, float tMax,
                                                 bool onAborbingBoundary, bool onReflectingBoundary,
                                                 std::vector<IntersectionPoint<DIM>>& intersectionPts) -> int {
        // clear buffers
        int nIntersections = 0;
        intersectionPts.clear();

        if (absorbingBoundaryAggregate != nullptr) {
            // initialize query
            Vector<DIM> queryOrigin = onAborbingBoundary ?
                                      geometricQueries.offsetPointAlongDirection(origin, -normal) :
                                      origin;
            Vector<DIM> queryDir = dir;

            // intersect absorbing boundary
            fcpw::Ray<DIM> queryRay(queryOrigin, queryDir, tMax);
            std::vector<fcpw::Interaction<DIM>> queryInteractions;
            int nHits = absorbingBoundaryAggregate->intersect(queryRay, queryInteractions, false, true);
            nIntersections += nHits;

            for (int i = 0; i < nHits; i++) {
                intersectionPts.emplace_back(IntersectionPoint<DIM>(queryInteractions[i].p,
                                                                    queryInteractions[i].n,
                                                                    queryInteractions[i].d));
            }
        }

        if (reflectingBoundaryAggregate != nullptr) {
            // initialize query
            Vector<DIM> queryOrigin = onReflectingBoundary ?
                                      geometricQueries.offsetPointAlongDirection(origin, -normal) :
                                      origin;
            Vector<DIM> queryDir = dir;

            // intersect reflecting boundary
            fcpw::Ray<DIM> queryRay(queryOrigin, queryDir, tMax);
            std::vector<fcpw::Interaction<DIM>> queryInteractions;
            int nHits = reflectingBoundaryAggregate->intersect(queryRay, queryInteractions, false, true);
            nIntersections += nHits;

            for (int i = 0; i < nHits; i++) {
                intersectionPts.emplace_back(IntersectionPoint<DIM>(queryInteractions[i].p,
                                                                    queryInteractions[i].n,
                                                                    queryInteractions[i].d));
            }
        }

        return nIntersections;
    };
    geometricQueries.intersectsWithReflectingBoundary = [&geometricQueries, reflectingBoundaryAggregate](
                                                         const Vector<DIM>& xi, const Vector<DIM>& xj,
                                                         const Vector<DIM>& ni, const Vector<DIM>& nj,
                                                         bool offseti, bool offsetj) -> bool {
        if (reflectingBoundaryAggregate != nullptr) {
            Vector<DIM> pt1 = offseti ? geometricQueries.offsetPointAlongDirection(xi, -ni) : xi;
            Vector<DIM> pt2 = offsetj ? geometricQueries.offsetPointAlongDirection(xj, -nj) : xj;

            return !reflectingBoundaryAggregate->hasLineOfSight(pt1, pt2);
        }

        return false;
    };
    geometricQueries.sampleReflectingBoundary = [reflectingBoundaryAggregate, &branchTraversalWeight](
                                                 const Vector<DIM>& x, float radius, const Vector<DIM>& randNums,
                                                 BoundarySample<DIM>& boundarySample) -> bool {
        if (reflectingBoundaryAggregate != nullptr) {
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
        }

        return false;
    };
    geometricQueries.insideDomain = [&geometricQueries](const Vector<DIM>& x, bool useRayIntersections) -> bool {
        if (!geometricQueries.domainIsWatertight) return true;
        if (useRayIntersections) {
            bool isInside = true;
            Vector<DIM> zero = Vector<DIM>::Zero();
            for (size_t i = 0; i < DIM; i++) {
                Vector<DIM> dir = zero;
                dir(i) = 1.0f;
                std::vector<IntersectionPoint<DIM>> is;
                int hits = geometricQueries.intersectBoundaryAllHits(x, zero, dir, maxFloat, false, false, is);
                isInside = isInside && (hits%2 == 1);
            }

            return isInside;
        }

        return geometricQueries.computeDistToBoundary(x, true) < 0.0f;
    };
    geometricQueries.outsideBoundingDomain = [boundingBox](const Vector<DIM>& x) -> bool {
        return !boundingBox.contains(x);
    };
    geometricQueries.computeSignedDomainVolume = [absorbingBoundaryAggregate, reflectingBoundaryAggregate]() -> float {
        float signedVolume = 0.0f;
        if (absorbingBoundaryAggregate != nullptr) signedVolume += absorbingBoundaryAggregate->signedVolume();
        if (reflectingBoundaryAggregate != nullptr) signedVolume += reflectingBoundaryAggregate->signedVolume();

        return signedVolume;
    };
}

template <size_t DIM, typename NeumannBoundaryAggregateType>
void populateStarRadiusQueryForNeumannBoundary(const NeumannBoundaryAggregateType *reflectingBoundaryAggregate,
                                               GeometricQueries<DIM>& geometricQueries)
{
    geometricQueries.computeStarRadiusForReflectingBoundary = [reflectingBoundaryAggregate](
                                                               const Vector<DIM>& x, float minRadius, float maxRadius,
                                                               float silhouettePrecision, bool flipNormalOrientation) -> float {
        if (minRadius > maxRadius) return maxRadius;
        if (reflectingBoundaryAggregate != nullptr) {
            Vector<DIM> queryPt = x;
            bool flipNormals = true; // FCPW's internal convention requires normals to be flipped
            if (flipNormalOrientation) flipNormals = !flipNormals;

            float squaredSphereRadius = maxRadius < fcpw::maxFloat ? maxRadius*maxRadius : fcpw::maxFloat;
            fcpw::Interaction<DIM> interaction;
            fcpw::BoundingSphere<DIM> querySphere(queryPt, squaredSphereRadius);
            bool found = reflectingBoundaryAggregate->findClosestSilhouettePoint(
                querySphere, interaction, flipNormals, minRadius*minRadius, silhouettePrecision);
            if (found) return std::max(interaction.d, minRadius);
        }

        return std::max(maxRadius, minRadius);
    };
}

template <size_t DIM, typename RobinBoundaryAggregateType>
void populateStarRadiusQueryForRobinBoundary(const RobinBoundaryAggregateType *reflectingBoundaryAggregate,
                                             GeometricQueries<DIM>& geometricQueries)
{
    geometricQueries.computeStarRadiusForReflectingBoundary = [reflectingBoundaryAggregate](
                                                               const Vector<DIM>& x, float minRadius, float maxRadius,
                                                               float silhouettePrecision, bool flipNormalOrientation) -> float {
        if (minRadius > maxRadius) return maxRadius;
        if (reflectingBoundaryAggregate != nullptr) {
            Vector<DIM> queryPt = x;
            bool flipNormals = true; // FCPW's internal convention requires normals to be flipped
            if (flipNormalOrientation) flipNormals = !flipNormals;

            float squaredSphereRadius = maxRadius < fcpw::maxFloat ? maxRadius*maxRadius : fcpw::maxFloat;
            fcpw::BoundingSphere<DIM> querySphere(queryPt, squaredSphereRadius);
            reflectingBoundaryAggregate->computeSquaredStarRadius(querySphere, flipNormals, silhouettePrecision);
            return std::max(std::sqrt(querySphere.r2), minRadius);
        }

        return std::max(maxRadius, minRadius);
    };
}

template <size_t DIM>
void populateGeometricQueries(FcpwBoundaryHandler<DIM, false>& absorbingBoundaryHandler,
                              const std::pair<Vector<DIM>, Vector<DIM>>& boundingBoxExtents,
                              GeometricQueries<DIM>& geometricQueries)
{
    fcpw::Aggregate<DIM> *absorbingBoundaryAggregate = absorbingBoundaryHandler.scene.getSceneData()->aggregate.get();
    populateGeometricQueries<DIM, fcpw::Aggregate<DIM>, fcpw::Aggregate<DIM>>(
        absorbingBoundaryAggregate, nullptr, {}, boundingBoxExtents, geometricQueries);
}

template <size_t DIM, bool useRobinConditions>
void populateGeometricQueries(const FcpwBoundaryHandler<DIM, false>& absorbingBoundaryHandler,
                              const FcpwBoundaryHandler<DIM, useRobinConditions>& reflectingBoundaryHandler,
                              const std::function<float(float)>& branchTraversalWeight,
                              const std::pair<Vector<DIM>, Vector<DIM>>& boundingBoxExtents,
                              GeometricQueries<DIM>& geometricQueries)
{
    std::cerr << "populateGeometricQueries: Unsupported dimension: " << DIM
              << ", useRobinConditions: " << useRobinConditions
              << std::endl;
    exit(EXIT_FAILURE);
}

template <>
void populateGeometricQueries<2, false>(FcpwBoundaryHandler<2, false>& absorbingBoundaryHandler,
                                        FcpwBoundaryHandler<2, false>& reflectingBoundaryHandler,
                                        const std::function<float(float)>& branchTraversalWeight,
                                        const std::pair<Vector2, Vector2>& boundingBoxExtents,
                                        GeometricQueries<2>& geometricQueries)
{
    fcpw::Aggregate<2> *absorbingBoundaryAggregate = absorbingBoundaryHandler.scene.getSceneData()->aggregate.get();
    fcpw::Aggregate<2> *reflectingBoundaryAggregate = reflectingBoundaryHandler.scene.getSceneData()->aggregate.get();
    populateGeometricQueries<2, fcpw::Aggregate<2>, fcpw::Aggregate<2>>(
        absorbingBoundaryAggregate, reflectingBoundaryAggregate,
        branchTraversalWeight, boundingBoxExtents, geometricQueries);
    populateStarRadiusQueryForNeumannBoundary<2, fcpw::Aggregate<2>>(
        reflectingBoundaryAggregate, geometricQueries);
}

template <>
void populateGeometricQueries<3, false>(FcpwBoundaryHandler<3, false>& absorbingBoundaryHandler,
                                        FcpwBoundaryHandler<3, false>& reflectingBoundaryHandler,
                                        const std::function<float(float)>& branchTraversalWeight,
                                        const std::pair<Vector3, Vector3>& boundingBoxExtents,
                                        GeometricQueries<3>& geometricQueries)
{
    fcpw::Aggregate<3> *absorbingBoundaryAggregate = absorbingBoundaryHandler.scene.getSceneData()->aggregate.get();
    fcpw::Aggregate<3> *reflectingBoundaryAggregate = reflectingBoundaryHandler.scene.getSceneData()->aggregate.get();
    populateGeometricQueries<3, fcpw::Aggregate<3>, fcpw::Aggregate<3>>(
        absorbingBoundaryAggregate, reflectingBoundaryAggregate,
        branchTraversalWeight, boundingBoxExtents, geometricQueries);
    populateStarRadiusQueryForNeumannBoundary<3, fcpw::Aggregate<3>>(
        reflectingBoundaryAggregate, geometricQueries);
}

template <>
void populateGeometricQueries<2, true>(FcpwBoundaryHandler<2, false>& absorbingBoundaryHandler,
                                       FcpwBoundaryHandler<2, true>& reflectingBoundaryHandler,
                                       const std::function<float(float)>& branchTraversalWeight,
                                       const std::pair<Vector2, Vector2>& boundingBoxExtents,
                                       GeometricQueries<2>& geometricQueries)
{
    fcpw::Aggregate<2> *absorbingBoundaryAggregate = absorbingBoundaryHandler.scene.getSceneData()->aggregate.get();
    if (reflectingBoundaryHandler.baseline != nullptr) {
        using RobinAggregateType = RobinBaseline<2, RobinLineSegment>;
        RobinAggregateType *reflectingBoundaryAggregate = reflectingBoundaryHandler.baseline.get();
        populateGeometricQueries<2, fcpw::Aggregate<2>, RobinAggregateType>(
            absorbingBoundaryAggregate, reflectingBoundaryAggregate,
            branchTraversalWeight, boundingBoxExtents, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);

#ifdef FCPW_USE_ENOKI
    } else if (reflectingBoundaryHandler.mbvh != nullptr) {
        using RobinAggregateType = RobinMbvh<FCPW_SIMD_WIDTH, 2, RobinLineSegment, RobinMbvhNode<2>>;
        RobinAggregateType *reflectingBoundaryAggregate = reflectingBoundaryHandler.mbvh.get();
        populateGeometricQueries<2, fcpw::Aggregate<2>, RobinAggregateType>(
            absorbingBoundaryAggregate, reflectingBoundaryAggregate,
            branchTraversalWeight, boundingBoxExtents, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
#endif
    } else if (reflectingBoundaryHandler.bvh != nullptr) {
        using RobinAggregateType = RobinBvh<2, RobinBvhNode<2>, RobinLineSegment>;
        RobinAggregateType *reflectingBoundaryAggregate = reflectingBoundaryHandler.bvh.get();
        populateGeometricQueries<2, fcpw::Aggregate<2>, RobinAggregateType>(
            absorbingBoundaryAggregate, reflectingBoundaryAggregate,
            branchTraversalWeight, boundingBoxExtents, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<2, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
    }
}

template <>
void populateGeometricQueries<3, true>(FcpwBoundaryHandler<3, false>& absorbingBoundaryHandler,
                                       FcpwBoundaryHandler<3, true>& reflectingBoundaryHandler,
                                       const std::function<float(float)>& branchTraversalWeight,
                                       const std::pair<Vector3, Vector3>& boundingBoxExtents,
                                       GeometricQueries<3>& geometricQueries)
{
    fcpw::Aggregate<3> *absorbingBoundaryAggregate = absorbingBoundaryHandler.scene.getSceneData()->aggregate.get();
    if (reflectingBoundaryHandler.baseline != nullptr) {
        using RobinAggregateType = RobinBaseline<3, RobinTriangle>;
        RobinAggregateType *reflectingBoundaryAggregate = reflectingBoundaryHandler.baseline.get();
        populateGeometricQueries<3, fcpw::Aggregate<3>, RobinAggregateType>(
            absorbingBoundaryAggregate, reflectingBoundaryAggregate,
            branchTraversalWeight, boundingBoxExtents, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);

#ifdef FCPW_USE_ENOKI
    } else if (reflectingBoundaryHandler.mbvh != nullptr) {
        using RobinAggregateType = RobinMbvh<FCPW_SIMD_WIDTH, 3, RobinTriangle, RobinMbvhNode<3>>;
        RobinAggregateType *reflectingBoundaryAggregate = reflectingBoundaryHandler.mbvh.get();
        populateGeometricQueries<3, fcpw::Aggregate<3>, RobinAggregateType>(
            absorbingBoundaryAggregate, reflectingBoundaryAggregate,
            branchTraversalWeight, boundingBoxExtents, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
#endif
    } else if (reflectingBoundaryHandler.bvh != nullptr) {
        using RobinAggregateType = RobinBvh<3, RobinBvhNode<3>, RobinTriangle>;
        RobinAggregateType *reflectingBoundaryAggregate = reflectingBoundaryHandler.bvh.get();
        populateGeometricQueries<3, fcpw::Aggregate<3>, RobinAggregateType>(
            absorbingBoundaryAggregate, reflectingBoundaryAggregate,
            branchTraversalWeight, boundingBoxExtents, geometricQueries);
        populateStarRadiusQueryForRobinBoundary<3, RobinAggregateType>(
            reflectingBoundaryAggregate, geometricQueries);
    }
}

} // zombie

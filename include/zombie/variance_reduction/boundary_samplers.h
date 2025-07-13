// This file defines a boundary sampler for generating uniformly distributed sample points
// on a 2D or 3D boundary mesh defined by a set of vertices and indices. These sample points
// are required by the Boundary Value Caching (BVC) and Reverse Walk Splatting (RWS) algorithms
// for reducing variance of the walk-on-spheres and walk-on-stars estimators. BVC and RWS currently
// require sample points on the absorbing boundary to be displaced slightly along the boundary normal.

#pragma once

#include <zombie/core/geometry_helpers.h>
#include <zombie/point_estimation/common.h>
#include <unordered_map>

namespace zombie {

template <typename T, size_t DIM>
class BoundarySampler {
public:
    // destructor
    virtual ~BoundarySampler() = default;

    // performs any sampler specific initialization
    virtual void initialize(float normalOffsetForBoundary, bool solveDoubleSided) = 0;

    // returns the number of sample points to be generated on the user-specified side of the boundary
    virtual int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const = 0;

    // generates sample points on the boundary
    virtual void generateSamples(int nSamples, SampleType sampleType,
                                 float normalOffsetForBoundary,
                                 const GeometricQueries<DIM>& queries,
                                 std::vector<SamplePoint<T, DIM>>& samplePts,
                                 bool generateBoundaryNormalAlignedSamples=false) = 0;
};

template <typename T>
class UniformLineSegmentBoundarySampler: public BoundarySampler<T, 2> {
public:
    // constructor
    UniformLineSegmentBoundarySampler(const std::vector<Vector2>& positions_,
                                      const std::vector<Vector2i>& indices_,
                                      std::function<bool(const Vector2&)> insideSolveRegion_,
                                      bool computeWeightedNormals=false);

    // performs sampler specific initialization
    void initialize(float normalOffsetForBoundary, bool solveDoubleSided);

    // returns the number of sample points to be generated on the user-specified side of the boundary
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         const GeometricQueries<2>& queries,
                         std::vector<SamplePoint<T, 2>>& samplePts,
                         bool generateBoundaryNormalAlignedSamples=false);

    // getters
    const std::vector<Vector2>& getPositions() const { return positions; }
    const std::vector<Vector2i>& getIndices() const { return indices; }
    const std::vector<Vector2>& getNormals() const { return normals; }
    const CDFTable& getCDFTable(bool returnBoundaryNormalAligned) const {
        return returnBoundaryNormalAligned ? cdfTableNormalAligned : cdfTable;
    }
    float getBoundaryArea(bool returnBoundaryNormalAligned) const {
        return returnBoundaryNormalAligned ? boundaryAreaNormalAligned : boundaryArea;
    }

private:
    // computes normals
    void computeNormals(bool computeWeighted);

    // builds a cdf table for sampling
    void buildCDFTable(CDFTable& table, float& area, float normalOffsetForBoundary);

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         const GeometricQueries<2>& queries,
                         const CDFTable& table, float area,
                         std::vector<SamplePoint<T, 2>>& samplePts);

    // members
    pcg32 rng;
    const std::vector<Vector2>& positions;
    const std::vector<Vector2i>& indices;
    std::function<bool(const Vector2&)> insideSolveRegion;
    std::vector<Vector2> normals;
    CDFTable cdfTable, cdfTableNormalAligned;
    float boundaryArea, boundaryAreaNormalAligned;
};

template <typename T>
std::shared_ptr<BoundarySampler<T, 2>> createUniformLineSegmentBoundarySampler(
                                        const std::vector<Vector2>& positions,
                                        const std::vector<Vector2i>& indices,
                                        std::function<bool(const Vector2&)> insideSolveRegion,
                                        bool computeWeightedNormals=false);

template <typename T>
class UniformTriangleBoundarySampler: public BoundarySampler<T, 3> {
public:
    // constructor
    UniformTriangleBoundarySampler(const std::vector<Vector3>& positions_,
                                   const std::vector<Vector3i>& indices_,
                                   std::function<bool(const Vector3&)> insideSolveRegion_,
                                   bool computeWeightedNormals=false);

    // performs sampler specific initialization
    void initialize(float normalOffsetForBoundary, bool solveDoubleSided);

    // returns the number of sample points to be generated on the user-specified side of the boundary
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         const GeometricQueries<3>& queries,
                         std::vector<SamplePoint<T, 3>>& samplePts,
                         bool generateBoundaryNormalAlignedSamples=false);

    // getters
    const std::vector<Vector3>& getPositions() const { return positions; }
    const std::vector<Vector3i>& getIndices() const { return indices; }
    const std::vector<Vector3>& getNormals() const { return normals; }
    const CDFTable& getCDFTable(bool returnBoundaryNormalAligned) const {
        return returnBoundaryNormalAligned ? cdfTableNormalAligned : cdfTable;
    }
    float getBoundaryArea(bool returnBoundaryNormalAligned) const {
        return returnBoundaryNormalAligned ? boundaryAreaNormalAligned : boundaryArea;
    }

private:
    // computes normals
    void computeNormals(bool computeWeighted);

    // builds a cdf table for sampling
    void buildCDFTable(CDFTable& table, float& area, float normalOffsetForBoundary);

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         const GeometricQueries<3>& queries,
                         const CDFTable& table, float area,
                         std::vector<SamplePoint<T, 3>>& samplePts);

    // members
    pcg32 rng;
    const std::vector<Vector3>& positions;
    const std::vector<Vector3i>& indices;
    std::function<bool(const Vector3&)> insideSolveRegion;
    std::vector<Vector3> normals;
    CDFTable cdfTable, cdfTableNormalAligned;
    float boundaryArea, boundaryAreaNormalAligned;
};

template <typename T>
std::shared_ptr<BoundarySampler<T, 3>> createUniformTriangleBoundarySampler(
                                        const std::vector<Vector3>& positions,
                                        const std::vector<Vector3i>& indices,
                                        std::function<bool(const Vector3&)> insideSolveRegion,
                                        bool computeWeightedNormals=false);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - improve stratification, since it helps reduce clumping/singular artifacts
// - sample points on the boundary in proportion to dirichlet/neumann/robin boundary values

template <typename T>
inline UniformLineSegmentBoundarySampler<T>::UniformLineSegmentBoundarySampler(const std::vector<Vector2>& positions_,
                                                                               const std::vector<Vector2i>& indices_,
                                                                               std::function<bool(const Vector2&)> insideSolveRegion_,
                                                                               bool computeWeightedNormals):
                                                                               positions(positions_), indices(indices_),
                                                                               insideSolveRegion(insideSolveRegion_),
                                                                               boundaryArea(0.0f), boundaryAreaNormalAligned(0.0f)
{
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    rng = pcg32(seed);
    computeNormals(computeWeightedNormals);
}

template <typename T>
inline void UniformLineSegmentBoundarySampler<T>::computeNormals(bool computeWeighted)
{
    int nPrimitives = (int)indices.size();
    int nPositions = (int)positions.size();
    normals.resize(nPositions, Vector2::Zero());

    for (int i = 0; i < nPrimitives; i++) {
        const Vector2i& index = indices[i];
        const Vector2& p0 = positions[index[0]];
        const Vector2& p1 = positions[index[1]];
        Vector2 n = computeLineSegmentNormal(p0, p1, !computeWeighted);

        normals[index[0]] += n;
        normals[index[1]] += n;
    }

    for (int i = 0; i < nPositions; i++) {
        normals[i].normalize();
    }
}

template <typename T>
inline void UniformLineSegmentBoundarySampler<T>::buildCDFTable(CDFTable& table, float& area,
                                                                float normalOffsetForBoundary)
{
    int nPrimitives = (int)indices.size();
    std::vector<float> weights(nPrimitives, 0.0f);

    for (int i = 0; i < nPrimitives; i++) {
        const Vector2i& index = indices[i];
        Vector2 p0 = positions[index[0]];
        Vector2 p1 = positions[index[1]];
        Vector2 pMid = (p0 + p1)/2.0f;
        Vector2 n = computeLineSegmentNormal(p0, p1, true);

        // don't generate any samples on the boundary outside the solve region
        if (insideSolveRegion(pMid + normalOffsetForBoundary*n)) {
            p0 += normalOffsetForBoundary*normals[index[0]];
            p1 += normalOffsetForBoundary*normals[index[1]];
            weights[i] = computeLineSegmentSurfaceArea(p0, p1);
        }
    }

    area = table.build(weights);
}

template <typename T>
inline void UniformLineSegmentBoundarySampler<T>::initialize(float normalOffsetForBoundary, bool solveDoubleSided)
{
    // build a cdf table for boundary vertices displaced along inward normals
    buildCDFTable(cdfTable, boundaryArea, -1.0f*normalOffsetForBoundary);

    if (solveDoubleSided) {
        // build a cdf table for boundary vertices displaced along outward normals
        buildCDFTable(cdfTableNormalAligned, boundaryAreaNormalAligned, normalOffsetForBoundary);
    }
}

template <typename T>
inline int UniformLineSegmentBoundarySampler<T>::getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples) const
{
    float totalBoundaryArea = boundaryArea + boundaryAreaNormalAligned;
    return boundaryNormalAlignedSamples ? std::ceil(nTotalSamples*boundaryAreaNormalAligned/totalBoundaryArea) :
                                          std::ceil(nTotalSamples*boundaryArea/totalBoundaryArea);
}

template <typename T>
inline void UniformLineSegmentBoundarySampler<T>::generateSamples(int nSamples, SampleType sampleType,
                                                                  float normalOffsetForBoundary,
                                                                  const GeometricQueries<2>& queries,
                                                                  const CDFTable& table, float area,
                                                                  std::vector<SamplePoint<T, 2>>& samplePts)
{
    samplePts.clear();
    if (area > 0.0f) {
        // generate stratified samples for CDF table sampling
        std::vector<float> stratifiedSamples;
        generateStratifiedSamples<1>(stratifiedSamples, nSamples, rng);

        // count the number of times a mesh face is sampled from the CDF table
        std::unordered_map<int, int> indexCount;
        for (int i = 0; i < nSamples; i++) {
            float u = stratifiedSamples[i];
            int offset = table.sample(u);

            if (indexCount.find(offset) == indexCount.end()) {
                indexCount[offset] = 1;

            } else {
                indexCount[offset]++;
            }
        }

        float pdf = 1.0f/area;
        for (auto& kv: indexCount) {
            // generate samples for selected mesh face
            std::vector<float> indexSamples;
            const Vector2i& index = indices[kv.first];
            if (kv.second == 1) {
                indexSamples.emplace_back(rng.nextFloat());

            } else {
                generateStratifiedSamples<1>(indexSamples, kv.second, rng);
            }

            for (int i = 0; i < kv.second; i++) {
                // generate sample point
                Vector2 p0 = positions[index[0]] + normalOffsetForBoundary*normals[index[0]];
                Vector2 p1 = positions[index[1]] + normalOffsetForBoundary*normals[index[1]];
                Vector2 normal = Vector2::Zero();
                float lineSegmentPdf = 0.0f;
                Vector2 pt = samplePointOnLineSegment(p0, p1, &indexSamples[i], normal, lineSegmentPdf);
                float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
                float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

                samplePts.emplace_back(SamplePoint<T, 2>(pt, normal, sampleType,
                                                         EstimationQuantity::Solution,
                                                         pdf, distToAbsorbingBoundary,
                                                         distToReflectingBoundary));
            }
        }

    } else {
        std::cout << "CDF table is empty!" << std::endl;
    }
}

template <typename T>
inline void UniformLineSegmentBoundarySampler<T>::generateSamples(int nSamples, SampleType sampleType,
                                                                  float normalOffsetForBoundary,
                                                                  const GeometricQueries<2>& queries,
                                                                  std::vector<SamplePoint<T, 2>>& samplePts,
                                                                  bool generateBoundaryNormalAlignedSamples)
{
    if (generateBoundaryNormalAlignedSamples) {
        generateSamples(nSamples, sampleType, normalOffsetForBoundary, queries,
                        cdfTableNormalAligned, boundaryAreaNormalAligned, samplePts);

    } else {
        generateSamples(nSamples, sampleType, -1.0f*normalOffsetForBoundary,
                        queries, cdfTable, boundaryArea, samplePts);
    }

    for (int i = 0; i < nSamples; i++) {
        samplePts[i].estimateBoundaryNormalAligned = generateBoundaryNormalAlignedSamples;
    }
}

template <typename T>
std::shared_ptr<BoundarySampler<T, 2>> createUniformLineSegmentBoundarySampler(
                                        const std::vector<Vector2>& positions,
                                        const std::vector<Vector2i>& indices,
                                        std::function<bool(const Vector2&)> insideSolveRegion,
                                        bool computeWeightedNormals)
{
    return std::make_shared<UniformLineSegmentBoundarySampler<T>>(
            positions, indices, insideSolveRegion, computeWeightedNormals);
}

template <typename T>
inline UniformTriangleBoundarySampler<T>::UniformTriangleBoundarySampler(const std::vector<Vector3>& positions_,
                                                                         const std::vector<Vector3i>& indices_,
                                                                         std::function<bool(const Vector3&)> insideSolveRegion_,
                                                                         bool computeWeightedNormals):
                                                                         positions(positions_), indices(indices_),
                                                                         insideSolveRegion(insideSolveRegion_),
                                                                         boundaryArea(0.0f), boundaryAreaNormalAligned(0.0f)
{
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    rng = pcg32(seed);
    computeNormals(computeWeightedNormals);
}

template <typename T>
inline void UniformTriangleBoundarySampler<T>::computeNormals(bool computeWeighted)
{
    int nPrimitives = (int)indices.size();
    int nPositions = (int)positions.size();
    normals.resize(nPositions, Vector3::Zero());

    for (int i = 0; i < nPrimitives; i++) {
        const Vector3i& index = indices[i];
        const Vector3& p0 = positions[index[0]];
        const Vector3& p1 = positions[index[1]];
        const Vector3& p2 = positions[index[2]];
        Vector3 n = computeTriangleNormal(p0, p1, p2, true);

        for (int j = 0; j < 3; j++) {
            const Vector3& p0 = positions[index[(j + 0)%3]];
            const Vector3& p1 = positions[index[(j + 1)%3]];
            const Vector3& p2 = positions[index[(j + 2)%3]];
            float angle = computeWeighted ? computeTriangleAngle(p0, p1, p2) : 1.0f;

            normals[index[j]] += angle*n;
        }
    }

    for (int i = 0; i < nPositions; i++) {
        normals[i].normalize();
    }
}

template <typename T>
inline void UniformTriangleBoundarySampler<T>::buildCDFTable(CDFTable& table, float& area,
                                                             float normalOffsetForBoundary)
{
    int nPrimitives = (int)indices.size();
    std::vector<float> weights(nPrimitives, 0.0f);

    for (int i = 0; i < nPrimitives; i++) {
        const Vector3i& index = indices[i];
        Vector3 p0 = positions[index[0]];
        Vector3 p1 = positions[index[1]];
        Vector3 p2 = positions[index[2]];
        Vector3 pMid = (p0 + p1 + p2)/3.0f;
        Vector3 n = computeTriangleNormal(p0, p1, p2, true);

        // don't generate any samples on the boundary outside the solve region
        if (insideSolveRegion(pMid + normalOffsetForBoundary*n)) {
            p0 += normalOffsetForBoundary*normals[index[0]];
            p1 += normalOffsetForBoundary*normals[index[1]];
            p2 += normalOffsetForBoundary*normals[index[2]];
            weights[i] = computeTriangleSurfaceArea(p0, p1, p2);
        }
    }

    area = table.build(weights);
}

template <typename T>
inline void UniformTriangleBoundarySampler<T>::initialize(float normalOffsetForBoundary, bool solveDoubleSided)
{
    // build a cdf table for boundary vertices displaced along inward normals
    buildCDFTable(cdfTable, boundaryArea, -1.0f*normalOffsetForBoundary);

    if (solveDoubleSided) {
        // build a cdf table for boundary vertices displaced along outward normals
        buildCDFTable(cdfTableNormalAligned, boundaryAreaNormalAligned, normalOffsetForBoundary);
    }
}

template <typename T>
inline int UniformTriangleBoundarySampler<T>::getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples) const
{
    float totalBoundaryArea = boundaryArea + boundaryAreaNormalAligned;
    return boundaryNormalAlignedSamples ? std::ceil(nTotalSamples*boundaryAreaNormalAligned/totalBoundaryArea) :
                                          std::ceil(nTotalSamples*boundaryArea/totalBoundaryArea);
}

template <typename T>
inline void UniformTriangleBoundarySampler<T>::generateSamples(int nSamples, SampleType sampleType,
                                                               float normalOffsetForBoundary,
                                                               const GeometricQueries<3>& queries,
                                                               const CDFTable& table, float area,
                                                               std::vector<SamplePoint<T, 3>>& samplePts)
{
    samplePts.clear();
    if (area > 0.0f) {
        // generate stratified samples for CDF table sampling
        std::vector<float> stratifiedSamples;
        generateStratifiedSamples<1>(stratifiedSamples, nSamples, rng);

        // count the number of times a mesh face is sampled from the CDF table
        std::unordered_map<int, int> indexCount;
        for (int i = 0; i < nSamples; i++) {
            float u = stratifiedSamples[i];
            int offset = table.sample(u);

            if (indexCount.find(offset) == indexCount.end()) {
                indexCount[offset] = 1;

            } else {
                indexCount[offset]++;
            }
        }

        float pdf = 1.0f/area;
        for (auto& kv: indexCount) {
            // generate samples for selected mesh face
            std::vector<float> indexSamples;
            const Vector3i& index = indices[kv.first];
            if (kv.second == 1) {
                indexSamples.emplace_back(rng.nextFloat());
                indexSamples.emplace_back(rng.nextFloat());

            } else {
                generateStratifiedSamples<2>(indexSamples, kv.second, rng);
            }

            for (int i = 0; i < kv.second; i++) {
                // generate sample point
                Vector3 p0 = positions[index[0]] + normalOffsetForBoundary*normals[index[0]];
                Vector3 p1 = positions[index[1]] + normalOffsetForBoundary*normals[index[1]];
                Vector3 p2 = positions[index[2]] + normalOffsetForBoundary*normals[index[2]];
                Vector3 normal = Vector3::Zero();
                float trianglePdf = 0.0f;
                Vector3 pt = samplePointOnTriangle(p0, p1, p2, &indexSamples[2*i], normal, trianglePdf);
                float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
                float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

                samplePts.emplace_back(SamplePoint<T, 3>(pt, normal, sampleType,
                                                         EstimationQuantity::Solution,
                                                         pdf, distToAbsorbingBoundary,
                                                         distToReflectingBoundary));
            }
        }

    } else {
        std::cout << "CDF table is empty!" << std::endl;
    }
}

template <typename T>
inline void UniformTriangleBoundarySampler<T>::generateSamples(int nSamples, SampleType sampleType,
                                                               float normalOffsetForBoundary,
                                                               const GeometricQueries<3>& queries,
                                                               std::vector<SamplePoint<T, 3>>& samplePts,
                                                               bool generateBoundaryNormalAlignedSamples)
{
    if (generateBoundaryNormalAlignedSamples) {
        generateSamples(nSamples, sampleType, normalOffsetForBoundary, queries,
                        cdfTableNormalAligned, boundaryAreaNormalAligned, samplePts);

    } else {
        generateSamples(nSamples, sampleType, -1.0f*normalOffsetForBoundary,
                        queries, cdfTable, boundaryArea, samplePts);
    }

    for (int i = 0; i < nSamples; i++) {
        samplePts[i].estimateBoundaryNormalAligned = generateBoundaryNormalAlignedSamples;
    }
}

template <typename T>
std::shared_ptr<BoundarySampler<T, 3>> createUniformTriangleBoundarySampler(
                                        const std::vector<Vector3>& positions,
                                        const std::vector<Vector3i>& indices,
                                        std::function<bool(const Vector3&)> insideSolveRegion,
                                        bool computeWeightedNormals)
{
    return std::make_shared<UniformTriangleBoundarySampler<T>>(
            positions, indices, insideSolveRegion, computeWeightedNormals);
}

} // zombie

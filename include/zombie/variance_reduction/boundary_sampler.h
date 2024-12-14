// This file defines a BoundarySampler for generating uniformly distributed sample points
// on a 2D or 3D boundary mesh defined by a set of vertices and indices. These sample points
// are required by the Boundary Value Caching (BVC) and Reverse Walk Splatting (RWS) techniques
// for reducing variance of the walk-on-spheres and walk-on-stars estimators. BVC and RWS currently
// require sample points on the absorbing boundary to be displaced slightly along the boundary normal.

#pragma once

#include <zombie/point_estimation/common.h>
#include <unordered_map>

namespace zombie {

template <typename T, size_t DIM>
class BoundarySampler {
public:
    // performs any sampler specific initialization
    virtual void initialize(float normalOffsetForBoundary, bool solveDoubleSided) = 0;

    // returns the number of sample points to be generated on the user-specified side of the boundary
    virtual int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const = 0;

    // generates sample points on the boundary
    virtual void generateSamples(int nSamples, SampleType sampleType,
                                 float normalOffsetForBoundary,
                                 std::vector<SamplePoint<T, DIM>>& samplePts,
                                 bool generateBoundaryNormalAlignedSamples=false) = 0;
};

template <typename T>
class UniformLineSegmentBoundarySampler : public BoundarySampler<T, 2> {
public:
    // constructor
    UniformLineSegmentBoundarySampler(const std::vector<Vector2>& positions_,
                                      const std::vector<std::vector<size_t>>& indices_,
                                      const GeometricQueries<2>& queries_,
                                      const std::function<bool(const Vector2&)>& insideSolveRegion_,
                                      bool computeWeightedNormals=false);

    // performs sampler specific initialization
    void initialize(float normalOffsetForBoundary, bool solveDoubleSided);

    // returns the number of sample points to be generated on the user-specified side of the boundary
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         std::vector<SamplePoint<T, 2>>& samplePts,
                         bool generateBoundaryNormalAlignedSamples=false);

private:
    // computes normals
    void computeNormals(bool computeWeighted);

    // builds a cdf table for sampling
    void buildCDFTable(CDFTable& table, float& area, float normalOffsetForBoundary);

    // generates uniformly distributed sample points on the boundary
    void generateSamples(const CDFTable& table, float area,
                         int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         std::vector<SamplePoint<T, 2>>& samplePts);

    // members
    pcg32 sampler;
    const std::vector<Vector2>& positions;
    const std::vector<std::vector<size_t>>& indices;
    const GeometricQueries<2>& queries;
    const std::function<bool(const Vector2&)>& insideSolveRegion;
    std::vector<Vector2> normals;
    CDFTable cdfTable, cdfTableNormalAligned;
    float boundaryArea, boundaryAreaNormalAligned;
};

template <typename T>
class UniformTriangleBoundarySampler : public BoundarySampler<T, 3> {
public:
    // constructor
    UniformTriangleBoundarySampler(const std::vector<Vector3>& positions_,
                                   const std::vector<std::vector<size_t>>& indices_,
                                   const GeometricQueries<3>& queries_,
                                   const std::function<bool(const Vector3&)>& insideSolveRegion_,
                                   bool computeWeightedNormals=false);

    // performs sampler specific initialization
    void initialize(float normalOffsetForBoundary, bool solveDoubleSided);

    // returns the number of sample points to be generated on the user-specified side of the boundary
    int getSampleCount(int nTotalSamples, bool boundaryNormalAlignedSamples=false) const;

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         std::vector<SamplePoint<T, 3>>& samplePts,
                         bool generateBoundaryNormalAlignedSamples=false);

private:
    // computes normals
    void computeNormals(bool computeWeighted);

    // builds a cdf table for sampling
    void buildCDFTable(CDFTable& table, float& area, float normalOffsetForBoundary);

    // generates uniformly distributed sample points on the boundary
    void generateSamples(const CDFTable& table, float area,
                         int nSamples, SampleType sampleType,
                         float normalOffsetForBoundary,
                         std::vector<SamplePoint<T, 3>>& samplePts);

    // members
    pcg32 sampler;
    const std::vector<Vector3>& positions;
    const std::vector<std::vector<size_t>>& indices;
    const GeometricQueries<3>& queries;
    const std::function<bool(const Vector3&)>& insideSolveRegion;
    std::vector<Vector3> normals;
    CDFTable cdfTable, cdfTableNormalAligned;
    float boundaryArea, boundaryAreaNormalAligned;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - improve stratification, since it helps reduce clumping/singular artifacts
// - sample points on the boundary in proportion to dirichlet/neumann/robin boundary values

class UniformLineSegmentSampler {
public:
    // returns normal
    static Vector2 normal(const Vector2& pa, const Vector2& pb, bool normalize) {
        Vector2 s = pb - pa;
        Vector2 n(s[1], -s[0]);

        return normalize ? n.normalized() : n;
    }

    // returns surface area
    static float surfaceArea(const Vector2& pa, const Vector2& pb) {
        return normal(pa, pb, false).norm();
    }

    // samples point uniformly
    static float samplePoint(const Vector2& pa, const Vector2& pb, float *u,
                             Vector2& pt, Vector2& n) {
        Vector2 s = pb - pa;
        pt = pa + u[0]*s;
        n = Vector2(s[1], -s[0]);
        float norm = n.norm();
        n /= norm;

        return 1.0f/norm;
    }
    static float samplePoint(const Vector2& pa, const Vector2& pb, pcg32& sampler,
                             Vector2& pt, Vector2& n) {
        float u[1] = { sampler.nextFloat() };
        return samplePoint(pa, pb, u, pt, n);
    }
};

template <typename T>
inline UniformLineSegmentBoundarySampler<T>::UniformLineSegmentBoundarySampler(const std::vector<Vector2>& positions_,
                                                                               const std::vector<std::vector<size_t>>& indices_,
                                                                               const GeometricQueries<2>& queries_,
                                                                               const std::function<bool(const Vector2&)>& insideSolveRegion_,
                                                                               bool computeWeightedNormals):
                                                                               positions(positions_), indices(indices_),
                                                                               queries(queries_), insideSolveRegion(insideSolveRegion_),
                                                                               boundaryArea(0.0f), boundaryAreaNormalAligned(0.0f)
{
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    sampler = pcg32(seed);
    computeNormals(computeWeightedNormals);
}

template <typename T>
inline void UniformLineSegmentBoundarySampler<T>::computeNormals(bool computeWeighted)
{
    int nPrimitives = (int)indices.size();
    int nPositions = (int)positions.size();
    normals.resize(nPositions, Vector2::Zero());

    for (int i = 0; i < nPrimitives; i++) {
        const std::vector<size_t>& index = indices[i];
        const Vector2& p0 = positions[index[0]];
        const Vector2& p1 = positions[index[1]];
        Vector2 n = UniformLineSegmentSampler::normal(p0, p1, !computeWeighted);

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
        const std::vector<size_t>& index = indices[i];
        Vector2 p0 = positions[index[0]];
        Vector2 p1 = positions[index[1]];
        Vector2 pMid = (p0 + p1)/2.0f;
        Vector2 n = UniformLineSegmentSampler::normal(p0, p1, true);

        // don't generate any samples on the boundary outside the solve region
        if (insideSolveRegion(pMid + normalOffsetForBoundary*n)) {
            p0 += normalOffsetForBoundary*normals[index[0]];
            p1 += normalOffsetForBoundary*normals[index[1]];
            weights[i] = UniformLineSegmentSampler::surfaceArea(p0, p1);
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
inline void UniformLineSegmentBoundarySampler<T>::generateSamples(const CDFTable& table, float area,
                                                                  int nSamples, SampleType sampleType,
                                                                  float normalOffsetForBoundary,
                                                                  std::vector<SamplePoint<T, 2>>& samplePts)
{
    samplePts.clear();
    if (area > 0.0f) {
        // generate stratified samples for CDF table sampling
        std::vector<float> stratifiedSamples;
        generateStratifiedSamples<1>(stratifiedSamples, nSamples, sampler);

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
            const std::vector<size_t>& index = indices[kv.first];
            if (kv.second == 1) {
                indexSamples.emplace_back(sampler.nextFloat());

            } else {
                generateStratifiedSamples<1>(indexSamples, kv.second, sampler);
            }

            for (int i = 0; i < kv.second; i++) {
                // generate sample point
                Vector2 pt = Vector2::Zero();
                Vector2 normal = Vector2::Zero();
                Vector2 p0 = positions[index[0]] + normalOffsetForBoundary*normals[index[0]];
                Vector2 p1 = positions[index[1]] + normalOffsetForBoundary*normals[index[1]];
                UniformLineSegmentSampler::samplePoint(p0, p1, &indexSamples[i], pt, normal);
                float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
                float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

                samplePts.emplace_back(SamplePoint<T, 2>(pt, normal, sampleType,
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
                                                                  std::vector<SamplePoint<T, 2>>& samplePts,
                                                                  bool generateBoundaryNormalAlignedSamples)
{
    if (generateBoundaryNormalAlignedSamples) {
        generateSamples(cdfTableNormalAligned, boundaryAreaNormalAligned,
                        nSamples, sampleType, normalOffsetForBoundary, samplePts);

    } else {
        generateSamples(cdfTable, boundaryArea, nSamples, sampleType,
                        -1.0f*normalOffsetForBoundary, samplePts);
    }

    for (int i = 0; i < nSamples; i++) {
        samplePts[i].estimateBoundaryNormalAligned = generateBoundaryNormalAlignedSamples;
    }
}

class UniformTriangleSampler {
public:
    // returns normal
    static Vector3 normal(const Vector3& pa, const Vector3& pb, const Vector3& pc, bool normalize) {
        Vector3 n = (pb - pa).cross(pc - pa);
        return normalize ? n.normalized() : n;
    }

    // returns surface area
    static float surfaceArea(const Vector3& pa, const Vector3& pb, const Vector3& pc) {
        return 0.5f*normal(pa, pb, pc, false).norm();
    }

    // returns angle
    static float angle(const Vector3& pa, const Vector3& pb, const Vector3& pc) {
        Vector3 u = (pb - pa).normalized();
        Vector3 v = (pc - pa).normalized();

        return std::acos(std::max(-1.0f, std::min(1.0f, u.dot(v))));
    }

    // samples point uniformly
    static float samplePoint(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                             float *u, Vector3& pt, Vector3& n) {
        float u1 = std::sqrt(u[0]);
        float u2 = u[1];
        float a = 1.0f - u1;
        float b = u2*u1;
        float c = 1.0f - a - b;
        pt = pa*a + pb*b + pc*c;
        n = (pb - pa).cross(pc - pa);
        float norm = n.norm();
        n /= norm;

        return 2.0f/norm;
    }
    static float samplePoint(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                             pcg32& sampler, Vector3& pt, Vector3& n) {
        float u[2] = { sampler.nextFloat(), sampler.nextFloat() };
        return samplePoint(pa, pb, pc, u, pt, n);
    }
};

template <typename T>
inline UniformTriangleBoundarySampler<T>::UniformTriangleBoundarySampler(const std::vector<Vector3>& positions_,
                                                                         const std::vector<std::vector<size_t>>& indices_,
                                                                         const GeometricQueries<3>& queries_,
                                                                         const std::function<bool(const Vector3&)>& insideSolveRegion_,
                                                                         bool computeWeightedNormals):
                                                                         positions(positions_), indices(indices_),
                                                                         queries(queries_), insideSolveRegion(insideSolveRegion_),
                                                                         boundaryArea(0.0f), boundaryAreaNormalAligned(0.0f)
{
    auto now = std::chrono::high_resolution_clock::now();
    uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
    sampler = pcg32(seed);
    computeNormals(computeWeightedNormals);
}

template <typename T>
inline void UniformTriangleBoundarySampler<T>::computeNormals(bool computeWeighted)
{
    int nPrimitives = (int)indices.size();
    int nPositions = (int)positions.size();
    normals.resize(nPositions, Vector3::Zero());

    for (int i = 0; i < nPrimitives; i++) {
        const std::vector<size_t>& index = indices[i];
        const Vector3& p0 = positions[index[0]];
        const Vector3& p1 = positions[index[1]];
        const Vector3& p2 = positions[index[2]];
        Vector3 n = UniformTriangleSampler::normal(p0, p1, p2, true);

        for (int j = 0; j < 3; j++) {
            const Vector3& p0 = positions[index[(j + 0)%3]];
            const Vector3& p1 = positions[index[(j + 1)%3]];
            const Vector3& p2 = positions[index[(j + 2)%3]];
            float angle = computeWeighted ? UniformTriangleSampler::angle(p0, p1, p2) : 1.0f;

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
        const std::vector<size_t>& index = indices[i];
        Vector3 p0 = positions[index[0]];
        Vector3 p1 = positions[index[1]];
        Vector3 p2 = positions[index[2]];
        Vector3 pMid = (p0 + p1 + p2)/3.0f;
        Vector3 n = UniformTriangleSampler::normal(p0, p1, p2, true);

        // don't generate any samples on the boundary outside the solve region
        if (insideSolveRegion(pMid + normalOffsetForBoundary*n)) {
            p0 += normalOffsetForBoundary*normals[index[0]];
            p1 += normalOffsetForBoundary*normals[index[1]];
            p2 += normalOffsetForBoundary*normals[index[2]];
            weights[i] = UniformTriangleSampler::surfaceArea(p0, p1, p2);
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
inline void UniformTriangleBoundarySampler<T>::generateSamples(const CDFTable& table, float area,
                                                               int nSamples, SampleType sampleType,
                                                               float normalOffsetForBoundary,
                                                               std::vector<SamplePoint<T, 3>>& samplePts)
{
    samplePts.clear();
    if (area > 0.0f) {
        // generate stratified samples for CDF table sampling
        std::vector<float> stratifiedSamples;
        generateStratifiedSamples<1>(stratifiedSamples, nSamples, sampler);

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
            const std::vector<size_t>& index = indices[kv.first];
            if (kv.second == 1) {
                indexSamples.emplace_back(sampler.nextFloat());
                indexSamples.emplace_back(sampler.nextFloat());

            } else {
                generateStratifiedSamples<2>(indexSamples, kv.second, sampler);
            }

            for (int i = 0; i < kv.second; i++) {
                // generate sample point
                Vector3 pt = Vector3::Zero();
                Vector3 normal = Vector3::Zero();
                Vector3 p0 = positions[index[0]] + normalOffsetForBoundary*normals[index[0]];
                Vector3 p1 = positions[index[1]] + normalOffsetForBoundary*normals[index[1]];
                Vector3 p2 = positions[index[2]] + normalOffsetForBoundary*normals[index[2]];
                UniformTriangleSampler::samplePoint(p0, p1, p2, &indexSamples[2*i], pt, normal);
                float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
                float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

                samplePts.emplace_back(SamplePoint<T, 3>(pt, normal, sampleType,
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
                                                               std::vector<SamplePoint<T, 3>>& samplePts,
                                                               bool generateBoundaryNormalAlignedSamples)
{
    if (generateBoundaryNormalAlignedSamples) {
        generateSamples(cdfTableNormalAligned, boundaryAreaNormalAligned,
                        nSamples, sampleType, normalOffsetForBoundary, samplePts);

    } else {
        generateSamples(cdfTable, boundaryArea, nSamples, sampleType,
                        -1.0f*normalOffsetForBoundary, samplePts);
    }

    for (int i = 0; i < nSamples; i++) {
        samplePts[i].estimateBoundaryNormalAligned = generateBoundaryNormalAlignedSamples;
    }
}

} // zombie

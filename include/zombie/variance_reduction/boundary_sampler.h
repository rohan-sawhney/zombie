// This file defines a BoundarySampler for generating uniformly distributed sample points
// on a 2D or 3D boundary mesh defined by a set of vertices and indices. These sample points
// are required by the Boundary Value Caching (BVC) technique for reducing variance of the
// walk-on-spheres and walk-on-stars estimators. BVC currently requires the sample points on
// the absorbing boundary to be displaced slightly along the boundary normal.

#pragma once

#include <zombie/point_estimation/walk_on_stars.h>
#include <unordered_map>

namespace zombie {

// NOTE: currently specialized to line segments in 2D and triangles in 3D
template <typename T, size_t DIM>
class BoundarySampler {
public:
    // constructor
    BoundarySampler(const std::vector<Vector<DIM>>& positions_,
                    const std::vector<std::vector<size_t>>& indices_,
                    const GeometricQueries<DIM>& queries_,
                    const std::function<bool(const Vector<DIM>&)>& insideSolveRegion_,
                    const std::function<bool(const Vector<DIM>&)>& onReflectingBoundary_);

    // initialize sampler
    void initialize(float normalOffsetForAbsorbingBoundary,
                    float normalOffsetForReflectingBoundary,
                    bool solveDoubleSided,
                    bool computeWeightedNormals=false);

    // generates uniformly distributed sample points on the boundary
    void generateSamples(int nTotalSamples,
                         float normalOffsetForAbsorbingBoundary,
                         float normalOffsetForReflectingBoundary,
                         bool solveDoubleSided, T initVal,
                         std::vector<SamplePoint<T, DIM>>& samplePts,
                         std::vector<SamplePoint<T, DIM>>& samplePtsNormalAligned);

private:
    // computes normals
    void computeNormals(bool computeWeighted);

    // builds a cdf table for sampling; FUTURE: to get truly unbiased results,
    // introduce additional primitives at the absorbing-reflecting boundary
    // interface that can be sampled
    void buildCDFTable(CDFTable& table, float& totalArea,
                       float normalOffsetForAbsorbingBoundary,
                       float normalOffsetForReflectingBoundary);

    // generates uniformly distributed sample points on the boundary
    void generateSamples(const CDFTable& table, int nSamples, float totalArea,
                         float normalOffsetForAbsorbingBoundary,
                         float normalOffsetForReflectingBoundary, T initVal,
                         std::vector<SamplePoint<T, DIM>>& samplePts);

    // members
    pcg32 sampler;
    const std::vector<Vector<DIM>>& positions;
    const std::vector<std::vector<size_t>>& indices;
    const GeometricQueries<DIM>& queries;
    const std::function<bool(const Vector<DIM>&)>& insideSolveRegion;
    const std::function<bool(const Vector<DIM>&)>& onReflectingBoundary;
    std::vector<Vector<DIM>> normals;
    CDFTable cdfTable, cdfTableNormalAligned;
    float boundaryArea, boundaryAreaNormalAligned;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - improve stratification, since it helps reduce clumping/singular artifacts
// - sample points on the boundary in proportion to dirichlet/neumann/robin boundary values

template <typename T, size_t DIM>
inline BoundarySampler<T, DIM>::BoundarySampler(const std::vector<Vector<DIM>>& positions_,
                                                const std::vector<std::vector<size_t>>& indices_,
                                                const GeometricQueries<DIM>& queries_,
                                                const std::function<bool(const Vector<DIM>&)>& insideSolveRegion_,
                                                const std::function<bool(const Vector<DIM>&)>& onReflectingBoundary_):
                                                positions(positions_), indices(indices_), queries(queries_),
                                                insideSolveRegion(insideSolveRegion_),
                                                onReflectingBoundary(onReflectingBoundary_),
                                                boundaryArea(0.0f), boundaryAreaNormalAligned(0.0f) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    sampler = pcg32(seed);
}

template <typename T, size_t DIM>
inline void BoundarySampler<T, DIM>::initialize(float normalOffsetForAbsorbingBoundary,
                                                float normalOffsetForReflectingBoundary,
                                                bool solveDoubleSided,
                                                bool computeWeightedNormals) {
    // compute normals
    computeNormals(computeWeightedNormals);

    // build a cdf table for boundary vertices displaced along inward normals
    buildCDFTable(cdfTable, boundaryArea, -1.0f*normalOffsetForAbsorbingBoundary,
                  -1.0f*normalOffsetForReflectingBoundary);

    if (solveDoubleSided) {
        // build a cdf table for boundary vertices displaced along outward normals
        buildCDFTable(cdfTableNormalAligned, boundaryAreaNormalAligned,
                      normalOffsetForAbsorbingBoundary,
                      normalOffsetForReflectingBoundary);
    }
}

template <typename T, size_t DIM>
inline void BoundarySampler<T, DIM>::generateSamples(int nTotalSamples,
                                                     float normalOffsetForAbsorbingBoundary,
                                                     float normalOffsetForReflectingBoundary,
                                                     bool solveDoubleSided, T initVal,
                                                     std::vector<SamplePoint<T, DIM>>& samplePts,
                                                     std::vector<SamplePoint<T, DIM>>& samplePtsNormalAligned) {
    if (solveDoubleSided) {
        // decide sample count split based on boundary areas
        float totalBoundaryArea = boundaryArea + boundaryAreaNormalAligned;
        int nSamples = std::ceil(nTotalSamples*boundaryArea/totalBoundaryArea);
        int nSamplesNormalAligned = std::ceil(nTotalSamples*boundaryAreaNormalAligned/totalBoundaryArea);

        // generate samples
        generateSamples(cdfTable, nSamples, boundaryArea, -1.0f*normalOffsetForAbsorbingBoundary,
                        -1.0f*normalOffsetForReflectingBoundary, initVal, samplePts);
        generateSamples(cdfTableNormalAligned, nSamplesNormalAligned,
                        boundaryAreaNormalAligned, normalOffsetForAbsorbingBoundary,
                        normalOffsetForReflectingBoundary, initVal, samplePtsNormalAligned);

    } else {
        generateSamples(cdfTable, nTotalSamples, boundaryArea, -1.0f*normalOffsetForAbsorbingBoundary,
                        -1.0f*normalOffsetForReflectingBoundary, initVal, samplePts);
    }
}

template <typename T, size_t DIM>
inline void BoundarySampler<T, DIM>::computeNormals(bool computeWeighted) {
    int nPrimitives = (int)indices.size();
    int nPositions = (int)positions.size();
    normals.clear();
    normals.resize(nPositions, Vector<DIM>::Zero());

    for (int i = 0; i < nPrimitives; i++) {
        const std::vector<size_t>& index = indices[i];

        if (DIM == 2) {
            const Vector<DIM>& pa = positions[index[0]];
            const Vector<DIM>& pb = positions[index[1]];
            Vector<DIM> n = lineSegmentNormal<DIM>(pa, pb, !computeWeighted);

            normals[index[0]] += n;
            normals[index[1]] += n;

        } else if (DIM == 3) {
            const Vector<DIM>& pa = positions[index[0]];
            const Vector<DIM>& pb = positions[index[1]];
            const Vector<DIM>& pc = positions[index[2]];
            Vector<DIM> n = triangleNormal<DIM>(pa, pb, pc, true);

            for (int j = 0; j < 3; j++) {
                const Vector<DIM>& p0 = positions[index[(j + 0)%3]];
                const Vector<DIM>& p1 = positions[index[(j + 1)%3]];
                const Vector<DIM>& p2 = positions[index[(j + 2)%3]];
                float angle = computeWeighted ? triangleAngle<DIM>(p0, p1, p2) : 1.0f;

                normals[index[j]] += angle*n;
            }
        }
    }

    for (int i = 0; i < nPositions; i++) {
        normals[i].normalize();
    }
}

template <typename T, size_t DIM>
inline void BoundarySampler<T, DIM>::buildCDFTable(CDFTable& table, float& totalArea,
                                                   float normalOffsetForAbsorbingBoundary,
                                                   float normalOffsetForReflectingBoundary) {
    int nPrimitives = (int)indices.size();
    std::vector<float> weights(nPrimitives, 0.0f);

    for (int i = 0; i < nPrimitives; i++) {
        const std::vector<size_t>& index = indices[i];

        if (DIM == 2) {
            Vector<DIM> pa = positions[index[0]];
            Vector<DIM> pb = positions[index[1]];
            Vector<DIM> pMid = (pa + pb)/2.0f;
            Vector<DIM> n = lineSegmentNormal<DIM>(pa, pb, true);

            // don't generate any samples on the boundary outside the solve region
            if (onReflectingBoundary(pMid)) {
                if (insideSolveRegion(pMid + normalOffsetForReflectingBoundary*n)) {
                    pa += normalOffsetForReflectingBoundary*normals[index[0]];
                    pb += normalOffsetForReflectingBoundary*normals[index[1]];
                    weights[i] = lineSegmentSurfaceArea<DIM>(pa, pb);
                }

            } else {
                if (insideSolveRegion(pMid + normalOffsetForAbsorbingBoundary*n)) {
                    pa += normalOffsetForAbsorbingBoundary*normals[index[0]];
                    pb += normalOffsetForAbsorbingBoundary*normals[index[1]];
                    weights[i] = lineSegmentSurfaceArea<DIM>(pa, pb);
                }
            }

        } else if (DIM == 3) {
            Vector<DIM> pa = positions[index[0]];
            Vector<DIM> pb = positions[index[1]];
            Vector<DIM> pc = positions[index[2]];
            Vector<DIM> pMid = (pa + pb + pc)/3.0f;
            Vector<DIM> n = triangleNormal<DIM>(pa, pb, pc, true);

            // don't generate any samples on the boundary outside the solve region
            if (onReflectingBoundary(pMid)) {
                if (insideSolveRegion(pMid + normalOffsetForReflectingBoundary*n)) {
                    pa += normalOffsetForReflectingBoundary*normals[index[0]];
                    pb += normalOffsetForReflectingBoundary*normals[index[1]];
                    pc += normalOffsetForReflectingBoundary*normals[index[2]];
                    weights[i] = triangleSurfaceArea<DIM>(pa, pb, pc);
                }

            } else {
                if (insideSolveRegion(pMid + normalOffsetForAbsorbingBoundary*n)) {
                    pa += normalOffsetForAbsorbingBoundary*normals[index[0]];
                    pb += normalOffsetForAbsorbingBoundary*normals[index[1]];
                    pc += normalOffsetForAbsorbingBoundary*normals[index[2]];
                    weights[i] = triangleSurfaceArea<DIM>(pa, pb, pc);
                }
            }
        }
    }

    totalArea = table.build(weights);
}

template <typename T, size_t DIM>
inline void BoundarySampler<T, DIM>::generateSamples(const CDFTable& table, int nSamples, float totalArea,
                                                     float normalOffsetForAbsorbingBoundary,
                                                     float normalOffsetForReflectingBoundary, T initVal,
                                                     std::vector<SamplePoint<T, DIM>>& samplePts) {
    samplePts.clear();
    float pdf = 1.0f/totalArea;

    if (totalArea > 0.0f) {
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

        for (auto& kv: indexCount) {
            // generate samples for selected mesh face
            std::vector<float> indexSamples;
            const std::vector<size_t>& index = indices[kv.first];
            if (kv.second == 1) {
                for (int i = 0; i < DIM - 1; i++) {
                    indexSamples.emplace_back(sampler.nextFloat());
                }

            } else {
                generateStratifiedSamples<DIM - 1>(indexSamples, kv.second, sampler);
            }

            for (int i = 0; i < kv.second; i++) {
                // generate sample point
                SampleType sampleType;
                Vector<DIM> pt = Vector<DIM>::Zero();
                Vector<DIM> normal = Vector<DIM>::Zero();

                if (DIM == 2) {
                    Vector<DIM> pa = positions[index[0]];
                    Vector<DIM> pb = positions[index[1]];
                    Vector<DIM> pMid = (pa + pb)/2.0f;

                    if (onReflectingBoundary(pMid)) {
                        sampleType = SampleType::OnReflectingBoundary;
                        pa += normalOffsetForReflectingBoundary*normals[index[0]];
                        pb += normalOffsetForReflectingBoundary*normals[index[1]];

                    } else {
                        sampleType = SampleType::OnAbsorbingBoundary;
                        pa += normalOffsetForAbsorbingBoundary*normals[index[0]];
                        pb += normalOffsetForAbsorbingBoundary*normals[index[1]];
                    }

                    sampleLineSegmentUniformly<DIM>(pa, pb, &indexSamples[(DIM - 1)*i], pt, normal);

                } else if (DIM == 3) {
                    Vector<DIM> pa = positions[index[0]];
                    Vector<DIM> pb = positions[index[1]];
                    Vector<DIM> pc = positions[index[2]];
                    Vector<DIM> pMid = (pa + pb + pc)/3.0f;

                    if (onReflectingBoundary(pMid)) {
                        sampleType = SampleType::OnReflectingBoundary;
                        pa += normalOffsetForReflectingBoundary*normals[index[0]];
                        pb += normalOffsetForReflectingBoundary*normals[index[1]];
                        pc += normalOffsetForReflectingBoundary*normals[index[2]];

                    } else {
                        sampleType = SampleType::OnAbsorbingBoundary;
                        pa += normalOffsetForAbsorbingBoundary*normals[index[0]];
                        pb += normalOffsetForAbsorbingBoundary*normals[index[1]];
                        pc += normalOffsetForAbsorbingBoundary*normals[index[2]];
                    }

                    sampleTriangleUniformly<DIM>(pa, pb, pc, &indexSamples[(DIM - 1)*i], pt, normal);
                }

                float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
                float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);
                samplePts.emplace_back(SamplePoint<T, DIM>(pt, normal, sampleType,
                                                           pdf, distToAbsorbingBoundary,
                                                           distToReflectingBoundary, initVal));
            }
        }

        if (normalOffsetForAbsorbingBoundary > 0.0f || normalOffsetForReflectingBoundary > 0.0f) {
            // invert the orientation of the boundary normals during estimation,
            // with boundary vertices displaced along these normals
            for (int i = 0; i < nSamples; i++) {
                samplePts[i].estimateBoundaryNormalAligned = true;
            }
        }

    } else {
        std::cout << "CDF table is empty!" << std::endl;
    }
}

} // zombie

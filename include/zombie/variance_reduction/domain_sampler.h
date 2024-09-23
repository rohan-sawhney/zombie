// This file defines a DomainSampler for generating uniformly distributed sample points
// in a 2D or 3D domain. These sample points are required by the Boundary Value Caching (BVC)
// technique for reducing variance of the walk-on-spheres and walk-on-stars estimators for
// PDEs with non-zero source.

#pragma once

#include <zombie/core/pde.h>
#include <zombie/core/geometric_queries.h>
#include <zombie/core/sampling.h>

namespace zombie {

template <typename T, size_t DIM>
class DomainSampler {
public:
    // constructor
    DomainSampler(const GeometricQueries<DIM>& queries_,
                  const std::function<bool(const Vector<DIM>&)>& insideSolveRegion_,
                  const Vector<DIM>& solveRegionMin_,
                  const Vector<DIM>& solveRegionMax_,
                  float solveRegionVolume_);

    // generates uniformly distributed sample points inside the solve region;
    // NOTE: may not generate exactly the requested number of samples when the
    // solve region volume does not match the volume of its bounding extents
    void generateSamples(const PDE<T, DIM>& pde, int nTotalSamples,
                         std::vector<SamplePoint<T, DIM>>& samplePts);

protected:
    // members
    pcg32 sampler;
    const GeometricQueries<DIM>& queries;
    const std::function<bool(const Vector<DIM>&)>& insideSolveRegion;
    const Vector<DIM>& solveRegionMin;
    const Vector<DIM>& solveRegionMax;
    float solveRegionVolume;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - improve stratification, since it helps reduce clumping/singular artifacts
// - sample points in the domain in proportion to source values

template <typename T, size_t DIM>
inline DomainSampler<T, DIM>::DomainSampler(const GeometricQueries<DIM>& queries_,
                                            const std::function<bool(const Vector<DIM>&)>& insideSolveRegion_,
                                            const Vector<DIM>& solveRegionMin_,
                                            const Vector<DIM>& solveRegionMax_,
                                            float solveRegionVolume_):
                                            queries(queries_),
                                            insideSolveRegion(insideSolveRegion_),
                                            solveRegionMin(solveRegionMin_),
                                            solveRegionMax(solveRegionMax_),
                                            solveRegionVolume(solveRegionVolume_) {
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    sampler = pcg32(seed);
}

template <typename T, size_t DIM>
inline void DomainSampler<T, DIM>::generateSamples(const PDE<T, DIM>& pde, int nTotalSamples,
                                                   std::vector<SamplePoint<T, DIM>>& samplePts) {
    // initialize sample points
    samplePts.clear();
    Vector<DIM> regionExtent = solveRegionMax - solveRegionMin;
    float pdf = 1.0f/solveRegionVolume;

    // generate stratified samples
    std::vector<float> stratifiedSamples;
    int nStratifiedSamples = nTotalSamples;
    if (solveRegionVolume > 0.0f) nStratifiedSamples *= regionExtent.prod()*pdf;
    generateStratifiedSamples<DIM>(stratifiedSamples, nStratifiedSamples, sampler);

    // generate sample points inside the solve region
    for (int i = 0; i < nStratifiedSamples; i++) {
        Vector<DIM> randomVector = Vector<DIM>::Zero();
        for (int j = 0; j < DIM; j++) randomVector[j] = stratifiedSamples[DIM*i + j];
        Vector<DIM> pt = (solveRegionMin.array() + regionExtent.array()*randomVector.array()).matrix();
        if (!insideSolveRegion(pt)) continue;

        T source = pde.source(pt);
        float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
        float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);
        SamplePoint<T, DIM> samplePt(pt, Vector<DIM>::Zero(), SampleType::InDomain, pdf,
                                     distToAbsorbingBoundary, distToReflectingBoundary);
        samplePt.source = source;
        samplePts.emplace_back(samplePt);
    }
}

} // zombie

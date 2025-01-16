// This file implements a "reverse walk" splatting technique for reducing variance
// of the walk-on-spheres and walk-on-stars estimators at a set of user-selected
// evaluation points.
//
// Resources:
// - A Bidirectional Formulation for Walk on Spheres [2022]
// - Walkin’ Robin: Walk on Stars with Robin Boundary Conditions [2024]

#pragma once

#include <zombie/point_estimation/reverse_walk_on_stars.h>
#include "oneapi/tbb/spin_mutex.h"

namespace zombie {

namespace rws {

template <typename T, size_t DIM>
struct EvaluationPoint {
    // constructors
    EvaluationPoint();
    EvaluationPoint(const Vector<DIM>& pt_,
                    const Vector<DIM>& normal_,
                    SampleType type_,
                    float distToAbsorbingBoundary_,
                    float distToReflectingBoundary_);

    // returns estimated solution
    T getEstimatedSolution(int nAbsorbingBoundarySamples,
                           int nAbsorbingBoundaryNormalAlignedSamples,
                           int nReflectingBoundarySamples,
                           int nReflectingBoundaryNormalAlignedSamples,
                           int nSourceSamples) const;

    // resets state
    void reset();

    // members
    Vector<DIM> pt;
    Vector<DIM> normal;
    SampleType type;
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;
    float totalPoissonKernelContribution;
    T totalAbsorbingBoundaryContribution;
    T totalAbsorbingBoundaryNormalAlignedContribution;
    T totalReflectingBoundaryContribution;
    T totalReflectingBoundaryNormalAlignedContribution;
    T totalSourceContribution;
    std::shared_ptr<tbb::spin_mutex> mutex;
};

template <typename T, size_t DIM, typename NearestNeighborFinder>
class ReverseWalkOnStarsSolver {
public:
    // constructor
    ReverseWalkOnStarsSolver(const PDE<T, DIM>& pde_,
                             const GeometricQueries<DIM>& queries_,
                             const std::unique_ptr<BoundarySampler<T, DIM>>& absorbingBoundarySampler_,
                             const std::unique_ptr<BoundarySampler<T, DIM>>& reflectingBoundarySampler_,
                             const std::unique_ptr<DomainSampler<T, DIM>>& domainSampler_,
                             const float& normalOffsetForAbsorbingBoundary_,
                             const float& radiusClamp_,
                             const float& kernelRegularization_,
                             std::vector<EvaluationPoint<T, DIM>>& evalPts_);

    // updates internal solver state after evaluation points have been modified
    // (called automatically by the constructor)
    void modifiedEvaluationPoints();

    // generates boundary and domain samples
    void generateSamples(int absorbingBoundarySampleCount,
                         int reflectingBoundarySampleCount,
                         int domainSampleCount,
                         bool solveDoubleSided);

    // splats contributions to evaluation points
    void solve(const WalkSettings& walkSettings,
               bool runSingleThreaded=false,
               std::function<void(int,int)> reportProgress={});

    // returns the boundary and domain sample points
    const std::vector<SamplePoint<T, DIM>>& getAbsorbingBoundarySamplePts(bool returnBoundaryNormalAligned = false) const;
    const std::vector<SamplePoint<T, DIM>>& getReflectingBoundarySamplePts(bool returnBoundaryNormalAligned = false) const;
    const std::vector<SamplePoint<T, DIM>>& getDomainSamplePts() const;

    // returns the number of boundary and domain sample points
    int getAbsorbingBoundarySampleCount(bool returnBoundaryNormalAligned = false) const;
    int getReflectingBoundarySampleCount(bool returnBoundaryNormalAligned = false) const;
    int getDomainSampleCount() const;

protected:
    // members
    const PDE<T, DIM>& pde;
    const GeometricQueries<DIM>& queries;
    const std::unique_ptr<BoundarySampler<T, DIM>>& absorbingBoundarySampler;
    const std::unique_ptr<BoundarySampler<T, DIM>>& reflectingBoundarySampler;
    const std::unique_ptr<DomainSampler<T, DIM>>& domainSampler;
    const float& normalOffsetForAbsorbingBoundary;
    std::vector<EvaluationPoint<T, DIM>>& evalPts;
    NearestNeighborFinder nearestNeighborFinder;
    SplatContributionCallback<T, DIM> splatContributionCallback;
    std::vector<SamplePoint<T, DIM>> absorbingBoundarySamplePts;
    std::vector<SamplePoint<T, DIM>> absorbingBoundaryNormalAlignedSamplePts;
    std::vector<SamplePoint<T, DIM>> reflectingBoundarySamplePts;
    std::vector<SamplePoint<T, DIM>> reflectingBoundaryNormalAlignedSamplePts;
    std::vector<SamplePoint<T, DIM>> domainSamplePts;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - improve Poisson kernel estimation on Dirichlet boundary (currently using finite differencing on
//   Greens function and self-normalization to reduce bias, but self-normalization only works with
//   interior Poisson problems with pure Dirichlet or mixed Dirichet/Neumann boundary conditions)
// - splat gradient estimates (challenge is again with Poisson kernel on Dirichlet boundary, rather
//   than Greens function for reflecting Neumann/Robin boundaries and source term)

template <typename T, size_t DIM>
inline EvaluationPoint<T, DIM>::EvaluationPoint():
pt(Vector<DIM>::Zero()),
normal(Vector<DIM>::Zero()),
type(SampleType::InDomain),
distToAbsorbingBoundary(0.0f),
distToReflectingBoundary(0.0f)
{
    mutex = std::make_shared<tbb::spin_mutex>();
    reset();
}

template <typename T, size_t DIM>
inline EvaluationPoint<T, DIM>::EvaluationPoint(const Vector<DIM>& pt_,
                                                const Vector<DIM>& normal_,
                                                SampleType type_,
                                                float distToAbsorbingBoundary_,
                                                float distToReflectingBoundary_):
                                                pt(pt_), normal(normal_), type(type_),
                                                distToAbsorbingBoundary(distToAbsorbingBoundary_),
                                                distToReflectingBoundary(distToReflectingBoundary_)
{
    mutex = std::make_shared<tbb::spin_mutex>();
    reset();
}

template <typename T, size_t DIM>
T EvaluationPoint<T, DIM>::getEstimatedSolution(int nAbsorbingBoundarySamples,
                                                int nAbsorbingBoundaryNormalAlignedSamples,
                                                int nReflectingBoundarySamples,
                                                int nReflectingBoundaryNormalAlignedSamples,
                                                int nSourceSamples) const
{
    if (type == SampleType::OnAbsorbingBoundary) {
        return totalAbsorbingBoundaryContribution;
    }

    T solution(0.0f);
    if (nAbsorbingBoundarySamples > 0) {
        if (totalPoissonKernelContribution > 0.0f) {
            solution += totalAbsorbingBoundaryContribution/totalPoissonKernelContribution;

        } else {
            solution += totalAbsorbingBoundaryContribution/nAbsorbingBoundarySamples;
        }
    }

    if (nAbsorbingBoundaryNormalAlignedSamples > 0) {
        solution += totalAbsorbingBoundaryNormalAlignedContribution/nAbsorbingBoundaryNormalAlignedSamples;
    }

    if (nReflectingBoundarySamples > 0) {
        solution += totalReflectingBoundaryContribution/nReflectingBoundarySamples;
    }

    if (nReflectingBoundaryNormalAlignedSamples > 0) {
        solution += totalReflectingBoundaryNormalAlignedContribution/nReflectingBoundaryNormalAlignedSamples;
    }

    if (nSourceSamples > 0) {
        solution += totalSourceContribution/nSourceSamples;
    }

    return solution;
}

template <typename T, size_t DIM>
void EvaluationPoint<T, DIM>::reset()
{
    totalPoissonKernelContribution = 0.0f;
    totalAbsorbingBoundaryContribution = T(0.0f);
    totalAbsorbingBoundaryNormalAlignedContribution = T(0.0f);
    totalReflectingBoundaryContribution = T(0.0f);
    totalReflectingBoundaryNormalAlignedContribution = T(0.0f);
    totalSourceContribution = T(0.0f);
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
void splatContribution(const WalkState<T, DIM>& state,
                       const SamplePoint<T, DIM>& samplePt,
                       const GeometricQueries<DIM>& queries,
                       const NearestNeighborFinder& nearestNeighborFinder,
                       const PDE<T, DIM>& pde,
                       const float& normalOffsetForAbsorbingBoundary,
                       const float& radiusClamp,
                       const float& kernelRegularization,
                       std::vector<EvaluationPoint<T, DIM>>& evalPts)
{
    // perform nearest neighbor queries to determine evaluation points that lie
    // within the sphere centered at the current random walk position
    std::vector<size_t> nnIndices;
    size_t nnCount = nearestNeighborFinder.radiusSearch(state.currentPt, state.greensFn->R, nnIndices);
    bool hasRobinCoeffs = pde.robin ? true : false;
    bool useSelfNormalization = queries.domainIsWatertight && pde.absorptionCoeff == 0.0f;
    if (pde.robin && useSelfNormalization) useSelfNormalization = pde.areRobinConditionsPureNeumann;

    for (size_t i = 0; i < nnCount; i++) {
        EvaluationPoint<T, DIM>& evalPt = evalPts[nnIndices[i]];

        // ignore evaluation points on the absorbing boundary
        if (evalPt.type == SampleType::OnAbsorbingBoundary) continue;

        // ensure evaluation points are visible from current random walk position
        if (!queries.intersectsWithReflectingBoundary(
                state.currentPt, evalPt.pt, state.currentNormal, evalPt.normal,
                state.onReflectingBoundary, evalPt.type == SampleType::OnReflectingBoundary)) {
            // compute greens function weighting
            float samplePtAlpha = state.onReflectingBoundary ? 2.0f : 1.0f;
            state.greensFn->rClamp = radiusClamp;
            float G = state.greensFn->evaluate(state.currentPt, evalPt.pt);
            if (kernelRegularization > 0.0f) {
                float r = std::max(radiusClamp, (state.currentPt - evalPt.pt).norm());
                r /= kernelRegularization;
                G *= KernelRegularization<DIM>::regularizationForGreensFn(r);
            }

            float weight = samplePtAlpha*state.throughput*G/samplePt.pdf;

            // add sample contribution to evaluation point
            tbb::spin_mutex::scoped_lock lock(*evalPt.mutex);
            if (samplePt.type == SampleType::OnAbsorbingBoundary) {
                weight /= normalOffsetForAbsorbingBoundary;
                if (samplePt.estimateBoundaryNormalAligned) {
                    evalPt.totalAbsorbingBoundaryNormalAlignedContribution += weight*samplePt.contribution;

                } else {
                    evalPt.totalAbsorbingBoundaryContribution += weight*samplePt.contribution;
                    if (useSelfNormalization) evalPt.totalPoissonKernelContribution += weight;
                }

            } else if (samplePt.type == SampleType::OnReflectingBoundary) {
                if (samplePt.estimateBoundaryNormalAligned) {
                    evalPt.totalReflectingBoundaryNormalAlignedContribution += weight*samplePt.contribution;

                } else {
                    evalPt.totalReflectingBoundaryContribution += weight*samplePt.contribution;
                }

            } else if (samplePt.type == SampleType::InDomain) {
                evalPt.totalSourceContribution += weight*samplePt.contribution;
            }
        }
    }
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::ReverseWalkOnStarsSolver(const PDE<T, DIM>& pde_,
                                                                                         const GeometricQueries<DIM>& queries_,
                                                                                         const std::unique_ptr<BoundarySampler<T, DIM>>& absorbingBoundarySampler_,
                                                                                         const std::unique_ptr<BoundarySampler<T, DIM>>& reflectingBoundarySampler_,
                                                                                         const std::unique_ptr<DomainSampler<T, DIM>>& domainSampler_,
                                                                                         const float& normalOffsetForAbsorbingBoundary_,
                                                                                         const float& radiusClamp_,
                                                                                         const float& kernelRegularization_,
                                                                                         std::vector<EvaluationPoint<T, DIM>>& evalPts_):
                                                                                         pde(pde_), queries(queries_),
                                                                                         absorbingBoundarySampler(absorbingBoundarySampler_),
                                                                                         reflectingBoundarySampler(reflectingBoundarySampler_),
                                                                                         domainSampler(domainSampler_),
                                                                                         normalOffsetForAbsorbingBoundary(normalOffsetForAbsorbingBoundary_),
                                                                                         evalPts(evalPts_)
{
    // build nearest neighbor acceleration structure
    modifiedEvaluationPoints();

    // bind splat contribution callback
    splatContributionCallback = std::bind(&splatContribution<T, DIM, NearestNeighborFinder>,
                                          std::placeholders::_1, std::placeholders::_2,
                                          std::cref(queries), std::cref(nearestNeighborFinder),
                                          std::cref(pde), std::cref(normalOffsetForAbsorbingBoundary_),
                                          std::cref(radiusClamp_), std::cref(kernelRegularization_),
                                          std::ref(evalPts));
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline void ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::modifiedEvaluationPoints()
{
    // initialize nearest neigbhbor finder with positions of evaluation points
    std::vector<Vector<DIM>> positions;
    for (auto& evalPt: evalPts) {
        positions.push_back(evalPt.pt);
    }

    nearestNeighborFinder.buildAccelerationStructure(positions);
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline void ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::generateSamples(int absorbingBoundarySampleCount,
                                                                                     int reflectingBoundarySampleCount,
                                                                                     int domainSampleCount,
                                                                                     bool solveDoubleSided)
{
    absorbingBoundarySampler->generateSamples(absorbingBoundarySampler->getSampleCount(absorbingBoundarySampleCount, false),
                                              SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                              absorbingBoundarySamplePts, false);
    if (solveDoubleSided) {
        absorbingBoundarySampler->generateSamples(absorbingBoundarySampler->getSampleCount(absorbingBoundarySampleCount, true),
                                                  SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                                  absorbingBoundaryNormalAlignedSamplePts, true);
    }

    reflectingBoundarySampler->generateSamples(reflectingBoundarySampler->getSampleCount(reflectingBoundarySampleCount, false),
                                               SampleType::OnReflectingBoundary, 0.0f,
                                               reflectingBoundarySamplePts, false);
    if (solveDoubleSided) {
        reflectingBoundarySampler->generateSamples(reflectingBoundarySampler->getSampleCount(reflectingBoundarySampleCount, true),
                                                   SampleType::OnReflectingBoundary, 0.0f,
                                                   reflectingBoundaryNormalAlignedSamplePts, true);
    }

    domainSampler->generateSamples(domainSampleCount, domainSamplePts);
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline void ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::solve(const WalkSettings& walkSettings,
                                                                           bool runSingleThreaded,
                                                                           std::function<void(int,int)> reportProgress)
{
    ReverseWalkOnStars<T, DIM> reverseWalkOnStars(queries, splatContributionCallback);
    reverseWalkOnStars.solve(pde, walkSettings, absorbingBoundarySamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, absorbingBoundaryNormalAlignedSamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, reflectingBoundarySamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, reflectingBoundaryNormalAlignedSamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, domainSamplePts, runSingleThreaded, reportProgress);

    // assign solution values to evaluation points on the absorbing boundary
    for (auto& evalPt: evalPts) {
        if (evalPt.type == SampleType::OnAbsorbingBoundary) {
            evalPt.totalAbsorbingBoundaryContribution = !walkSettings.ignoreAbsorbingBoundaryContribution ?
                                                        pde.dirichlet(evalPt.pt, false) : T(0.0f);
        }
    }
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline const std::vector<SamplePoint<T, DIM>>& ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::getAbsorbingBoundarySamplePts(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? absorbingBoundaryNormalAlignedSamplePts : absorbingBoundarySamplePts;
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline const std::vector<SamplePoint<T, DIM>>& ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::getReflectingBoundarySamplePts(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? reflectingBoundaryNormalAlignedSamplePts : reflectingBoundarySamplePts;
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline const std::vector<SamplePoint<T, DIM>>& ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::getDomainSamplePts() const
{
    return domainSamplePts;
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline int ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::getAbsorbingBoundarySampleCount(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? (int)absorbingBoundaryNormalAlignedSamplePts.size() :
                                         (int)absorbingBoundarySamplePts.size();
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline int ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::getReflectingBoundarySampleCount(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? (int)reflectingBoundaryNormalAlignedSamplePts.size() :
                                         (int)reflectingBoundarySamplePts.size();
}

template <typename T, size_t DIM, typename NearestNeighborFinder>
inline int ReverseWalkOnStarsSolver<T, DIM, NearestNeighborFinder>::getDomainSampleCount() const
{
    return (int)domainSamplePts.size();
}

} // rws

} // zombie

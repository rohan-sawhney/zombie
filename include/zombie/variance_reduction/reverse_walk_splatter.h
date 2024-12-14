// This file implements a "reverse walk" splatting technique for reducing variance
// of the walk-on-spheres and walk-on-stars estimators at a set of user-selected
// evaluation points.
//
// Resources:
// - A Bidirectional Formulation for Walk on Spheres [2022]
// - Walkin’ Robin: Walk on Stars with Robin Boundary Conditions [2024]

#pragma once

#include <zombie/point_estimation/reverse_walk_on_stars.h>
#include "tbb/mutex.h"

namespace zombie {

namespace rws {

template <typename T, size_t DIM>
struct EvaluationPoint {
    // constructor
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

    // resets statistics
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
    std::unique_ptr<tbb::mutex> mutex;
};

template <typename T, size_t DIM, typename NearestNeighborFinder>
void splatContribution(const WalkState<T, DIM>& state,
                       const SampleContribution<T>& sampleContribution,
                       const GeometricQueries<DIM>& queries,
                       const NearestNeighborFinder& nearestNeighborFinder,
                       const PDE<T, DIM>& pde,
                       float normalOffsetForAbsorbingBoundary,
                       float radiusClamp, float kernelRegularization,
                       std::vector<EvaluationPoint<T, DIM>>& evalPts);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - improve Poisson kernel estimation on Dirichlet boundary (currently using finite differencing on
//   Greens function and self-normalization to reduce bias, but self-normalization only works with
//   interior Poisson problems with pure Dirichlet or mixed Dirichet/Neumann boundary conditions)
// - splat gradient estimates (challenge is again with Poisson kernel on Dirichlet boundary, rather
//   than Greens function for reflecting Neumann/Robin boundaries and source term)

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
    mutex = std::make_unique<tbb::mutex>();
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
                       const SampleContribution<T>& sampleContribution,
                       const GeometricQueries<DIM>& queries,
                       const NearestNeighborFinder& nearestNeighborFinder,
                       const PDE<T, DIM>& pde,
                       float normalOffsetForAbsorbingBoundary,
                       float radiusClamp, float kernelRegularization,
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

            float weight = samplePtAlpha*state.throughput*G/sampleContribution.pdf;

            // add sample contribution to evaluation point
            tbb::mutex::scoped_lock lock(*evalPt.mutex);
            if (sampleContribution.type == SampleType::OnAbsorbingBoundary) {
                weight /= normalOffsetForAbsorbingBoundary;
                if (sampleContribution.boundaryNormalAligned) {
                    evalPt.totalAbsorbingBoundaryNormalAlignedContribution += weight*sampleContribution.contribution;

                } else {
                    evalPt.totalAbsorbingBoundaryContribution += weight*sampleContribution.contribution;
                    if (useSelfNormalization) evalPt.totalPoissonKernelContribution += weight;
                }

            } else if (sampleContribution.type == SampleType::OnReflectingBoundary) {
                if (sampleContribution.boundaryNormalAligned) {
                    evalPt.totalReflectingBoundaryNormalAlignedContribution += weight*sampleContribution.contribution;

                } else {
                    evalPt.totalReflectingBoundaryContribution += weight*sampleContribution.contribution;
                }

            } else if (sampleContribution.type == SampleType::InDomain) {
                evalPt.totalSourceContribution += weight*sampleContribution.contribution;
            }
        }
    }
}

} // rws

} // zombie

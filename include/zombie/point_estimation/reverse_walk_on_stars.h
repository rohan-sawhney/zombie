// Like the Monte Carlo walk-on-stars algorithm, this file implements an algorithm
// for solving Poisson and screened Poisson equations with Dirichlet, Neumann and/or
// Robin boundary conditions. Unlike walk-on-stars which independently computes the
// solution at each input point, this algorithm uses a reverse walk strategy to splat
// contributions from the source term and various boundary conditions to multiple
// evaluation points of the user's choice, with walks starting at the input sample points
// inside the domain or on the boundary. This algorithm is useful when the user wants to
// solve PDEs with localized sources or boundary conditions. Each call to solve(...)
// improves the estimate of the PDE solution, enabling progressive evaluation.
//
// Resources:
// - A Bidirectional Formulation for Walk on Spheres [2022]
// - Walkin’ Robin: Walk on Stars with Robin Boundary Conditions [2024]

#pragma once

#include <zombie/point_estimation/common.h>
#include "oneapi/tbb/parallel_for.h"

namespace zombie {

template <typename T, size_t DIM>
using SplatContributionCallback = std::function<void(const WalkState<T, DIM>&,
                                                     const SamplePoint<T, DIM>&)>;

template <typename T, size_t DIM>
class ReverseWalkOnStars {
public:
    // constructor
    ReverseWalkOnStars(const GeometricQueries<DIM>& queries_,
                       SplatContributionCallback<T, DIM> splatContribution_);

    // solves the given PDE by splatting contributions (dirichlet/neumann/robin/source)
    // from a walk starting at the input sample point
    void solve(const PDE<T, DIM>& pde,
               const WalkSettings& walkSettings,
               SamplePoint<T, DIM>& samplePt) const;

    // solves the given PDE by splatting contributions (dirichlet/neumann/robin/source)
    // from walks starting at the input sample points
    void solve(const PDE<T, DIM>& pde,
               const WalkSettings& walkSettings,
               std::vector<SamplePoint<T, DIM>>& samplePts,
               bool runSingleThreaded=false,
               std::function<void(int, int)> reportProgress={}) const;

protected:
    // computes the throughput of a single walk step
    float computeWalkStepThroughput(const PDE<T, DIM>& pde,
                                    const WalkSettings& walkSettings,
                                    const WalkState<T, DIM>& state) const;

    // performs a single reflecting random walk starting at the input point
    WalkCompletionCode walk(const PDE<T, DIM>& pde,
                            const WalkSettings& walkSettings,
                            const SamplePoint<T, DIM>& samplePt,
                            float distToAbsorbingBoundary,
                            pcg32& sampler, WalkState<T, DIM>& state) const;

    // members
    const GeometricQueries<DIM>& queries;
    SplatContributionCallback<T, DIM> splatContribution;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline ReverseWalkOnStars<T, DIM>::ReverseWalkOnStars(const GeometricQueries<DIM>& queries_,
                                                      SplatContributionCallback<T, DIM> splatContribution_):
                                                      queries(queries_), splatContribution(splatContribution_)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void ReverseWalkOnStars<T, DIM>::solve(const PDE<T, DIM>& pde,
                                              const WalkSettings& walkSettings,
                                              SamplePoint<T, DIM>& samplePt) const
{
    // set sample contribution
    bool didSetContribution = true;
    if (samplePt.type == SampleType::InDomain &&
        !walkSettings.ignoreSourceContribution) {
        samplePt.contribution = pde.source(samplePt.pt);

    } else if (samplePt.type == SampleType::OnAbsorbingBoundary &&
               !walkSettings.ignoreAbsorbingBoundaryContribution) {
        // project the walk position to the absorbing boundary and grab the known boundary value
        // NOTE: boundary value should ideally be grabbed before offsetting sample along normal
        float signedDistance = queries.computeDistToAbsorbingBoundary(samplePt.pt, true);
        Vector<DIM> pt = samplePt.pt - signedDistance*samplePt.normal;

        bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                signedDistance > 0.0f;
        samplePt.contribution = pde.dirichlet(pt, returnBoundaryNormalAlignedValue);

    } else if (samplePt.type == SampleType::OnReflectingBoundary &&
               !walkSettings.ignoreReflectingBoundaryContribution) {
        bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                samplePt.estimateBoundaryNormalAligned;
        samplePt.contribution = pde.robin(samplePt.pt, returnBoundaryNormalAlignedValue);

    } else {
        didSetContribution = false;
    }

    if (didSetContribution) {
        // set the direction of approach of the walk for double-sided boundary conditions
        Vector<DIM> prevDirection = samplePt.normal;
        float prevDistance = std::numeric_limits<float>::max();
        bool onReflectingBoundary = samplePt.type == SampleType::OnReflectingBoundary;

        if (walkSettings.solveDoubleSided && onReflectingBoundary) {
            if (samplePt.estimateBoundaryNormalAligned) {
                prevDirection *= -1.0f;
            }
        }

        // initialize the walk state
        WalkState<T, DIM> state(samplePt.pt, samplePt.normal, prevDirection, prevDistance,
                                1.0f, 0, onReflectingBoundary);

        // initialize the greens function
        if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
            state.greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);

        } else {
            state.greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
        }

        // perform walk
        WalkCompletionCode code = walk(pde, walkSettings, samplePt,
                                       samplePt.distToAbsorbingBoundary,
                                       samplePt.sampler, state);
    }
}

template <typename T, size_t DIM>
inline void ReverseWalkOnStars<T, DIM>::solve(const PDE<T, DIM>& pde,
                                              const WalkSettings& walkSettings,
                                              std::vector<SamplePoint<T, DIM>>& samplePts,
                                              bool runSingleThreaded,
                                              std::function<void(int, int)> reportProgress) const
{
    int nPoints = (int)samplePts.size();
    if (runSingleThreaded || walkSettings.printLogs) {
        for (int i = 0; i < nPoints; i++) {
            solve(pde, walkSettings, samplePts[i]);
            if (reportProgress) reportProgress(1, 0);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                solve(pde, walkSettings, samplePts[i]);
            }

            if (reportProgress) {
                int tbb_thread_id = tbb::this_task_arena::current_thread_index();
                reportProgress(range.end() - range.begin(), tbb_thread_id);
            }
        };

        tbb::blocked_range<int> range(0, nPoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline float ReverseWalkOnStars<T, DIM>::computeWalkStepThroughput(const PDE<T, DIM>& pde,
                                                                   const WalkSettings& walkSettings,
                                                                   const WalkState<T, DIM>& state) const
{
    if (state.onReflectingBoundary && state.prevDistance > std::numeric_limits<float>::epsilon()) {
        float robinCoeff = 0.0f;
        Vector<DIM> normal = state.currentNormal;

        if (!pde.areRobinConditionsPureNeumann) {
            bool flipNormalOrientation = false;
            if (walkSettings.solveDoubleSided) {
                flipNormalOrientation = state.prevDirection.dot(state.currentNormal) < 0.0f;
                normal *= flipNormalOrientation ? -1.0f : 1.0f;
            }

            bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided && flipNormalOrientation;
            robinCoeff = pde.robinCoeff(state.currentPt, returnBoundaryNormalAlignedValue);
        }

        float reflectance = state.greensFn->reflectance(state.prevDistance, state.prevDirection,
                                                        normal, robinCoeff);
        return std::clamp(reflectance, 0.0f, 1.0f);
    }

    return state.greensFn->directionSampledPoissonKernel(state.currentPt);
}

template <typename T, size_t DIM>
inline WalkCompletionCode ReverseWalkOnStars<T, DIM>::walk(const PDE<T, DIM>& pde,
                                                           const WalkSettings& walkSettings,
                                                           const SamplePoint<T, DIM>& samplePt,
                                                           float distToAbsorbingBoundary,
                                                           pcg32& sampler, WalkState<T, DIM>& state) const
{
    // recursively perform a random walk till it reaches the absorbing boundary
    while (distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary) {
        // for problems with double-sided boundary conditions, flip the current
        // normal orientation if the geometry is front-facing
        bool flipNormalOrientation = false;
        if (walkSettings.solveDoubleSided && state.onReflectingBoundary) {
            if (state.prevDistance > 0.0f && state.prevDirection.dot(state.currentNormal) < 0.0f) {
                state.currentNormal *= -1.0f;
                flipNormalOrientation = true;
            }
        }

        // compute the star radius
        float starRadius;
        if (walkSettings.stepsBeforeUsingMaximalSpheres <= state.walkLength) {
            starRadius = distToAbsorbingBoundary;

        } else if (state.onReflectingBoundary && pde.hasNonZeroRobinCoeff(state.currentPt)) {
            // NOTE: reflectance, and hence sphere radius, is zero exactly on the boundary,
            // therefore we use a small epsilon for the sphere radius to ensure the walk continues
            starRadius = walkSettings.epsilonShellForReflectingBoundary;

        } else {
            // NOTE: using distToAbsorbingBoundary as the maximum radius for the star radius
            // query can result in a smaller than maximal star-shaped region: should ideally
            // use the distance to the closest visible point on the absorbing boundary
            starRadius = queries.computeStarRadiusForReflectingBoundary(
                state.currentPt, walkSettings.epsilonShellForReflectingBoundary, distToAbsorbingBoundary,
                walkSettings.silhouettePrecision, flipNormalOrientation);

            // shrink the radius slightly for numerical robustness---using a conservative
            // distance does not impact correctness
            if (walkSettings.epsilonShellForReflectingBoundary <= distToAbsorbingBoundary) {
                starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius,
                                        walkSettings.epsilonShellForReflectingBoundary);
            }
        }

        // update the ball center and radius
        state.greensFn->updateBall(state.currentPt, starRadius);

        // splat sample contribution within the current star-shaped region
        splatContribution(state, samplePt);

        // sample a direction uniformly
        Vector<DIM> direction = SphereSampler<DIM>::sampleUnitSphereUniform(sampler);

        // perform hemispherical sampling if on the reflecting boundary, which cancels
        // the alpha term in our integral expression
        if (state.onReflectingBoundary && state.currentNormal.dot(direction) > 0.0f) {
            direction *= -1.0f;
        }

        // check if there is an intersection with the reflecting boundary along the ray:
        // currentPt + starRadius * direction
        IntersectionPoint<DIM> intersectionPt;
        bool intersectedReflectingBoundary = queries.intersectReflectingBoundary(
            state.currentPt, state.currentNormal, direction, starRadius,
            state.onReflectingBoundary, intersectionPt);

        // check if there is no intersection with the reflecting boundary
        if (!intersectedReflectingBoundary) {
            // apply small offset to the current pt for numerical robustness if it on
            // the reflecting boundary---the same offset is applied during ray intersections
            Vector<DIM> currentPt = state.onReflectingBoundary ?
                                    queries.offsetPointAlongDirection(state.currentPt, -state.currentNormal) :
                                    state.currentPt;

            // set intersectionPt to a point on the spherical arc of the ball
            intersectionPt.pt = currentPt + starRadius*direction;
            intersectionPt.dist = starRadius;
        }

        // update walk position
        state.prevDistance = intersectionPt.dist;
        state.prevDirection = direction;
        state.currentPt = intersectionPt.pt;
        state.currentNormal = intersectionPt.normal; // NOTE: stale unless intersectedReflectingBoundary is true
        state.onReflectingBoundary = intersectedReflectingBoundary;

        // check if the current pt lies outside the domain; for interior problems,
        // this tests for walks that escape due to numerical error
        if (!state.onReflectingBoundary && queries.outsideBoundingDomain(state.currentPt)) {
            if (walkSettings.printLogs) {
                std::cout << "Walk escaped domain!" << std::endl;
            }

            return WalkCompletionCode::EscapedDomain;
        }

        // update the walk throughput and use russian roulette to decide whether to terminate the walk
        state.throughput *= computeWalkStepThroughput(pde, walkSettings, state);
        if (state.throughput < walkSettings.russianRouletteThreshold) {
            float survivalProb = state.throughput/walkSettings.russianRouletteThreshold;
            if (survivalProb < sampler.nextFloat()) {
                state.throughput = 0.0f;
                return WalkCompletionCode::TerminatedWithRussianRoulette;
            }

            state.throughput = walkSettings.russianRouletteThreshold;
        }

        // update the walk length and break if the max walk length is exceeded
        state.walkLength++;
        if (state.walkLength > walkSettings.maxWalkLength) {
            if (walkSettings.printLogs) {
                std::cout << "Maximum walk length exceeded!" << std::endl;
            }

            return WalkCompletionCode::ExceededMaxWalkLength;
        }

        // check whether to start applying Tikhonov regularization
        if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == state.walkLength) {
            state.greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);
        }

        // compute the distance to the absorbing boundary
        distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);
    }

    return WalkCompletionCode::ReachedAbsorbingBoundary;
}

} // zombie

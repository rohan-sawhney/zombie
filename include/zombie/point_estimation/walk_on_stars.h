// This file implements the Monte Carlo walk-on-stars algorithm for solving
// Poisson and screened Poisson equations with Dirichlet, Neumann and/or Robin
// boundary conditions. This algorithm is a strict generalization of walk-on-spheres,
// and hence can also be used to solve PDEs with pure Dirichlet boundary conditions.
// The estimated PDE solution, gradient, and other statistics can be queried
// from the SampleStatistics struct, which is stored in the SamplePoint struct.
// Each call to solve(...) improves the estimate of the PDE solution at the
// user-selected set of input points, enabling progressive evaluation.
//
// Resources:
// - Walk on Stars: Grid-Free Monte Carlo for PDEs with Neumann Conditions [2023]
// - Walkin’ Robin: Walk on Stars with Robin Boundary Conditions [2024]

#pragma once

#include <zombie/point_estimation/common.h>
#include <queue>
#include "oneapi/tbb/parallel_for.h"

namespace zombie {

template <typename T, size_t DIM>
class WalkOnStars {
public:
    // constructors
    WalkOnStars(const GeometricQueries<DIM>& queries_);
    WalkOnStars(const GeometricQueries<DIM>& queries_,
                std::function<void(const WalkState<T, DIM>&)> walkStateCallback_,
                std::function<T(WalkCompletionCode, const WalkState<T, DIM>&)> terminalContributionCallback_);

    // solves the given PDE at the input point; NOTE: assumes the point does not
    // lie on the boundary when estimating the gradient
    void solve(const PDE<T, DIM>& pde,
               const WalkSettings& walkSettings,
               int nWalks, SamplePoint<T, DIM>& samplePt) const;

    // solves the given PDE at the input points (in parallel by default);
    // NOTE: assumes points do not lie on the boundary when estimating gradients
    void solve(const PDE<T, DIM>& pde,
               const WalkSettings& walkSettings,
               const std::vector<int>& nWalks,
               std::vector<SamplePoint<T, DIM>>& samplePts,
               bool runSingleThreaded=false,
               std::function<void(int, int)> reportProgress={}) const;

protected:
    // computes the contribution from the reflecting boundary at a particular point in the walk
    void computeReflectingBoundaryContribution(const PDE<T, DIM>& pde,
                                               const WalkSettings& walkSettings,
                                               float starRadius, bool flipNormalOrientation,
                                               pcg32& rng, WalkState<T, DIM>& state) const;

    // computes the source contribution at a particular point in the walk
    void computeSourceContribution(const PDE<T, DIM>& pde,
                                   const WalkSettings& walkSettings,
                                   const Vector<DIM>& direction,
                                   float intersectionDist, pcg32& rng,
                                   WalkState<T, DIM>& state) const;

    // computes the throughput of a single walk step
    float computeWalkStepThroughput(const PDE<T, DIM>& pde,
                                    const WalkSettings& walkSettings,
                                    const WalkState<T, DIM>& state) const;

    // applies a weight window to a walk based on the current state
    bool applyWeightWindow(const WalkSettings& walkSettings,
                           pcg32& rng, WalkState<T, DIM>& state,
                           std::queue<WalkState<T, DIM>>& stateQueue) const;

    // performs a single reflecting random walk starting at the input point
    WalkCompletionCode walk(const PDE<T, DIM>& pde,
                            const WalkSettings& walkSettings,
                            float distToAbsorbingBoundary,
                            float firstSphereRadius,
                            bool flipNormalOrientation,
                            pcg32& rng, WalkState<T, DIM>& state,
                            std::queue<WalkState<T, DIM>>& stateQueue) const;

    // returns the terminal contribution from the end of the walk
    T getTerminalContribution(WalkCompletionCode code,
                              const PDE<T, DIM>& pde,
                              const WalkSettings& walkSettings,
                              WalkState<T, DIM>& state) const;

    // estimates only the solution of the given PDE at the input point
    void estimateSolution(const PDE<T, DIM>& pde,
                          const WalkSettings& walkSettings,
                          int nWalks, SamplePoint<T, DIM>& samplePt) const;

    // estimates the solution and gradient of the given PDE at the input point;
    // NOTE: assumes the point does not lie on the boundary; the directional derivative
    // can be accessed through samplePt.statistics.getEstimatedDerivative()
    void estimateSolutionAndGradient(const PDE<T, DIM>& pde,
                                     const WalkSettings& walkSettings,
                                     int nWalks, SamplePoint<T, DIM>& samplePt) const;

    // members
    const GeometricQueries<DIM>& queries;
    std::function<void(const WalkState<T, DIM>&)> walkStateCallback;
    std::function<T(WalkCompletionCode, const WalkState<T, DIM>&)> terminalContributionCallback;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline WalkOnStars<T, DIM>::WalkOnStars(const GeometricQueries<DIM>& queries_):
                                        queries(queries_), walkStateCallback({}),
                                        terminalContributionCallback({})
{
    // do nothing
}

template <typename T, size_t DIM>
inline WalkOnStars<T, DIM>::WalkOnStars(const GeometricQueries<DIM>& queries_,
                                        std::function<void(const WalkState<T, DIM>&)> walkStateCallback_,
                                        std::function<T(WalkCompletionCode, const WalkState<T, DIM>&)> terminalContributionCallback_):
                                        queries(queries_), walkStateCallback(walkStateCallback_),
                                        terminalContributionCallback(terminalContributionCallback_)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void WalkOnStars<T, DIM>::solve(const PDE<T, DIM>& pde,
                                       const WalkSettings& walkSettings,
                                       int nWalks, SamplePoint<T, DIM>& samplePt) const
{
    if (samplePt.estimationQuantity != EstimationQuantity::None) {
        if (samplePt.estimationQuantity == EstimationQuantity::SolutionAndGradient) {
            estimateSolutionAndGradient(pde, walkSettings, nWalks, samplePt);

        } else {
            estimateSolution(pde, walkSettings, nWalks, samplePt);
        }
    }
}

template <typename T, size_t DIM>
inline void WalkOnStars<T, DIM>::solve(const PDE<T, DIM>& pde,
                                       const WalkSettings& walkSettings,
                                       const std::vector<int>& nWalks,
                                       std::vector<SamplePoint<T, DIM>>& samplePts,
                                       bool runSingleThreaded,
                                       std::function<void(int, int)> reportProgress) const
{
    // solve the PDE at each point independently
    int nPoints = (int)samplePts.size();
    if (runSingleThreaded || walkSettings.printLogs) {
        for (int i = 0; i < nPoints; i++) {
            solve(pde, walkSettings, nWalks[i], samplePts[i]);
            if (reportProgress) reportProgress(1, 0);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                solve(pde, walkSettings, nWalks[i], samplePts[i]);
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
inline void WalkOnStars<T, DIM>::computeReflectingBoundaryContribution(const PDE<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       float starRadius, bool flipNormalOrientation,
                                                                       pcg32& rng, WalkState<T, DIM>& state) const
{
    if (queries.hasNonEmptyReflectingBoundary &&
        !walkSettings.ignoreReflectingBoundaryContribution) {
        // compute the non-zero reflecting boundary contribution inside the star-shaped region
        // (defined to be zero outside this region)
        BoundarySample<DIM> boundarySample;
        Vector<DIM> randNumsForBoundarySampling;
        for (int i = 0; i < DIM; i++) randNumsForBoundarySampling[i] = rng.nextFloat();
        if (queries.sampleReflectingBoundary(state.currentPt, starRadius,
                                             randNumsForBoundarySampling,
                                             boundarySample)) {
            Vector<DIM> directionToSample = boundarySample.pt - state.currentPt;
            float distToSample = directionToSample.norm();
            float alpha = state.onReflectingBoundary ? 2.0f : 1.0f;
            bool estimateBoundaryNormalAligned = false;

            if (walkSettings.solveDoubleSided) {
                // normalize the direction to the sample, and flip the sample normal
                // orientation if the geometry is front-facing; NOTE: using a precision
                // parameter since unlike direction sampling, samples can lie on the same
                // halfplane as the current walk location
                directionToSample /= distToSample;
                if (flipNormalOrientation) {
                    estimateBoundaryNormalAligned = true;

                } else if (directionToSample.dot(boundarySample.normal) < -walkSettings.silhouettePrecision) {
                    bool flipBoundarySampleNormal = true;
                    if (alpha > 1.0f) {
                        // on concave boundaries, we want to sample back-facing values
                        // on front-facing geometry below the hemisphere, so we avoid
                        // flipping the normal orientation in this case
                        flipBoundarySampleNormal = directionToSample.dot(state.currentNormal) <
                                                   -walkSettings.silhouettePrecision;
                    }

                    if (flipBoundarySampleNormal) {
                        estimateBoundaryNormalAligned = true;
                    }
                }
            }

            Vector<DIM> boundarySampleNormal = boundarySample.normal*(estimateBoundaryNormalAligned ? -1.0f : 1.0f);
            if (boundarySample.pdf > 0.0f && distToSample < starRadius &&
                !queries.intersectsWithReflectingBoundary(state.currentPt, boundarySample.pt,
                                                          state.currentNormal, boundarySampleNormal,
                                                          state.onReflectingBoundary, true)) {
                float r = std::max(state.greensFn->rClamp, (boundarySample.pt - state.currentPt).norm());
                float G = state.greensFn->evaluate(r);
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        estimateBoundaryNormalAligned;
                T h = pde.robin(boundarySample.pt, boundarySample.normal, returnBoundaryNormalAlignedValue);
                state.totalReflectingBoundaryContribution += state.throughput*alpha*G*h/boundarySample.pdf;
            }
        }
    }
}

template <typename T, size_t DIM>
inline void WalkOnStars<T, DIM>::computeSourceContribution(const PDE<T, DIM>& pde,
                                                           const WalkSettings& walkSettings,
                                                           const Vector<DIM>& direction,
                                                           float intersectionDist, pcg32& rng,
                                                           WalkState<T, DIM>& state) const
{
    if (!walkSettings.ignoreSourceContribution) {
        // compute the source contribution inside the star-shaped region;
        // define the source value to be zero outside this region
        float sourceRadius, sourcePdf;
        Vector<DIM> sourcePt = state.greensFn->sampleVolume(direction, rng, sourceRadius, sourcePdf);
        if (sourceRadius <= intersectionDist) {
            // NOTE: hemispherical sampling causes the alpha term to be cancelled out when
            // currentPt is on the reflecting boundary; in this case, the green's function
            // norm remains unchanged even though our domain is a hemisphere;
            // for double-sided problems in watertight domains, both the current pt
            // and source pt lie either inside or outside the domain by construction
            T sourceContribution = state.greensFn->norm()*pde.source(sourcePt);
            state.totalSourceContribution += state.throughput*sourceContribution;
        }
    }
}

template <typename T, size_t DIM>
inline float WalkOnStars<T, DIM>::computeWalkStepThroughput(const PDE<T, DIM>& pde,
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

            bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                    flipNormalOrientation;
            robinCoeff = pde.robinCoeff(state.currentPt, state.currentNormal,
                                        returnBoundaryNormalAlignedValue);
        }

        float reflectance = state.greensFn->reflectance(state.prevDistance, state.prevDirection,
                                                        normal, robinCoeff);
        float maxReflectance = pde.areRobinCoeffsNonnegative ? 1.0f : 2.0f;
        return std::clamp(reflectance, 0.0f, maxReflectance);
    }

    return state.greensFn->directionSampledPoissonKernel(state.currentPt);
}

template <typename T, size_t DIM>
inline bool WalkOnStars<T, DIM>::applyWeightWindow(const WalkSettings& walkSettings,
                                                   pcg32& rng, WalkState<T, DIM>& state,
                                                   std::queue<WalkState<T, DIM>>& stateQueue) const
{
    if (state.throughput > walkSettings.splittingThreshold) {
        // split the walk
        float throughputLeft = state.throughput - walkSettings.splittingThreshold;
        state.throughput = walkSettings.splittingThreshold;
        WalkState<T, DIM> splitState(state.currentPt, state.currentNormal,
                                     state.prevDirection, state.prevDistance,
                                     state.throughput, 0, state.onReflectingBoundary);

        while (throughputLeft > walkSettings.splittingThreshold) {
            throughputLeft -= walkSettings.splittingThreshold;
            stateQueue.emplace(splitState);
        }

        if (rng.nextFloat() < throughputLeft/walkSettings.splittingThreshold) {
            stateQueue.emplace(splitState);
        }

    } else if (state.throughput < walkSettings.russianRouletteThreshold) {
        // terminate the walk using russian roulette
        float survivalProb = state.throughput/walkSettings.russianRouletteThreshold;
        if (survivalProb < rng.nextFloat()) {
            state.throughput = 0.0f;
            return true;
        }

        state.throughput = walkSettings.russianRouletteThreshold;
    }

    return false;
}

template <typename T, size_t DIM>
inline WalkCompletionCode WalkOnStars<T, DIM>::walk(const PDE<T, DIM>& pde,
                                                    const WalkSettings& walkSettings,
                                                    float distToAbsorbingBoundary,
                                                    float firstSphereRadius,
                                                    bool flipNormalOrientation,
                                                    pcg32& rng, WalkState<T, DIM>& state,
                                                    std::queue<WalkState<T, DIM>>& stateQueue) const
{
    // recursively perform a random walk till it reaches the absorbing boundary
    bool firstStep = true;
    while (distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary) {
        // compute the star radius
        float starRadius;
        if (firstStep && firstSphereRadius > 0.0f) {
            starRadius = firstSphereRadius;

        } else {
            // for problems with double-sided boundary conditions, flip the current
            // normal orientation if the geometry is front-facing
            flipNormalOrientation = false;
            if (walkSettings.solveDoubleSided && state.onReflectingBoundary) {
                if (state.prevDistance > 0.0f && state.prevDirection.dot(state.currentNormal) < 0.0f) {
                    state.currentNormal *= -1.0f;
                    flipNormalOrientation = true;
                }
            }

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
                    state.currentPt, walkSettings.epsilonShellForReflectingBoundary,
                    distToAbsorbingBoundary, walkSettings.silhouettePrecision,
                    flipNormalOrientation);

                // shrink the radius slightly for numerical robustness---using a conservative
                // distance does not impact correctness
                if (queries.hasNonEmptyReflectingBoundary &&
                    walkSettings.epsilonShellForReflectingBoundary <= distToAbsorbingBoundary) {
                    starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius,
                                          walkSettings.epsilonShellForReflectingBoundary);
                }
            }
        }

        // update the ball center and radius
        state.greensFn->updateBall(state.currentPt, starRadius);

        // callback for the current walk state
        if (walkStateCallback) {
            walkStateCallback(state);
        }

        // sample a direction uniformly
        Vector<DIM> direction = SphereSampler<DIM>::sampleUnitSphereUniform(rng);

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

        // compute the contribution from the reflecting boundary
        computeReflectingBoundaryContribution(pde, walkSettings, starRadius,
                                              flipNormalOrientation, rng, state);

        // compute the source contribution
        computeSourceContribution(pde, walkSettings, direction,
                                  intersectionPt.dist, rng, state);

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

        // update the walk throughput and apply a weight window to decide whether
        // to split or terminate the walk
        state.throughput *= computeWalkStepThroughput(pde, walkSettings, state);
        bool terminateWalk = applyWeightWindow(walkSettings, rng, state, stateQueue);
        if (terminateWalk) return WalkCompletionCode::TerminatedWithRussianRoulette;

        // update the walk length and break if the max walk length is exceeded
        state.walkLength++;
        if (state.walkLength > walkSettings.maxWalkLength) {
            if (walkSettings.printLogs && !terminalContributionCallback) {
                std::cout << "Maximum walk length exceeded!" << std::endl;
            }

            return WalkCompletionCode::ExceededMaxWalkLength;
        }

        // check whether to start applying Tikhonov regularization
        if (pde.absorptionCoeff > 0.0f &&
            walkSettings.stepsBeforeApplyingTikhonov == state.walkLength) {
            state.greensFn = std::make_shared<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);
        }

        // compute the distance to the absorbing boundary
        distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);
        firstStep = false;
    }

    return WalkCompletionCode::ReachedAbsorbingBoundary;
}

template <typename T, size_t DIM>
inline T WalkOnStars<T, DIM>::getTerminalContribution(WalkCompletionCode code,
                                                      const PDE<T, DIM>& pde,
                                                      const WalkSettings& walkSettings,
                                                      WalkState<T, DIM>& state) const
{
    if (code == WalkCompletionCode::ReachedAbsorbingBoundary &&
        !walkSettings.ignoreAbsorbingBoundaryContribution) {
        // project the walk position to the absorbing boundary and grab the known boundary value
        float signedDistance;
        queries.projectToAbsorbingBoundary(state.currentPt, state.currentNormal,
                                           signedDistance, walkSettings.solveDoubleSided);
        bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                signedDistance > 0.0f;
        return pde.dirichlet(state.currentPt, returnBoundaryNormalAlignedValue);

    } else if (terminalContributionCallback) {
        // get the user-specified terminal contribution
        return terminalContributionCallback(code, state);
    }

    // return 0 terminal contribution if ignoring absorbing boundary values,
    // or if walk is terminated with russian roulette or exceeds max walk length
    return T(0.0f);
}

template <typename T, size_t DIM>
inline void WalkOnStars<T, DIM>::estimateSolution(const PDE<T, DIM>& pde,
                                                  const WalkSettings& walkSettings,
                                                  int nWalks, SamplePoint<T, DIM>& samplePt) const
{
    // check if there are no previous estimates
    bool hasPrevEstimates = samplePt.statistics.getSolutionEstimateCount() > 0;

    // check if the sample pt is on the absorbing boundary
    if (samplePt.type == SampleType::OnAbsorbingBoundary) {
        if (!hasPrevEstimates) {
            // record the known boundary value
            T totalContribution(0.0f);
            if (!walkSettings.ignoreAbsorbingBoundaryContribution) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                totalContribution = pde.dirichlet(samplePt.pt, returnBoundaryNormalAlignedValue);
            }

            // update statistics and set the first sphere radius to 0
            samplePt.statistics.addSolutionEstimate(totalContribution);
            samplePt.firstSphereRadius = 0.0f;
        }

        // no need to run any random walks
        return;

    } else if (samplePt.distToAbsorbingBoundary <= walkSettings.epsilonShellForAbsorbingBoundary) {
        // run just a single walk since the sample pt is inside the epsilon shell
        nWalks = 1;
    }

    // for problems with double-sided boundary conditions, initialize the direction
    // of approach for walks, and flip the current normal orientation if the geometry
    // is front-facing
    Vector<DIM> currentNormal = samplePt.normal;
    Vector<DIM> prevDirection = samplePt.normal;
    float prevDistance = std::numeric_limits<float>::max();
    bool flipNormalOrientation = false;

    if (walkSettings.solveDoubleSided && samplePt.type == SampleType::OnReflectingBoundary) {
        if (samplePt.estimateBoundaryNormalAligned) {
            currentNormal *= -1.0f;
            prevDirection *= -1.0f;
            flipNormalOrientation = true;
        }
    }

    // precompute the first sphere radius for all walks
    if (!hasPrevEstimates) {
        if (samplePt.distToAbsorbingBoundary <= walkSettings.epsilonShellForAbsorbingBoundary ||
            walkSettings.stepsBeforeUsingMaximalSpheres == 0) {
            samplePt.firstSphereRadius = samplePt.distToAbsorbingBoundary;

        } else if (samplePt.type == SampleType::OnReflectingBoundary &&
                   pde.hasNonZeroRobinCoeff(samplePt.pt)) {
            // NOTE: reflectance, and hence sphere radius, is zero exactly on the boundary,
            // therefore we use a small epsilon for the sphere radius to ensure the walk continues
            samplePt.firstSphereRadius = walkSettings.epsilonShellForReflectingBoundary;

        } else {
            // NOTE: using distToAbsorbingBoundary as the maximum radius for the star radius
            // query can result in a smaller than maximal star-shaped region: should ideally
            // use the distance to the closest visible point on the absorbing boundary
            float starRadius = queries.computeStarRadiusForReflectingBoundary(
                samplePt.pt, walkSettings.epsilonShellForReflectingBoundary,
                samplePt.distToAbsorbingBoundary, walkSettings.silhouettePrecision,
                flipNormalOrientation);

            // shrink the radius slightly for numerical robustness---using a conservative
            // distance does not impact correctness
            if (queries.hasNonEmptyReflectingBoundary &&
                walkSettings.epsilonShellForReflectingBoundary <= samplePt.distToAbsorbingBoundary) {
                starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius,
                                      walkSettings.epsilonShellForReflectingBoundary);
            }

            samplePt.firstSphereRadius = starRadius;
        }
    }

    // perform random walks
    std::queue<WalkState<T, DIM>> stateQueue;
    for (int w = 0; w < nWalks; w++) {
        // initialize the walk state
        WalkState<T, DIM> state(samplePt.pt, currentNormal, prevDirection, prevDistance, 1.0f,
                                0, samplePt.type == SampleType::OnReflectingBoundary);

        // add the state to the queue
        stateQueue.emplace(state);
        int splitsPerformed = -1;
        T totalContribution = T(0.0f);
        bool success = false;

        while (!stateQueue.empty()) {
            state = stateQueue.front();
            stateQueue.pop();
            splitsPerformed++;

            // initialize the greens function
            if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
                state.greensFn = std::make_shared<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);

            } else {
                state.greensFn = std::make_shared<HarmonicGreensFnBall<DIM>>();
            }

            // recompute the distance to the absorbing boundary and the first sphere radius
            // if splits have been performed
            float distToAbsorbingBoundary = samplePt.distToAbsorbingBoundary;
            float firstSphereRadius = samplePt.firstSphereRadius;
            if (splitsPerformed > 0) {
                distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);
                firstSphereRadius = 0.0f;
            }

            // perform the walk with the dequeued state
            WalkCompletionCode code = walk(pde, walkSettings, distToAbsorbingBoundary,
                                           firstSphereRadius, flipNormalOrientation,
                                           samplePt.rng, state, stateQueue);

            if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                code == WalkCompletionCode::ExceededMaxWalkLength) {
                // compute the walk contribution
                T terminalContribution = getTerminalContribution(code, pde, walkSettings, state);
                totalContribution += state.throughput*terminalContribution +
                                     state.totalReflectingBoundaryContribution +
                                     state.totalSourceContribution;

                // record the walk length
                samplePt.statistics.addWalkLength(state.walkLength);
                success = true;
            }
        }

        if (success) {
            // update statistics
            samplePt.statistics.addSolutionEstimate(totalContribution);
            samplePt.statistics.addSplits(splitsPerformed);
        }
    }
}

template <typename T, size_t DIM>
inline void WalkOnStars<T, DIM>::estimateSolutionAndGradient(const PDE<T, DIM>& pde,
                                                             const WalkSettings& walkSettings,
                                                             int nWalks, SamplePoint<T, DIM>& samplePt) const
{
    // reduce nWalks by 2 if using antithetic sampling
    int nAntitheticIters = 1;
    if (walkSettings.useGradientAntitheticVariates) {
        nWalks = std::max(1, nWalks/2);
        nAntitheticIters = 2;
    }

    // use the distance to the boundary as the first sphere radius for all walks;
    // shrink the radius slightly for numerical robustness---using a conservative
    // distance does not impact correctness
    float boundaryDist = std::min(samplePt.distToAbsorbingBoundary,
                                  samplePt.distToReflectingBoundary);
    samplePt.firstSphereRadius = RADIUS_SHRINK_PERCENTAGE*boundaryDist;

    // generate stratified samples
    std::vector<float> stratifiedSamples;
    generateStratifiedSamples<DIM - 1>(stratifiedSamples, 2*nWalks, samplePt.rng);

    // perform random walks
    std::queue<WalkState<T, DIM>> stateQueue;
    for (int w = 0; w < nWalks; w++) {
        // initialize temporary variables for antithetic sampling
        float sourceRadius, sourcePdf, boundaryPdf;
        Vector<DIM> sourcePt, boundaryPt;
        uint64_t seed = 0;

        // compute control variates for the gradient estimate
        T boundaryGradientControlVariate(0.0f);
        T sourceGradientControlVariate(0.0f);
        if (walkSettings.useGradientControlVariates) {
            boundaryGradientControlVariate = samplePt.statistics.getEstimatedSolution();
            sourceGradientControlVariate = samplePt.statistics.getMeanFirstSourceContribution();
        }

        for (int antitheticIter = 0; antitheticIter < nAntitheticIters; antitheticIter++) {
            // initialize the walk state
            WalkState<T, DIM> state(samplePt.pt, Vector<DIM>::Zero(), Vector<DIM>::Zero(),
                                    0.0f, 1.0f, 0, false);

            // initialize the greens function
            if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
                state.greensFn = std::make_shared<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);

            } else {
                state.greensFn = std::make_shared<HarmonicGreensFnBall<DIM>>();
            }

            // update the ball center and radius
            GreensFnBall<DIM> *greensFn = state.greensFn.get();
            greensFn->updateBall(state.currentPt, samplePt.firstSphereRadius);

            // compute the source contribution inside the ball
            T firstSourceContribution(0.0f);
            Vector<DIM> sourceGradientDirection = Vector<DIM>::Zero();
            if (!walkSettings.ignoreSourceContribution) {
                if (antitheticIter == 0) {
                    float *u = &stratifiedSamples[(DIM - 1)*(2*w + 0)];
                    Vector<DIM> sourceDirection = SphereSampler<DIM>::sampleUnitSphereUniform(u);
                    sourcePt = greensFn->sampleVolume(sourceDirection, samplePt.rng,
                                                      sourceRadius, sourcePdf);

                } else {
                    Vector<DIM> sourceDirection = sourcePt - state.currentPt;
                    sourcePt = state.currentPt - sourceDirection;
                }

                float greensFnNorm = greensFn->norm();
                T sourceContribution = greensFnNorm*pde.source(sourcePt);
                state.totalSourceContribution += state.throughput*sourceContribution;
                firstSourceContribution = sourceContribution;
                sourceGradientDirection = greensFn->gradient(sourceRadius, sourcePt)/(sourcePdf*greensFnNorm);
            }

            // sample a point uniformly on the sphere; update the current position
            // of the walk, its throughput and record the boundary gradient direction
            if (antitheticIter == 0) {
                float *u = &stratifiedSamples[(DIM - 1)*(2*w + 1)];
                Vector<DIM> boundaryDirection;
                if (walkSettings.useCosineSamplingForDerivatives) {
                    boundaryDirection = SphereSampler<DIM>::sampleUnitHemisphereCosine(u);
                    if (samplePt.rng.nextFloat() < 0.5f) boundaryDirection[DIM - 1] *= -1.0f;
                    boundaryPdf = 0.5f*SphereSampler<DIM>::pdfSampleUnitHemisphereCosine(std::fabs(boundaryDirection[DIM - 1]));
                    SphereSampler<DIM>::transformCoordinates(samplePt.directionForDerivative, boundaryDirection);

                } else {
                    boundaryDirection = SphereSampler<DIM>::sampleUnitSphereUniform(u);
                    boundaryPdf = SphereSampler<DIM>::pdfSampleSphereUniform(1.0f);
                }

                boundaryPt = greensFn->c + greensFn->R*boundaryDirection;

            } else {
                Vector<DIM> boundaryDirection = boundaryPt - state.currentPt;
                boundaryPt = state.currentPt - boundaryDirection;
            }

            state.prevDistance = greensFn->R;
            state.prevDirection = (boundaryPt - state.currentPt)/greensFn->R;
            state.currentPt = boundaryPt;
            state.throughput *= greensFn->poissonKernel()/boundaryPdf;
            Vector<DIM> boundaryGradientDirection = greensFn->poissonKernelGradient(boundaryPt)/(boundaryPdf*state.throughput);

            // reseed the rng for antithetic sampling
            if (antitheticIter == 0) seed = samplePt.rng.state;
            else samplePt.rng.seed(seed);

            // add the state to the queue
            stateQueue.emplace(state);
            int splitsPerformed = -1;
            T totalContribution = T(0.0f);
            bool success = false;

            while (!stateQueue.empty()) {
                state = stateQueue.front();
                stateQueue.pop();
                splitsPerformed++;

                // initialize the greens function
                if (splitsPerformed > 0) {
                    if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
                        state.greensFn = std::make_shared<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);

                    } else {
                        state.greensFn = std::make_shared<HarmonicGreensFnBall<DIM>>();
                    }
                }

                // compute the distance to the absorbing boundary
                float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);

                // perform the walk with the dequeued state
                WalkCompletionCode code = walk(pde, walkSettings, distToAbsorbingBoundary, 0.0f,
                                               false, samplePt.rng, state, stateQueue);

                if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                    code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                    code == WalkCompletionCode::ExceededMaxWalkLength) {
                    // compute the walk contribution
                    T terminalContribution = getTerminalContribution(code, pde, walkSettings, state);
                    totalContribution += state.throughput*terminalContribution +
                                         state.totalReflectingBoundaryContribution +
                                         state.totalSourceContribution;

                    // record the walk length
                    samplePt.statistics.addWalkLength(state.walkLength);
                    success = true;
                }
            }

            if (success) {
                // compute the gradient contribution
                T boundaryGradientEstimate[DIM];
                T sourceGradientEstimate[DIM];
                T boundaryContribution = totalContribution - firstSourceContribution;
                T directionalDerivative(0.0f);

                for (int i = 0; i < DIM; i++) {
                    boundaryGradientEstimate[i] = (boundaryContribution - boundaryGradientControlVariate)*boundaryGradientDirection[i];
                    sourceGradientEstimate[i] = (firstSourceContribution - sourceGradientControlVariate)*sourceGradientDirection[i];
                    directionalDerivative += boundaryGradientEstimate[i]*samplePt.directionForDerivative[i];
                    directionalDerivative += sourceGradientEstimate[i]*samplePt.directionForDerivative[i];
                }

                // update statistics
                samplePt.statistics.addSolutionEstimate(totalContribution);
                samplePt.statistics.addFirstSourceContribution(firstSourceContribution);
                samplePt.statistics.addGradientEstimate(boundaryGradientEstimate, sourceGradientEstimate);
                samplePt.statistics.addDerivativeContribution(directionalDerivative);
                samplePt.statistics.addSplits(splitsPerformed);
            }
        }
    }
}

} // zombie

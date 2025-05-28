// This file implements the Monte Carlo walk-on-spheres algorithm for solving
// Poisson and screened Poisson equations with Dirichlet boundary conditions.
// The estimated PDE solution, gradient, and other statistics can be queried
// from the SampleStatistics struct, which is stored in the SamplePoint struct.
// Each call to solve(...) improves the estimate of the PDE solution at the
// user-selected set of input points, enabling progressive evaluation.
//
// Resources:
// - Some Continuous Monte Carlo Methods for the Dirichlet Problem [1956]
// - Monte Carlo Geometry Processing: A Grid-Free Approach to PDE-Based Methods on Volumetric Domains [2020]

#pragma once

#include <zombie/point_estimation/common.h>
#include <queue>
#include "oneapi/tbb/parallel_for.h"

namespace zombie {

template <typename T, size_t DIM>
class WalkOnSpheres {
public:
    // constructors
    WalkOnSpheres(const GeometricQueries<DIM>& queries_);
    WalkOnSpheres(const GeometricQueries<DIM>& queries_,
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
    // computes the source contribution at a particular point in the walk
    void computeSourceContribution(const PDE<T, DIM>& pde,
                                   const WalkSettings& walkSettings,
                                   pcg32& sampler, WalkState<T, DIM>& state) const;

    // applies a weight window to a walk based on the current state
    bool applyWeightWindow(const WalkSettings& walkSettings,
                           pcg32& sampler, WalkState<T, DIM>& state,
                           std::queue<WalkState<T, DIM>>& stateQueue) const;

    // performs a single random walk starting at the input point
    WalkCompletionCode walk(const PDE<T, DIM>& pde,
                            const WalkSettings& walkSettings,
                            float distToAbsorbingBoundary,
                            pcg32& sampler, WalkState<T, DIM>& state,
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
inline WalkOnSpheres<T, DIM>::WalkOnSpheres(const GeometricQueries<DIM>& queries_):
                                            queries(queries_), walkStateCallback({}),
                                            terminalContributionCallback({})
{
    // do nothing
}

template <typename T, size_t DIM>
inline WalkOnSpheres<T, DIM>::WalkOnSpheres(const GeometricQueries<DIM>& queries_,
                                            std::function<void(const WalkState<T, DIM>&)> walkStateCallback_,
                                            std::function<T(WalkCompletionCode, const WalkState<T, DIM>&)> terminalContributionCallback_):
                                            queries(queries_), walkStateCallback(walkStateCallback_),
                                            terminalContributionCallback(terminalContributionCallback_)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::solve(const PDE<T, DIM>& pde,
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
inline void WalkOnSpheres<T, DIM>::solve(const PDE<T, DIM>& pde,
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
inline void WalkOnSpheres<T, DIM>::computeSourceContribution(const PDE<T, DIM>& pde,
                                                             const WalkSettings& walkSettings,
                                                             pcg32& sampler, WalkState<T, DIM>& state) const
{
    if (!walkSettings.ignoreSourceContribution) {
        // compute the source contribution inside sphere
        float sourceRadius, sourcePdf;
        Vector<DIM> sourcePt = state.greensFn->sampleVolume(sampler, sourceRadius, sourcePdf);
        T sourceContribution = state.greensFn->norm()*pde.source(sourcePt);
        state.totalSourceContribution += state.throughput*sourceContribution;
    }
}

template <typename T, size_t DIM>
inline bool WalkOnSpheres<T, DIM>::applyWeightWindow(const WalkSettings& walkSettings,
                                                     pcg32& sampler, WalkState<T, DIM>& state,
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

        if (sampler.nextFloat() < throughputLeft/walkSettings.splittingThreshold) {
            stateQueue.emplace(splitState);
        }

    } else if (state.throughput < walkSettings.russianRouletteThreshold) {
        // terminate the walk using russian roulette
        float survivalProb = state.throughput/walkSettings.russianRouletteThreshold;
        if (survivalProb < sampler.nextFloat()) {
            state.throughput = 0.0f;
            return true;
        }

        state.throughput = walkSettings.russianRouletteThreshold;
    }

    return false;
}

template <typename T, size_t DIM>
inline WalkCompletionCode WalkOnSpheres<T, DIM>::walk(const PDE<T, DIM>& pde,
                                                      const WalkSettings& walkSettings,
                                                      float distToAbsorbingBoundary,
                                                      pcg32& sampler, WalkState<T, DIM>& state,
                                                      std::queue<WalkState<T, DIM>>& stateQueue) const
{
    // recursively perform a random walk till it reaches the absorbing boundary
    while (distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary) {
        // update the ball center and radius
        state.greensFn->updateBall(state.currentPt, distToAbsorbingBoundary);

        // callback for the current walk state
        if (walkStateCallback) {
            walkStateCallback(state);
        }

        // compute the source contribution
        computeSourceContribution(pde, walkSettings, sampler, state);

        // sample a direction uniformly
        Vector<DIM> direction = SphereSampler<DIM>::sampleUnitSphereUniform(sampler);

        // update walk position
        state.currentPt += distToAbsorbingBoundary*direction;

        // check if the current pt lies outside the domain; for interior problems,
        // this tests for walks that escape due to numerical error
        if (queries.outsideBoundingDomain(state.currentPt)) {
            if (walkSettings.printLogs) {
                std::cout << "Walk escaped domain!" << std::endl;
            }

            return WalkCompletionCode::EscapedDomain;
        }

        // update the walk throughput and apply a weight window to decide whether to split or terminate the walk
        state.throughput *= state.greensFn->directionSampledPoissonKernel(state.currentPt);
        bool terminateWalk = applyWeightWindow(walkSettings, sampler, state, stateQueue);
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
        if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == state.walkLength) {
            state.greensFn = std::make_shared<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);
        }

        // compute the distance to the absorbing boundary
        distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);
    }

    return WalkCompletionCode::ReachedAbsorbingBoundary;
}

template <typename T, size_t DIM>
inline T WalkOnSpheres<T, DIM>::getTerminalContribution(WalkCompletionCode code,
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
    // or if walk exceeds max walk length or is terminated with russian roulette
    return T(0.0f);
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::estimateSolution(const PDE<T, DIM>& pde,
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

    // precompute the first sphere radius for all walks
    if (!hasPrevEstimates) {
        samplePt.firstSphereRadius = samplePt.distToAbsorbingBoundary;
    }

    // perform random walks
    std::queue<WalkState<T, DIM>> stateQueue;
    for (int w = 0; w < nWalks; w++) {
        // initialize the walk state
        WalkState<T, DIM> state(samplePt.pt, Vector<DIM>::Zero(), Vector<DIM>::Zero(),
                                0.0f, 1.0f, 0, false);

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

            // recompute the distance to the absorbing boundary if a split has been performed
            float distToAbsorbingBoundary = splitsPerformed == 0 ?
                                            samplePt.distToAbsorbingBoundary :
                                            queries.computeDistToAbsorbingBoundary(state.currentPt, false);

            // perform the walk with the dequeued state
            WalkCompletionCode code = walk(pde, walkSettings, distToAbsorbingBoundary,
                                           samplePt.sampler, state, stateQueue);

            if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                code == WalkCompletionCode::ExceededMaxWalkLength) {
                // compute the walk contribution
                T terminalContribution = getTerminalContribution(code, pde, walkSettings, state);
                totalContribution += state.throughput*terminalContribution +
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
inline void WalkOnSpheres<T, DIM>::estimateSolutionAndGradient(const PDE<T, DIM>& pde,
                                                               const WalkSettings& walkSettings,
                                                               int nWalks, SamplePoint<T, DIM>& samplePt) const
{
    // reduce nWalks by 2 if using antithetic sampling
    int nAntitheticIters = 1;
    if (walkSettings.useGradientAntitheticVariates) {
        nWalks = std::max(1, nWalks/2);
        nAntitheticIters = 2;
    }

    // precompute the first sphere radius for all walks
    samplePt.firstSphereRadius = RADIUS_SHRINK_PERCENTAGE*samplePt.distToAbsorbingBoundary;

    // generate stratified samples
    std::vector<float> stratifiedSamples;
    generateStratifiedSamples<DIM - 1>(stratifiedSamples, 2*nWalks, samplePt.sampler);

    // perform random walks
    std::queue<WalkState<T, DIM>> stateQueue;
    for (int w = 0; w < nWalks; w++) {
        // initialize temporary variables for antithetic sampling
        float sourceRadius, sourcePdf, boundaryPdf;
        Vector<DIM> sourcePt, boundaryPt;
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();

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
                    sourcePt = greensFn->sampleVolume(sourceDirection, samplePt.sampler, sourceRadius, sourcePdf);

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
                    if (samplePt.sampler.nextFloat() < 0.5f) boundaryDirection[DIM - 1] *= -1.0f;
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

            state.currentPt = boundaryPt;
            state.throughput *= greensFn->poissonKernel()/boundaryPdf;
            Vector<DIM> boundaryGradientDirection = greensFn->poissonKernelGradient(boundaryPt)/(boundaryPdf*state.throughput);

            // reseed the sampler for antithetic sampling
            samplePt.sampler.seed(seed);

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
                WalkCompletionCode code = walk(pde, walkSettings, distToAbsorbingBoundary,
                                               samplePt.sampler, state, stateQueue);

                if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                    code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                    code == WalkCompletionCode::ExceededMaxWalkLength) {
                    // compute the walk contribution
                    T terminalContribution = getTerminalContribution(code, pde, walkSettings, state);
                    totalContribution += state.throughput*terminalContribution +
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

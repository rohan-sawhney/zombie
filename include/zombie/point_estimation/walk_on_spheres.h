// This file implements the Monte Carlo walk-on-spheres algorithm for solving
// Poisson and screened Poisson equations with Dirichlet boundary conditions.
// The estimated PDE solution, gradient, and other statistics can be queried
// from the SampleStatistics struct, which is stored in the SamplePoint struct.
// Each call to solve(...) improves the estimate of the PDE solution at the
// user-selected set of input points, enabling progressive evaluation.

#pragma once

#include <zombie/point_estimation/common.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

namespace zombie {

template <typename T, size_t DIM>
class WalkOnSpheres {
public:
    // constructor
    WalkOnSpheres(const GeometricQueries<DIM>& queries_,
                  std::function<void(WalkState<T, DIM>&)> getTerminalContribution_={});

    // solves the given PDE at the input point; NOTE: assumes the point does not
    // lie on the boundary when estimating the gradient
    void solve(const PDE<T, DIM>& pde,
               const WalkSettings<T>& walkSettings,
               const SampleEstimationData<DIM>& estimationData,
               SamplePoint<T, DIM>& samplePt) const;

    // solves the given PDE at the input points (in parallel by default); NOTE:
    // assumes points do not lie on the boundary when estimating gradients
    void solve(const PDE<T, DIM>& pde,
               const WalkSettings<T>& walkSettings,
               const std::vector<SampleEstimationData<DIM>>& estimationData,
               std::vector<SamplePoint<T, DIM>>& samplePts,
               bool runSingleThreaded=false,
               std::function<void(int, int)> reportProgress={}) const;

private:
    // computes the source contribution at a particular point in the walk
    void computeSourceContribution(const PDE<T, DIM>& pde,
                                   const WalkSettings<T>& walkSettings,
                                   const std::unique_ptr<GreensFnBall<DIM>>& greensFn,
                                   pcg32& sampler, WalkState<T, DIM>& state) const;

    // performs a single reflecting random walk starting at the input point
    WalkCompletionCode walk(const PDE<T, DIM>& pde,
                            const WalkSettings<T>& walkSettings,
                            float distToAbsorbingBoundary, pcg32& sampler,
                            std::unique_ptr<GreensFnBall<DIM>>& greensFn,
                            WalkState<T, DIM>& state) const;

    // sets the terminal contribution from the end of the walk
    void setTerminalContribution(WalkCompletionCode code, const PDE<T, DIM>& pde,
                                 const WalkSettings<T>& walkSettings,
                                 WalkState<T, DIM>& state) const;

    // estimates only the solution of the given PDE at the input point
    void estimateSolution(const PDE<T, DIM>& pde,
                          const WalkSettings<T>& walkSettings,
                          int nWalks, SamplePoint<T, DIM>& samplePt) const;

    // estimates the solution and gradient of the given PDE at the input point;
    // NOTE: assumes the point does not lie on the boundary; the directional derivative
    // can be accessed through samplePt.statistics->getEstimatedDerivative()
    void estimateSolutionAndGradient(const PDE<T, DIM>& pde,
                                     const WalkSettings<T>& walkSettings,
                                     const Vector<DIM>& directionForDerivative,
                                     int nWalks, SamplePoint<T, DIM>& samplePt) const;

    // members
    const GeometricQueries<DIM>& queries;
    std::function<void(WalkState<T, DIM>&)> getTerminalContribution;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline WalkOnSpheres<T, DIM>::WalkOnSpheres(const GeometricQueries<DIM>& queries_,
                                            std::function<void(WalkState<T, DIM>&)> getTerminalContribution_):
                                            queries(queries_), getTerminalContribution(getTerminalContribution_) {
    // do nothing
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::solve(const PDE<T, DIM>& pde,
                                         const WalkSettings<T>& walkSettings,
                                         const SampleEstimationData<DIM>& estimationData,
                                         SamplePoint<T, DIM>& samplePt) const {
    if (estimationData.estimationQuantity != EstimationQuantity::None) {
        if (estimationData.estimationQuantity == EstimationQuantity::SolutionAndGradient) {
            estimateSolutionAndGradient(pde, walkSettings,
                                        estimationData.directionForDerivative,
                                        estimationData.nWalks, samplePt);

        } else {
            estimateSolution(pde, walkSettings, estimationData.nWalks, samplePt);
        }
    }
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::solve(const PDE<T, DIM>& pde,
                                         const WalkSettings<T>& walkSettings,
                                         const std::vector<SampleEstimationData<DIM>>& estimationData,
                                         std::vector<SamplePoint<T, DIM>>& samplePts, bool runSingleThreaded,
                                         std::function<void(int, int)> reportProgress) const {
    // solve the PDE at each point independently
    int nPoints = (int)samplePts.size();
    if (runSingleThreaded || walkSettings.printLogs) {
        for (int i = 0; i < nPoints; i++) {
            solve(pde, walkSettings, estimationData[i], samplePts[i]);
            if (reportProgress) reportProgress(1, 0);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                solve(pde, walkSettings, estimationData[i], samplePts[i]);
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
                                                             const WalkSettings<T>& walkSettings,
                                                             const std::unique_ptr<GreensFnBall<DIM>>& greensFn,
                                                             pcg32& sampler, WalkState<T, DIM>& state) const {
    if (!walkSettings.ignoreSourceContribution) {
        // compute the source contribution inside sphere
        float sourcePdf;
        Vector<DIM> sourcePt = greensFn->sampleVolume(sampler, sourcePdf);
        T sourceContribution = greensFn->norm()*pde.source(sourcePt);
        state.totalSourceContribution += state.throughput*sourceContribution;
    }
}

template <typename T, size_t DIM>
inline WalkCompletionCode WalkOnSpheres<T, DIM>::walk(const PDE<T, DIM>& pde,
                                                      const WalkSettings<T>& walkSettings,
                                                      float distToAbsorbingBoundary, pcg32& sampler,
                                                      std::unique_ptr<GreensFnBall<DIM>>& greensFn,
                                                      WalkState<T, DIM>& state) const {
    // recursively perform a random walk till it reaches the absorbing boundary
    while (distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary) {
        // update the ball center and radius
        greensFn->updateBall(state.currentPt, distToAbsorbingBoundary);

        // compute the source contribution
        computeSourceContribution(pde, walkSettings, greensFn, sampler, state);

        // sample a direction uniformly
        Vector<DIM> direction = sampleUnitSphereUniform<DIM>(sampler);

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

        // update the walk throughput and use russian roulette to decide whether to terminate the walk
        state.throughput *= greensFn->directionSampledPoissonKernel(state.currentPt);
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
            if (walkSettings.printLogs && !getTerminalContribution) {
                std::cout << "Maximum walk length exceeded!" << std::endl;
            }

            return WalkCompletionCode::ExceededMaxWalkLength;
        }

        // check whether to start applying Tikhonov regularization
        if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == state.walkLength) {
            greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);
        }

        // compute the distance to the absorbing boundary
        distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);
    }

    return WalkCompletionCode::ReachedAbsorbingBoundary;
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::setTerminalContribution(WalkCompletionCode code, const PDE<T, DIM>& pde,
                                                           const WalkSettings<T>& walkSettings,
                                                           WalkState<T, DIM>& state) const {
    if (code == WalkCompletionCode::ReachedAbsorbingBoundary &&
        !walkSettings.ignoreAbsorbingBoundaryContribution) {
        // project the walk position to the absorbing boundary and grab the known boundary value
        float signedDistance;
        queries.projectToAbsorbingBoundary(state.currentPt, state.currentNormal,
                                           signedDistance, walkSettings.solveDoubleSided);
        state.terminalContribution = walkSettings.solveDoubleSided ?
                                     pde.dirichletDoubleSided(state.currentPt, signedDistance > 0.0f) :
                                     pde.dirichlet(state.currentPt);

    } else if (code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution) {
        // get the user-specified terminal contribution
        getTerminalContribution(state);

    } else {
        // terminated with russian roulette or ignoring absorbing boundary values
        state.terminalContribution = walkSettings.initVal;
    }
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::estimateSolution(const PDE<T, DIM>& pde,
                                                    const WalkSettings<T>& walkSettings,
                                                    int nWalks, SamplePoint<T, DIM>& samplePt) const {
    // initialize statistics if there are no previous estimates
    bool hasPrevEstimates = samplePt.statistics != nullptr;
    if (!hasPrevEstimates) {
        samplePt.statistics = std::make_shared<SampleStatistics<T, DIM>>(walkSettings.initVal);
    }

    // check if the sample pt is on the absorbing boundary
    if (samplePt.type == SampleType::OnAbsorbingBoundary) {
        if (!hasPrevEstimates) {
            // record the known boundary value
            T totalContribution = walkSettings.initVal;
            if (!walkSettings.ignoreAbsorbingBoundaryContribution) {
                totalContribution = walkSettings.solveDoubleSided ?
                                    pde.dirichletDoubleSided(samplePt.pt, samplePt.estimateBoundaryNormalAligned) :
                                    pde.dirichlet(samplePt.pt);
            }

            // update statistics and set the first sphere radius to 0
            samplePt.statistics->addSolutionEstimate(totalContribution);
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
    for (int w = 0; w < nWalks; w++) {
        // initialize the greens function
        std::unique_ptr<GreensFnBall<DIM>> greensFn = nullptr;
        if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
            greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);

        } else {
            greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
        }

        // initialize the walk state
        WalkState<T, DIM> state(samplePt.pt, Vector<DIM>::Zero(), Vector<DIM>::Zero(),
                                0.0f, 1.0f, false, 0, walkSettings.initVal);

        // perform walk
        WalkCompletionCode code = walk(pde, walkSettings, samplePt.firstSphereRadius,
                                       samplePt.sampler, greensFn, state);

        if ((code == WalkCompletionCode::ReachedAbsorbingBoundary ||
             code == WalkCompletionCode::TerminatedWithRussianRoulette) ||
            (code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution)) {
            // compute the walk contribution
            setTerminalContribution(code, pde, walkSettings, state);
            T totalContribution = state.throughput*state.terminalContribution +
                                  state.totalSourceContribution;

            // update statistics
            samplePt.statistics->addSolutionEstimate(totalContribution);
            samplePt.statistics->addWalkLength(state.walkLength);
        }
    }
}

template <typename T, size_t DIM>
inline void WalkOnSpheres<T, DIM>::estimateSolutionAndGradient(const PDE<T, DIM>& pde,
                                                               const WalkSettings<T>& walkSettings,
                                                               const Vector<DIM>& directionForDerivative,
                                                               int nWalks, SamplePoint<T, DIM>& samplePt) const {
    // initialize statistics if there are no previous estimates
    bool hasPrevEstimates = samplePt.statistics != nullptr;
    if (!hasPrevEstimates) {
        samplePt.statistics = std::make_shared<SampleStatistics<T, DIM>>(walkSettings.initVal);
    }

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
    for (int w = 0; w < nWalks; w++) {
        // initialize temporary variables for antithetic sampling
        float boundaryPdf, sourcePdf;
        Vector<DIM> boundaryPt, sourcePt;
        unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();

        // compute control variates for the gradient estimate
        T boundaryGradientControlVariate = walkSettings.initVal;
        T sourceGradientControlVariate = walkSettings.initVal;
        if (walkSettings.useGradientControlVariates) {
            boundaryGradientControlVariate = samplePt.statistics->getEstimatedSolution();
            sourceGradientControlVariate = samplePt.statistics->getMeanFirstSourceContribution();
        }

        for (int antitheticIter = 0; antitheticIter < nAntitheticIters; antitheticIter++) {
            // initialize the greens function
            std::unique_ptr<GreensFnBall<DIM>> greensFn = nullptr;
            if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0) {
                greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);

            } else {
                greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
            }

            // initialize the walk state
            WalkState<T, DIM> state(samplePt.pt, Vector<DIM>::Zero(), Vector<DIM>::Zero(),
                                    0.0f, 1.0f, false, 0, walkSettings.initVal);

            // update the ball center and radius
            greensFn->updateBall(state.currentPt, samplePt.firstSphereRadius);

            // compute the source contribution inside the ball
            if (!walkSettings.ignoreSourceContribution) {
                if (antitheticIter == 0) {
                    float *u = &stratifiedSamples[(DIM - 1)*(2*w + 0)];
                    Vector<DIM> sourceDirection = sampleUnitSphereUniform<DIM>(u);
                    sourcePt = greensFn->sampleVolume(sourceDirection, samplePt.sampler, sourcePdf);

                } else {
                    Vector<DIM> sourceDirection = sourcePt - state.currentPt;
                    greensFn->yVol = state.currentPt - sourceDirection;
                    greensFn->r = sourceDirection.norm();
                }

                float greensFnNorm = greensFn->norm();
                T sourceContribution = greensFnNorm*pde.source(greensFn->yVol);
                state.totalSourceContribution += state.throughput*sourceContribution;
                state.firstSourceContribution = sourceContribution;
                state.sourceGradientDirection = greensFn->gradient()/(sourcePdf*greensFnNorm);
            }

            // sample a point uniformly on the sphere; update the current position
            // of the walk, its throughput and record the boundary gradient direction
            if (antitheticIter == 0) {
                float *u = &stratifiedSamples[(DIM - 1)*(2*w + 1)];
                Vector<DIM> boundaryDirection;
                if (walkSettings.useCosineSamplingForDerivatives) {
                    boundaryDirection = sampleUnitHemisphereCosine<DIM>(u);
                    if (samplePt.sampler.nextFloat() < 0.5f) boundaryDirection[DIM - 1] *= -1.0f;
                    boundaryPdf = 0.5f*pdfSampleUnitHemisphereCosine<DIM>(std::fabs(boundaryDirection[DIM - 1]));
                    transformCoordinates<DIM>(directionForDerivative, boundaryDirection);

                } else {
                    boundaryDirection = sampleUnitSphereUniform<DIM>(u);
                    boundaryPdf = pdfSampleSphereUniform<DIM>(1.0f);
                }

                greensFn->ySurf = greensFn->c + greensFn->R*boundaryDirection;
                boundaryPt = greensFn->ySurf;

            } else {
                Vector<DIM> boundaryDirection = boundaryPt - state.currentPt;
                greensFn->ySurf = state.currentPt - boundaryDirection;
            }

            state.currentPt = greensFn->ySurf;
            state.throughput *= greensFn->poissonKernel()/boundaryPdf;
            state.boundaryGradientDirection = greensFn->poissonKernelGradient()/(boundaryPdf*state.throughput);

            // compute the distance to the absorbing boundary
            float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(state.currentPt, false);

            // perform walk
            samplePt.sampler.seed(seed);
            WalkCompletionCode code = walk(pde, walkSettings, distToAbsorbingBoundary,
                                           samplePt.sampler, greensFn, state);

            if ((code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                 code == WalkCompletionCode::TerminatedWithRussianRoulette) ||
                (code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution)) {
                // compute the walk contribution
                setTerminalContribution(code, pde, walkSettings, state);
                T totalContribution = state.throughput*state.terminalContribution +
                                      state.totalSourceContribution;

                // compute the gradient contribution
                T boundaryGradientEstimate[DIM];
                T sourceGradientEstimate[DIM];
                T boundaryContribution = totalContribution - state.firstSourceContribution;
                T directionalDerivative = walkSettings.initVal;

                for (int i = 0; i < DIM; i++) {
                    boundaryGradientEstimate[i] =
                        (boundaryContribution - boundaryGradientControlVariate)*state.boundaryGradientDirection[i];
                    sourceGradientEstimate[i] =
                        (state.firstSourceContribution - sourceGradientControlVariate)*state.sourceGradientDirection[i];

                    directionalDerivative += boundaryGradientEstimate[i]*directionForDerivative[i];
                    directionalDerivative += sourceGradientEstimate[i]*directionForDerivative[i];
                }

                // update statistics
                samplePt.statistics->addSolutionEstimate(totalContribution);
                samplePt.statistics->addFirstSourceContribution(state.firstSourceContribution);
                samplePt.statistics->addGradientEstimate(boundaryGradientEstimate, sourceGradientEstimate);
                samplePt.statistics->addDerivativeContribution(directionalDerivative);
                samplePt.statistics->addWalkLength(state.walkLength);
            }
        }
    }
}

} // zombie
// This file implements the Boundary Value Caching technique for reducing variance
// of the walk-on-spheres and walk-on-stars estimators at a set of user-selected
// evaluation points via sample caching and reuse.
//
// Resources:
// - Boundary Value Caching for Walk on Spheres [2023]

#pragma once

#include <zombie/point_estimation/walk_on_stars.h>

namespace zombie {

namespace bvc {

template <typename T, size_t DIM>
struct EvaluationPoint {
    // constructor
    EvaluationPoint(const Vector<DIM>& pt_,
                    const Vector<DIM>& normal_,
                    SampleType type_,
                    float distToAbsorbingBoundary_,
                    float distToReflectingBoundary_);

    // returns estimated solution
    T getEstimatedSolution() const;

    // returns estimated gradient
    void getEstimatedGradient(std::vector<T>& gradient) const;

    // resets statistics
    void reset();

    // members
    Vector<DIM> pt;
    Vector<DIM> normal;
    SampleType type;
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;

protected:
    // members
    std::unique_ptr<SampleStatistics<T, DIM>> absorbingBoundaryStatistics;
    std::unique_ptr<SampleStatistics<T, DIM>> absorbingBoundaryNormalAlignedStatistics;
    std::unique_ptr<SampleStatistics<T, DIM>> reflectingBoundaryStatistics;
    std::unique_ptr<SampleStatistics<T, DIM>> reflectingBoundaryNormalAlignedStatistics;
    std::unique_ptr<SampleStatistics<T, DIM>> sourceStatistics;

    template <typename A, size_t B>
    friend class BoundaryValueCaching;
};

template <typename T, size_t DIM>
class BoundaryValueCaching {
public:
    // constructor
    BoundaryValueCaching(const GeometricQueries<DIM>& queries_,
                         const WalkOnStars<T, DIM>& walkOnStars_);

    // solves the given PDE at the provided sample points
    void computeBoundaryEstimates(const PDE<T, DIM>& pde,
                                  const WalkSettings& walkSettings,
                                  int nWalksForSolutionEstimates,
                                  int nWalksForGradientEstimates,
                                  float robinCoeffCutoffForNormalDerivative,
                                  std::vector<SamplePoint<T, DIM>>& samplePts,
                                  bool useFiniteDifferences=false,
                                  bool runSingleThreaded=false,
                                  std::function<void(int,int)> reportProgress={}) const;

    // sets the source value at the provided sample points
    void setSourceValues(const PDE<T, DIM>& pde,
                         std::vector<SamplePoint<T, DIM>>& samplePts,
                         bool runSingleThreaded=false) const;

    // splats sample pt data to the input evaluation pt
    void splat(const PDE<T, DIM>& pde,
               const SamplePoint<T, DIM>& samplePt,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               EvaluationPoint<T, DIM>& evalPt) const;

    // splats sample pt data to the input evaluation pt
    void splat(const PDE<T, DIM>& pde,
               const std::vector<SamplePoint<T, DIM>>& samplePts,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               EvaluationPoint<T, DIM>& evalPt) const;

    // splats sample pt data to the input evaluation pts
    void splat(const PDE<T, DIM>& pde,
               const SamplePoint<T, DIM>& samplePt,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               std::vector<EvaluationPoint<T, DIM>>& evalPts,
               bool runSingleThreaded=false) const;

    // splats sample pt data to the input evaluation pts
    void splat(const PDE<T, DIM>& pde,
               const std::vector<SamplePoint<T, DIM>>& samplePts,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               std::vector<EvaluationPoint<T, DIM>>& evalPts,
               std::function<void(int, int)> reportProgress={}) const;

    // estimates the solution at the input evaluation pt near the boundary
    void estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                      const WalkSettings& walkSettings,
                                      bool useDistanceToAbsorbingBoundary,
                                      float cutoffDistToBoundary, int nWalks,
                                      EvaluationPoint<T, DIM>& evalPt) const;

    // estimates the solution at the input evaluation pts near the boundary
    void estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                      const WalkSettings& walkSettings,
                                      bool useDistanceToAbsorbingBoundary,
                                      float cutoffDistToBoundary, int nWalks,
                                      std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                      bool runSingleThreaded=false) const;

protected:
    // sets estimation data for each sample point to compute boundary estimates
    void setEstimationData(const PDE<T, DIM>& pde,
                           const WalkSettings& walkSettings,
                           int nWalksForSolutionEstimates,
                           int nWalksForGradientEstimates,
                           float robinCoeffCutoffForNormalDerivative,
                           bool useFiniteDifferences,
                           std::vector<SampleEstimationData<DIM>>& estimationData,
                           std::vector<SamplePoint<T, DIM>>& samplePts) const;

    // sets the estimated boundary data for each sample point
    void setEstimatedBoundaryData(const PDE<T, DIM>& pde,
                                  const WalkSettings& walkSettings,
                                  float robinCoeffCutoffForNormalDerivative,
                                  bool useFiniteDifferences,
                                  std::vector<SamplePoint<T, DIM>>& samplePts) const;

    // splats boundary sample data
    void splatBoundaryData(const SamplePoint<T, DIM>& samplePt,
                           const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                           float radiusClamp,
                           float kernelRegularization,
                           float robinCoeffCutoffForNormalDerivative,
                           EvaluationPoint<T, DIM>& evalPt) const;

    // splats source sample data
    void splatSourceData(const SamplePoint<T, DIM>& samplePt,
                         const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                         float radiusClamp,
                         float kernelRegularization,
                         EvaluationPoint<T, DIM>& evalPt) const;

    // members
    const GeometricQueries<DIM>& queries;
    const WalkOnStars<T, DIM>& walkOnStars;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation
// FUTURE:
// - virtual boundary creation and estimation
// - bias correction/compensation
// - Barnes-Hut acceleration for splatting

template <typename T, size_t DIM>
inline EvaluationPoint<T, DIM>::EvaluationPoint(const Vector<DIM>& pt_,
                                                const Vector<DIM>& normal_,
                                                SampleType type_,
                                                float distToAbsorbingBoundary_,
                                                float distToReflectingBoundary_):
                                                pt(pt_), normal(normal_), type(type_),
                                                distToAbsorbingBoundary(distToAbsorbingBoundary_),
                                                distToReflectingBoundary(distToReflectingBoundary_) {
    absorbingBoundaryStatistics = std::make_unique<SampleStatistics<T, DIM>>();
    absorbingBoundaryNormalAlignedStatistics = std::make_unique<SampleStatistics<T, DIM>>();
    reflectingBoundaryStatistics = std::make_unique<SampleStatistics<T, DIM>>();
    reflectingBoundaryNormalAlignedStatistics = std::make_unique<SampleStatistics<T, DIM>>();
    sourceStatistics = std::make_unique<SampleStatistics<T, DIM>>();
}

template <typename T, size_t DIM>
inline T EvaluationPoint<T, DIM>::getEstimatedSolution() const {
    T solution = absorbingBoundaryStatistics->getEstimatedSolution();
    solution += absorbingBoundaryNormalAlignedStatistics->getEstimatedSolution();
    solution += reflectingBoundaryStatistics->getEstimatedSolution();
    solution += reflectingBoundaryNormalAlignedStatistics->getEstimatedSolution();
    solution += sourceStatistics->getEstimatedSolution();

    return solution;
}

template <typename T, size_t DIM>
inline void EvaluationPoint<T, DIM>::getEstimatedGradient(std::vector<T>& gradient) const {
    gradient.resize(DIM);
    for (int i = 0; i < DIM; i++) {
        gradient[i] = absorbingBoundaryStatistics->getEstimatedGradient()[i];
        gradient[i] += absorbingBoundaryNormalAlignedStatistics->getEstimatedGradient()[i];
        gradient[i] += reflectingBoundaryStatistics->getEstimatedGradient()[i];
        gradient[i] += reflectingBoundaryNormalAlignedStatistics->getEstimatedGradient()[i];
        gradient[i] += sourceStatistics->getEstimatedGradient()[i];
    }
}

template <typename T, size_t DIM>
inline void EvaluationPoint<T, DIM>::reset() {
    absorbingBoundaryStatistics->reset();
    absorbingBoundaryNormalAlignedStatistics->reset();
    reflectingBoundaryStatistics->reset();
    reflectingBoundaryNormalAlignedStatistics->reset();
    sourceStatistics->reset();
}

template <typename T, size_t DIM>
inline BoundaryValueCaching<T, DIM>::BoundaryValueCaching(const GeometricQueries<DIM>& queries_,
                                                          const WalkOnStars<T, DIM>& walkOnStars_):
                                                          queries(queries_), walkOnStars(walkOnStars_) {
    // do nothing
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::computeBoundaryEstimates(const PDE<T, DIM>& pde,
                                                                   const WalkSettings& walkSettings,
                                                                   int nWalksForSolutionEstimates,
                                                                   int nWalksForGradientEstimates,
                                                                   float robinCoeffCutoffForNormalDerivative,
                                                                   std::vector<SamplePoint<T, DIM>>& samplePts,
                                                                   bool useFiniteDifferences,
                                                                   bool runSingleThreaded,
                                                                   std::function<void(int,int)> reportProgress) const {
    // initialize estimation quantities
    std::vector<SampleEstimationData<DIM>> estimationData;
    setEstimationData(pde, walkSettings, nWalksForSolutionEstimates,
                      nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                      useFiniteDifferences, estimationData, samplePts);

    // compute estimates
    walkOnStars.solve(pde, walkSettings, estimationData, samplePts,
                      runSingleThreaded, reportProgress);

    // set estimated boundary data
    setEstimatedBoundaryData(pde, walkSettings, robinCoeffCutoffForNormalDerivative,
                             useFiniteDifferences, samplePts);
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::setSourceValues(const PDE<T, DIM>& pde,
                                                          std::vector<SamplePoint<T, DIM>>& samplePts,
                                                          bool runSingleThreaded) const {
    int nSamplePoints = (int)samplePts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nSamplePoints; i++) {
            samplePts[i].source = pde.source(samplePts[i].pt);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                samplePts[i].source = pde.source(samplePts[i].pt);
            }
        };

        tbb::blocked_range<int> range(0, nSamplePoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splat(const PDE<T, DIM>& pde,
                                                const SamplePoint<T, DIM>& samplePt,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                EvaluationPoint<T, DIM>& evalPt) const {
    // don't evaluate if the distance to the boundary is smaller than the cutoff distance
    if (evalPt.distToAbsorbingBoundary < cutoffDistToAbsorbingBoundary ||
        evalPt.distToReflectingBoundary < cutoffDistToReflectingBoundary) return;

    // initialize the greens function
    std::unique_ptr<GreensFnFreeSpace<DIM>> greensFn = nullptr;
    if (pde.absorptionCoeff > 0.0f) {
        greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.absorptionCoeff);

    } else {
        greensFn = std::make_unique<HarmonicGreensFnFreeSpace<DIM>>();
    }

    greensFn->updatePole(evalPt.pt);

    // evaluate
    if (samplePt.type == SampleType::OnAbsorbingBoundary ||
        samplePt.type == SampleType::OnReflectingBoundary) {
        splatBoundaryData(samplePt, greensFn, radiusClamp, kernelRegularization,
                          robinCoeffCutoffForNormalDerivative, evalPt);

    } else {
        splatSourceData(samplePt, greensFn, radiusClamp, kernelRegularization, evalPt);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splat(const PDE<T, DIM>& pde,
                                                const std::vector<SamplePoint<T, DIM>>& samplePts,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                EvaluationPoint<T, DIM>& evalPt) const {
    // don't evaluate if the distance to the boundary is smaller than the cutoff distance
    if (evalPt.distToAbsorbingBoundary < cutoffDistToAbsorbingBoundary ||
        evalPt.distToReflectingBoundary < cutoffDistToReflectingBoundary) return;

    // initialize the greens function
    std::unique_ptr<GreensFnFreeSpace<DIM>> greensFn = nullptr;
    if (pde.absorptionCoeff > 0.0f) {
        greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.absorptionCoeff);

    } else {
        greensFn = std::make_unique<HarmonicGreensFnFreeSpace<DIM>>();
    }

    greensFn->updatePole(evalPt.pt);

    // evaluate
    for (int i = 0; i < (int)samplePts.size(); i++) {
        if (samplePts[i].type == SampleType::OnAbsorbingBoundary ||
            samplePts[i].type == SampleType::OnReflectingBoundary) {
            splatBoundaryData(samplePts[i], greensFn, radiusClamp, kernelRegularization,
                              robinCoeffCutoffForNormalDerivative, evalPt);

        } else {
            splatSourceData(samplePts[i], greensFn, radiusClamp, kernelRegularization, evalPt);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splat(const PDE<T, DIM>& pde,
                                                const SamplePoint<T, DIM>& samplePt,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                bool runSingleThreaded) const {
    int nEvalPoints = (int)evalPts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nEvalPoints; i++) {
            splat(pde, samplePt, radiusClamp, kernelRegularization,
                  robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                  cutoffDistToReflectingBoundary, evalPts[i]);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                splat(pde, samplePt, radiusClamp, kernelRegularization,
                      robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                      cutoffDistToReflectingBoundary, evalPts[i]);
            }
        };

        tbb::blocked_range<int> range(0, nEvalPoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splat(const PDE<T, DIM>& pde,
                                                const std::vector<SamplePoint<T, DIM>>& samplePts,
                                                float radiusClamp,
                                                float kernelRegularization,
                                                float robinCoeffCutoffForNormalDerivative,
                                                float cutoffDistToAbsorbingBoundary,
                                                float cutoffDistToReflectingBoundary,
                                                std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                std::function<void(int, int)> reportProgress) const {
    const int reportGranularity = 100;
    for (int i = 0; i < (int)samplePts.size(); i++) {
        splat(pde, samplePts[i], radiusClamp, kernelRegularization,
              robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
              cutoffDistToReflectingBoundary, evalPts);

        if (reportProgress && (i + 1)%reportGranularity == 0) {
            reportProgress(reportGranularity, 0);
        }
    }

    if (reportProgress) {
        reportProgress(samplePts.size()%reportGranularity, 0);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       bool useDistanceToAbsorbingBoundary,
                                                                       float cutoffDistToBoundary, int nWalks,
                                                                       EvaluationPoint<T, DIM>& evalPt) const {
    float distToBoundary = useDistanceToAbsorbingBoundary ?
                           evalPt.distToAbsorbingBoundary :
                           evalPt.distToReflectingBoundary;
    if (distToBoundary < cutoffDistToBoundary) {
        // NOTE: When the evaluation pt is on the boundary, this setup
        // evaluates the inward boundary normal aligned solution
        SamplePoint<T, DIM> samplePt(evalPt.pt, evalPt.normal, evalPt.type, 1.0f,
                                     evalPt.distToAbsorbingBoundary,
                                     evalPt.distToReflectingBoundary);
        SampleEstimationData<DIM> estimationData(nWalks, EstimationQuantity::Solution);
        walkOnStars.solve(pde, walkSettings, estimationData, samplePt);

        // update statistics
        evalPt.reset();
        T solutionEstimate = samplePt.statistics->getEstimatedSolution();
        if (evalPt.type == SampleType::OnAbsorbingBoundary) {
            evalPt.absorbingBoundaryStatistics->addSolutionEstimate(solutionEstimate);

        } else if (evalPt.type == SampleType::OnReflectingBoundary) {
            evalPt.reflectingBoundaryStatistics->addSolutionEstimate(solutionEstimate);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       bool useDistanceToAbsorbingBoundary,
                                                                       float cutoffDistToBoundary, int nWalks,
                                                                       std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                                       bool runSingleThreaded) const {
    int nEvalPoints = (int)evalPts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nEvalPoints; i++) {
            estimateSolutionNearBoundary(pde, walkSettings, useDistanceToAbsorbingBoundary,
                                         cutoffDistToBoundary, nWalks, evalPts[i]);
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                estimateSolutionNearBoundary(pde, walkSettings, useDistanceToAbsorbingBoundary,
                                             cutoffDistToBoundary, nWalks, evalPts[i]);
            }
        };

        tbb::blocked_range<int> range(0, nEvalPoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::setEstimationData(const PDE<T, DIM>& pde,
                                                            const WalkSettings& walkSettings,
                                                            int nWalksForSolutionEstimates,
                                                            int nWalksForGradientEstimates,
                                                            float robinCoeffCutoffForNormalDerivative,
                                                            bool useFiniteDifferences,
                                                            std::vector<SampleEstimationData<DIM>>& estimationData,
                                                            std::vector<SamplePoint<T, DIM>>& samplePts) const {
    int nSamples = (int)samplePts.size();
    estimationData.resize(nSamples);
    for (int i = 0; i < nSamples; i++) {
        SamplePoint<T, DIM>& samplePt = samplePts[i];

        if (samplePt.type == SampleType::OnAbsorbingBoundary) {
            if (useFiniteDifferences) {
                estimationData[i].estimationQuantity = EstimationQuantity::Solution;
                samplePt.type = SampleType::InDomain;

            } else {
                Vector<DIM> normal = samplePt.normal;
                if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
                    normal *= -1.0f;
                }

                estimationData[i].estimationQuantity = EstimationQuantity::SolutionAndGradient;
                estimationData[i].directionForDerivative = normal;
            }

            estimationData[i].nWalks = nWalksForGradientEstimates;

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            if (!pde.areRobinConditionsPureNeumann) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                samplePt.robinCoeff = pde.robinCoeff(samplePt.pt, returnBoundaryNormalAlignedValue);
            }

            if (samplePt.robinCoeff > robinCoeffCutoffForNormalDerivative) {
                Vector<DIM> normal = samplePt.normal;
                if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
                    normal *= -1.0f;
                }

                estimationData[i].estimationQuantity = EstimationQuantity::SolutionAndGradient;
                estimationData[i].directionForDerivative = normal;
                estimationData[i].nWalks = nWalksForGradientEstimates;

            } else {
                estimationData[i].estimationQuantity = EstimationQuantity::Solution;
                estimationData[i].nWalks = nWalksForSolutionEstimates;
            }

        } else {
            std::cerr << "BoundaryValueCaching::setEstimationData(): Invalid sample type!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::setEstimatedBoundaryData(const PDE<T, DIM>& pde,
                                                                   const WalkSettings& walkSettings,
                                                                   float robinCoeffCutoffForNormalDerivative,
                                                                   bool useFiniteDifferences,
                                                                   std::vector<SamplePoint<T, DIM>>& samplePts) const {
    for (int i = 0; i < (int)samplePts.size(); i++) {
        SamplePoint<T, DIM>& samplePt = samplePts[i];
        samplePt.solution = samplePt.statistics->getEstimatedSolution();

        if (samplePt.type == SampleType::OnReflectingBoundary) {
            if (!walkSettings.ignoreReflectingBoundaryContribution) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                if (pde.areRobinConditionsPureNeumann) {
                    samplePt.normalDerivative = pde.robin(samplePt.pt, returnBoundaryNormalAlignedValue);

                } else {
                    samplePt.robin = pde.robin(samplePt.pt, returnBoundaryNormalAlignedValue);
                    if (samplePt.robinCoeff > robinCoeffCutoffForNormalDerivative) {
                        samplePt.normalDerivative = samplePt.statistics->getEstimatedDerivative();
                    }
                }
            }

        } else {
            if (useFiniteDifferences) {
                // use biased gradient estimates
                float signedDistance;
                Vector<DIM> normal;
                Vector<DIM> pt = samplePt.pt;
                queries.projectToAbsorbingBoundary(pt, normal, signedDistance, walkSettings.solveDoubleSided);

                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        signedDistance > 0.0f;
                T dirichlet = pde.dirichlet(pt, returnBoundaryNormalAlignedValue);

                samplePt.normalDerivative = dirichlet - samplePt.solution;
                samplePt.normalDerivative /= std::fabs(signedDistance);
                samplePt.type = SampleType::OnAbsorbingBoundary;

            } else {
                // use unbiased gradient estimates
                samplePt.normalDerivative = samplePt.statistics->getEstimatedDerivative();
            }
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splatBoundaryData(const SamplePoint<T, DIM>& samplePt,
                                                            const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                                                            float radiusClamp,
                                                            float kernelRegularization,
                                                            float robinCoeffCutoffForNormalDerivative,
                                                            EvaluationPoint<T, DIM>& evalPt) const {
    // compute the contribution of the boundary sample
    const T& solution = samplePt.solution;
    const T& normalDerivative = samplePt.normalDerivative;
    const T& robin = samplePt.robin;
    const Vector<DIM>& pt = samplePt.pt;
    Vector<DIM> n = samplePt.normal*(samplePt.estimateBoundaryNormalAligned ? -1.0f : 1.0f);
    float pdf = samplePt.pdf;
    float robinCoeff = samplePt.robinCoeff;

    float r = std::max(radiusClamp, (pt - greensFn->x).norm());
    float G = greensFn->evaluate(r);
    float P = greensFn->poissonKernel(r, pt, n);
    Vector<DIM> dG = greensFn->gradient(r, pt);
    Vector<DIM> dP = greensFn->poissonKernelGradient(r, pt, n);
    float dGNorm = dG.norm();
    float dPNorm = dP.norm();

    if (std::isinf(G) || std::isinf(P) || std::isinf(dGNorm) || std::isinf(dPNorm) ||
        std::isnan(G) || std::isnan(P) || std::isnan(dGNorm) || std::isnan(dPNorm)) {
        return;
    }

    if (kernelRegularization > 0.0f) {
        r /= kernelRegularization;
        G *= KernelRegularization<DIM>::regularizationForGreensFn(r);
        P *= KernelRegularization<DIM>::regularizationForPoissonKernel(r);
    }

    T solutionEstimate;
    T gradientEstimate[DIM];
    float alpha = evalPt.type == SampleType::OnAbsorbingBoundary ||
                  evalPt.type == SampleType::OnReflectingBoundary ?
                  2.0f : 1.0f;

    if (robinCoeff > robinCoeffCutoffForNormalDerivative) {
        solutionEstimate = alpha*((G + P/robinCoeff)*normalDerivative - P*robin/robinCoeff)/pdf;

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*((dG[i] + dP[i]/robinCoeff)*normalDerivative - dP[i]*robin/robinCoeff)/pdf;
        }

    } else if (robinCoeff > 0.0f) {
        solutionEstimate = alpha*(G*robin - (P + robinCoeff*G)*solution)/pdf;

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*(dG[i]*robin - (dP[i] + robinCoeff*dG[i])*solution)/pdf;
        }

    } else {
        solutionEstimate = alpha*(G*normalDerivative - P*solution)/pdf;

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*(dG[i]*normalDerivative - dP[i]*solution)/pdf;
        }
    }

    // update statistics
    if (samplePt.estimateBoundaryNormalAligned) {
        if (samplePt.type == SampleType::OnAbsorbingBoundary) {
            evalPt.absorbingBoundaryNormalAlignedStatistics->addSolutionEstimate(solutionEstimate);
            evalPt.absorbingBoundaryNormalAlignedStatistics->addGradientEstimate(gradientEstimate);

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            evalPt.reflectingBoundaryNormalAlignedStatistics->addSolutionEstimate(solutionEstimate);
            evalPt.reflectingBoundaryNormalAlignedStatistics->addGradientEstimate(gradientEstimate);
        }

    } else {
        if (samplePt.type == SampleType::OnAbsorbingBoundary) {
            evalPt.absorbingBoundaryStatistics->addSolutionEstimate(solutionEstimate);
            evalPt.absorbingBoundaryStatistics->addGradientEstimate(gradientEstimate);

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            evalPt.reflectingBoundaryStatistics->addSolutionEstimate(solutionEstimate);
            evalPt.reflectingBoundaryStatistics->addGradientEstimate(gradientEstimate);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splatSourceData(const SamplePoint<T, DIM>& samplePt,
                                                          const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                                                          float radiusClamp,
                                                          float kernelRegularization,
                                                          EvaluationPoint<T, DIM>& evalPt) const {
    // compute the contribution of the source sample
    const T& source = samplePt.source;
    const Vector<DIM>& pt = samplePt.pt;
    float pdf = samplePt.pdf;

    float r = std::max(radiusClamp, (pt - greensFn->x).norm());
    float G = greensFn->evaluate(r);
    Vector<DIM> dG = greensFn->gradient(r, pt);
    float dGNorm = dG.norm();

    if (std::isinf(G) || std::isnan(G) || std::isinf(dGNorm) || std::isnan(dGNorm)) {
        return;
    }

    if (kernelRegularization > 0.0f) {
        r /= kernelRegularization;
        G *= KernelRegularization<DIM>::regularizationForGreensFn(r);
    }

    float alpha = evalPt.type == SampleType::OnAbsorbingBoundary ||
                  evalPt.type == SampleType::OnReflectingBoundary ?
                  2.0f : 1.0f;
    T solutionEstimate = alpha*G*source/pdf;

    T gradientEstimate[DIM];
    if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
    for (int i = 0; i < DIM; i++) {
        gradientEstimate[i] = alpha*dG[i]*source/pdf;
    }

    // update statistics
    evalPt.sourceStatistics->addSolutionEstimate(solutionEstimate);
    evalPt.sourceStatistics->addGradientEstimate(gradientEstimate);
}

} // bvc

} // zombie

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

    // returns estimated gradient coordinate value
    T getEstimatedGradient(int coord) const;

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
    SampleStatistics<T, DIM> absorbingBoundaryStatistics;
    SampleStatistics<T, DIM> absorbingBoundaryNormalAlignedStatistics;
    SampleStatistics<T, DIM> reflectingBoundaryStatistics;
    SampleStatistics<T, DIM> reflectingBoundaryNormalAlignedStatistics;
    SampleStatistics<T, DIM> sourceStatistics;

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
                           std::vector<int>& nWalks,
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

template <typename T, size_t DIM>
class BoundaryValueCachingSolver {
public:
    // constructor
    BoundaryValueCachingSolver(const GeometricQueries<DIM>& queries_,
                               std::shared_ptr<BoundarySampler<T, DIM>> absorbingBoundarySampler_,
                               std::shared_ptr<BoundarySampler<T, DIM>> reflectingBoundarySampler_,
                               std::shared_ptr<DomainSampler<T, DIM>> domainSampler_);

    // generates boundary and domain samples
    void generateSamples(int absorbingBoundaryCacheSize,
                         int reflectingBoundaryCacheSize,
                         int domainCacheSize,
                         float normalOffsetForAbsorbingBoundary,
                         float normalOffsetForReflectingBoundary,
                         bool solveDoubleSided);

    // computes sample estimates on the boundary
    void computeSampleEstimates(const PDE<T, DIM>& pde,
                                const WalkSettings& walkSettings,
                                int nWalksForSolutionEstimates,
                                int nWalksForGradientEstimates,
                                float robinCoeffCutoffForNormalDerivative,
                                bool useFiniteDifferences=false,
                                bool runSingleThreaded=false,
                                std::function<void(int,int)> reportProgress={});

    // splats solution and gradient estimates into the interior
    void splat(const PDE<T, DIM>& pde,
               float radiusClamp,
               float kernelRegularization,
               float robinCoeffCutoffForNormalDerivative,
               float cutoffDistToAbsorbingBoundary,
               float cutoffDistToReflectingBoundary,
               std::vector<EvaluationPoint<T, DIM>>& evalPts,
               std::function<void(int, int)> reportProgress={}) const;

    // estimates the solution at the input evaluation points near the boundary
    void estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                      const WalkSettings& walkSettings,
                                      float cutoffDistToAbsorbingBoundary,
                                      float cutoffDistToReflectingBoundary,
                                      int nWalksForSolutionEstimates,
                                      std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                      bool runSingleThreaded=false) const;

    // returns the boundary and domain cache
    const std::vector<SamplePoint<T, DIM>>& getAbsorbingBoundaryCache(bool returnBoundaryNormalAligned=false) const;
    const std::vector<SamplePoint<T, DIM>>& getReflectingBoundaryCache(bool returnBoundaryNormalAligned=false) const;
    const std::vector<SamplePoint<T, DIM>>& getDomainCache() const;

protected:
    // members
    const GeometricQueries<DIM>& queries;
    std::shared_ptr<BoundarySampler<T, DIM>> absorbingBoundarySampler;
    std::shared_ptr<BoundarySampler<T, DIM>> reflectingBoundarySampler;
    std::shared_ptr<DomainSampler<T, DIM>> domainSampler;
    WalkOnStars<T, DIM> walkOnStars;
    BoundaryValueCaching<T, DIM> boundaryValueCaching;
    std::vector<SamplePoint<T, DIM>> absorbingBoundaryCache;
    std::vector<SamplePoint<T, DIM>> absorbingBoundaryCacheNormalAligned;
    std::vector<SamplePoint<T, DIM>> reflectingBoundaryCache;
    std::vector<SamplePoint<T, DIM>> reflectingBoundaryCacheNormalAligned;
    std::vector<SamplePoint<T, DIM>> domainCache;
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
                                                distToReflectingBoundary(distToReflectingBoundary_)
{

}

template <typename T, size_t DIM>
inline T EvaluationPoint<T, DIM>::getEstimatedSolution() const
{
    T solution = absorbingBoundaryStatistics.getEstimatedSolution();
    solution += absorbingBoundaryNormalAlignedStatistics.getEstimatedSolution();
    solution += reflectingBoundaryStatistics.getEstimatedSolution();
    solution += reflectingBoundaryNormalAlignedStatistics.getEstimatedSolution();
    solution += sourceStatistics.getEstimatedSolution();

    return solution;
}

template <typename T, size_t DIM>
inline void EvaluationPoint<T, DIM>::getEstimatedGradient(std::vector<T>& gradient) const
{
    gradient.resize(DIM);
    for (int i = 0; i < DIM; i++) {
        gradient[i] = absorbingBoundaryStatistics.getEstimatedGradient()[i];
        gradient[i] += absorbingBoundaryNormalAlignedStatistics.getEstimatedGradient()[i];
        gradient[i] += reflectingBoundaryStatistics.getEstimatedGradient()[i];
        gradient[i] += reflectingBoundaryNormalAlignedStatistics.getEstimatedGradient()[i];
        gradient[i] += sourceStatistics.getEstimatedGradient()[i];
    }
}

template <typename T, size_t DIM>
inline T EvaluationPoint<T, DIM>::getEstimatedGradient(int coord) const
{
    T gradient = absorbingBoundaryStatistics.getEstimatedGradient()[coord];
    gradient += absorbingBoundaryNormalAlignedStatistics.getEstimatedGradient()[coord];
    gradient += reflectingBoundaryStatistics.getEstimatedGradient()[coord];
    gradient += reflectingBoundaryNormalAlignedStatistics.getEstimatedGradient()[coord];
    gradient += sourceStatistics.getEstimatedGradient()[coord];

    return gradient;
}

template <typename T, size_t DIM>
inline void EvaluationPoint<T, DIM>::reset()
{
    absorbingBoundaryStatistics.reset();
    absorbingBoundaryNormalAlignedStatistics.reset();
    reflectingBoundaryStatistics.reset();
    reflectingBoundaryNormalAlignedStatistics.reset();
    sourceStatistics.reset();
}

template <typename T, size_t DIM>
inline BoundaryValueCaching<T, DIM>::BoundaryValueCaching(const GeometricQueries<DIM>& queries_,
                                                          const WalkOnStars<T, DIM>& walkOnStars_):
                                                          queries(queries_), walkOnStars(walkOnStars_)
{
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
                                                                   std::function<void(int,int)> reportProgress) const
{
    // initialize estimation quantities
    std::vector<int> nWalks;
    setEstimationData(pde, walkSettings, nWalksForSolutionEstimates,
                      nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                      useFiniteDifferences, nWalks, samplePts);

    // compute estimates
    walkOnStars.solve(pde, walkSettings, nWalks, samplePts, runSingleThreaded, reportProgress);

    // set estimated boundary data
    setEstimatedBoundaryData(pde, walkSettings, robinCoeffCutoffForNormalDerivative,
                             useFiniteDifferences, samplePts);
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::setSourceValues(const PDE<T, DIM>& pde,
                                                          std::vector<SamplePoint<T, DIM>>& samplePts,
                                                          bool runSingleThreaded) const
{
    int nSamplePoints = (int)samplePts.size();
    if (runSingleThreaded) {
        for (int i = 0; i < nSamplePoints; i++) {
            samplePts[i].contribution = pde.source(samplePts[i].pt);
            samplePts[i].estimationQuantity = EstimationQuantity::None;
        }

    } else {
        auto run = [&](const tbb::blocked_range<int>& range) {
            for (int i = range.begin(); i < range.end(); ++i) {
                samplePts[i].contribution = pde.source(samplePts[i].pt);
                samplePts[i].estimationQuantity = EstimationQuantity::None;
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
                                                EvaluationPoint<T, DIM>& evalPt) const
{
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
                                                EvaluationPoint<T, DIM>& evalPt) const
{
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
                                                bool runSingleThreaded) const
{
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
                                                std::function<void(int, int)> reportProgress) const
{
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
                                                                       EvaluationPoint<T, DIM>& evalPt) const
{
    float distToBoundary = useDistanceToAbsorbingBoundary ? evalPt.distToAbsorbingBoundary :
                                                            evalPt.distToReflectingBoundary;
    if (distToBoundary < cutoffDistToBoundary) {
        // NOTE: When the evaluation pt is on the boundary, this setup
        // evaluates the inward boundary normal aligned solution
        SamplePoint<T, DIM> samplePt(evalPt.pt, evalPt.normal, evalPt.type,
                                     EstimationQuantity::Solution, 1.0f,
                                     evalPt.distToAbsorbingBoundary,
                                     evalPt.distToReflectingBoundary);
        walkOnStars.solve(pde, walkSettings, nWalks, samplePt);

        // update statistics
        evalPt.reset();
        T solutionEstimate = samplePt.statistics.getEstimatedSolution();
        if (evalPt.type == SampleType::OnAbsorbingBoundary) {
            evalPt.absorbingBoundaryStatistics.addSolutionEstimate(solutionEstimate);

        } else if (evalPt.type == SampleType::OnReflectingBoundary) {
            evalPt.reflectingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       bool useDistanceToAbsorbingBoundary,
                                                                       float cutoffDistToBoundary, int nWalks,
                                                                       std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                                       bool runSingleThreaded) const
{
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
                                                            std::vector<int>& nWalks,
                                                            std::vector<SamplePoint<T, DIM>>& samplePts) const
{
    int nSamples = (int)samplePts.size();
    nWalks.resize(nSamples, 0);
    for (int i = 0; i < nSamples; i++) {
        SamplePoint<T, DIM>& samplePt = samplePts[i];

        if (samplePt.type == SampleType::OnAbsorbingBoundary) {
            if (useFiniteDifferences) {
                samplePt.type = SampleType::InDomain;
                samplePt.estimationQuantity = EstimationQuantity::Solution;

            } else {
                Vector<DIM> normal = samplePt.normal;
                if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
                    normal *= -1.0f;
                }

                samplePt.directionForDerivative = normal;
                samplePt.estimationQuantity = EstimationQuantity::SolutionAndGradient;
            }

            nWalks[i] = nWalksForGradientEstimates;

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            if (!pde.areRobinConditionsPureNeumann) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                samplePt.robinCoeff = pde.robinCoeff(samplePt.pt, samplePt.normal,
                                                     returnBoundaryNormalAlignedValue);
            }

            if (std::fabs(samplePt.robinCoeff) > robinCoeffCutoffForNormalDerivative) {
                Vector<DIM> normal = samplePt.normal;
                if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
                    normal *= -1.0f;
                }

                nWalks[i] = nWalksForGradientEstimates;
                samplePt.directionForDerivative = normal;
                samplePt.estimationQuantity = EstimationQuantity::SolutionAndGradient;

            } else {
                nWalks[i] = nWalksForSolutionEstimates;
                samplePt.estimationQuantity = EstimationQuantity::Solution;
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
                                                                   std::vector<SamplePoint<T, DIM>>& samplePts) const
{
    for (int i = 0; i < (int)samplePts.size(); i++) {
        SamplePoint<T, DIM>& samplePt = samplePts[i];
        samplePt.solution = samplePt.statistics.getEstimatedSolution();

        if (samplePt.type == SampleType::OnReflectingBoundary) {
            if (!walkSettings.ignoreReflectingBoundaryContribution) {
                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        samplePt.estimateBoundaryNormalAligned;
                if (pde.areRobinConditionsPureNeumann) {
                    samplePt.normalDerivative = pde.robin(samplePt.pt, samplePt.normal,
                                                          returnBoundaryNormalAlignedValue);

                } else {
                    samplePt.contribution = pde.robin(samplePt.pt, samplePt.normal,
                                                      returnBoundaryNormalAlignedValue);
                    if (std::fabs(samplePt.robinCoeff) > robinCoeffCutoffForNormalDerivative) {
                        samplePt.normalDerivative = samplePt.statistics.getEstimatedDerivative();
                    }
                }
            }

        } else {
            if (useFiniteDifferences) {
                // use biased gradient estimates
                float signedDistance = queries.computeDistToAbsorbingBoundary(samplePt.pt, true);
                Vector<DIM> pt = samplePt.pt - signedDistance*samplePt.normal;

                bool returnBoundaryNormalAlignedValue = walkSettings.solveDoubleSided &&
                                                        signedDistance > 0.0f;
                T dirichlet = !walkSettings.ignoreAbsorbingBoundaryContribution ?
                              pde.dirichlet(pt, returnBoundaryNormalAlignedValue) : T(0.0f);

                samplePt.normalDerivative = dirichlet - samplePt.solution;
                samplePt.normalDerivative /= std::fabs(signedDistance);
                samplePt.type = SampleType::OnAbsorbingBoundary;

            } else {
                // use unbiased gradient estimates
                samplePt.normalDerivative = samplePt.statistics.getEstimatedDerivative();
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
                                                            EvaluationPoint<T, DIM>& evalPt) const
{
    // compute the contribution of the boundary sample
    const T& solution = samplePt.solution;
    const T& normalDerivative = samplePt.normalDerivative;
    const T& robin = samplePt.contribution;
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

    if (std::fabs(robinCoeff) > robinCoeffCutoffForNormalDerivative) {
        solutionEstimate = alpha*((G + P/robinCoeff)*normalDerivative - P*robin/robinCoeff)/pdf;

        if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
        for (int i = 0; i < DIM; i++) {
            gradientEstimate[i] = alpha*((dG[i] + dP[i]/robinCoeff)*normalDerivative - dP[i]*robin/robinCoeff)/pdf;
        }

    } else if (std::fabs(robinCoeff) > 0.0f) {
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
            evalPt.absorbingBoundaryNormalAlignedStatistics.addSolutionEstimate(solutionEstimate);
            evalPt.absorbingBoundaryNormalAlignedStatistics.addGradientEstimate(gradientEstimate);

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            evalPt.reflectingBoundaryNormalAlignedStatistics.addSolutionEstimate(solutionEstimate);
            evalPt.reflectingBoundaryNormalAlignedStatistics.addGradientEstimate(gradientEstimate);
        }

    } else {
        if (samplePt.type == SampleType::OnAbsorbingBoundary) {
            evalPt.absorbingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
            evalPt.absorbingBoundaryStatistics.addGradientEstimate(gradientEstimate);

        } else if (samplePt.type == SampleType::OnReflectingBoundary) {
            evalPt.reflectingBoundaryStatistics.addSolutionEstimate(solutionEstimate);
            evalPt.reflectingBoundaryStatistics.addGradientEstimate(gradientEstimate);
        }
    }
}

template <typename T, size_t DIM>
inline void BoundaryValueCaching<T, DIM>::splatSourceData(const SamplePoint<T, DIM>& samplePt,
                                                          const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
                                                          float radiusClamp,
                                                          float kernelRegularization,
                                                          EvaluationPoint<T, DIM>& evalPt) const
{
    // compute the contribution of the source sample
    const T& source = samplePt.contribution;
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
    evalPt.sourceStatistics.addSolutionEstimate(solutionEstimate);
    evalPt.sourceStatistics.addGradientEstimate(gradientEstimate);
}

template <typename T, size_t DIM>
inline BoundaryValueCachingSolver<T, DIM>::BoundaryValueCachingSolver(const GeometricQueries<DIM>& queries_,
                                                                      std::shared_ptr<BoundarySampler<T, DIM>> absorbingBoundarySampler_,
                                                                      std::shared_ptr<BoundarySampler<T, DIM>> reflectingBoundarySampler_,
                                                                      std::shared_ptr<DomainSampler<T, DIM>> domainSampler_):
                                                                      queries(queries_),
                                                                      absorbingBoundarySampler(absorbingBoundarySampler_),
                                                                      reflectingBoundarySampler(reflectingBoundarySampler_),
                                                                      domainSampler(domainSampler_), walkOnStars(queries),
                                                                      boundaryValueCaching(queries, walkOnStars)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::generateSamples(int absorbingBoundaryCacheSize,
                                                                int reflectingBoundaryCacheSize,
                                                                int domainCacheSize,
                                                                float normalOffsetForAbsorbingBoundary,
                                                                float normalOffsetForReflectingBoundary,
                                                                bool solveDoubleSided)
{
    absorbingBoundarySampler->generateSamples(absorbingBoundarySampler->getSampleCount(absorbingBoundaryCacheSize, false),
                                              SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                              queries, absorbingBoundaryCache, false);
    if (solveDoubleSided) {
        absorbingBoundarySampler->generateSamples(absorbingBoundarySampler->getSampleCount(absorbingBoundaryCacheSize, true),
                                                  SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                                  queries, absorbingBoundaryCacheNormalAligned, true);
    }

    reflectingBoundarySampler->generateSamples(reflectingBoundarySampler->getSampleCount(reflectingBoundaryCacheSize, false),
                                               SampleType::OnReflectingBoundary, normalOffsetForReflectingBoundary,
                                               queries, reflectingBoundaryCache, false);
    if (solveDoubleSided) {
        reflectingBoundarySampler->generateSamples(reflectingBoundarySampler->getSampleCount(reflectingBoundaryCacheSize, true),
                                                   SampleType::OnReflectingBoundary, normalOffsetForReflectingBoundary,
                                                   queries, reflectingBoundaryCacheNormalAligned, true);
    }

    domainSampler->generateSamples(domainCacheSize, queries, domainCache);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::computeSampleEstimates(const PDE<T, DIM>& pde,
                                                                       const WalkSettings& walkSettings,
                                                                       int nWalksForSolutionEstimates,
                                                                       int nWalksForGradientEstimates,
                                                                       float robinCoeffCutoffForNormalDerivative,
                                                                       bool useFiniteDifferences,
                                                                       bool runSingleThreaded,
                                                                       std::function<void(int,int)> reportProgress)
{
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  absorbingBoundaryCache, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  absorbingBoundaryCacheNormalAligned, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  reflectingBoundaryCache, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForSolutionEstimates,
                                                  nWalksForGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  reflectingBoundaryCacheNormalAligned, useFiniteDifferences,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.setSourceValues(pde, domainCache, runSingleThreaded);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::splat(const PDE<T, DIM>& pde,
                                                     float radiusClamp,
                                                     float kernelRegularization,
                                                     float robinCoeffCutoffForNormalDerivative,
                                                     float cutoffDistToAbsorbingBoundary,
                                                     float cutoffDistToReflectingBoundary,
                                                     std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                     std::function<void(int, int)> reportProgress) const
{
    boundaryValueCaching.splat(pde, absorbingBoundaryCache, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, absorbingBoundaryCacheNormalAligned, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, reflectingBoundaryCache, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, reflectingBoundaryCacheNormalAligned, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, domainCache, radiusClamp, kernelRegularization,
                               robinCoeffCutoffForNormalDerivative, cutoffDistToAbsorbingBoundary,
                               cutoffDistToReflectingBoundary, evalPts, reportProgress);
}

template <typename T, size_t DIM>
inline void BoundaryValueCachingSolver<T, DIM>::estimateSolutionNearBoundary(const PDE<T, DIM>& pde,
                                                                             const WalkSettings& walkSettings,
                                                                             float cutoffDistToAbsorbingBoundary,
                                                                             float cutoffDistToReflectingBoundary,
                                                                             int nWalksForSolutionEstimates,
                                                                             std::vector<EvaluationPoint<T, DIM>>& evalPts,
                                                                             bool runSingleThreaded) const
{
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, true, cutoffDistToAbsorbingBoundary,
                                                      nWalksForSolutionEstimates, evalPts, runSingleThreaded);
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, false, cutoffDistToReflectingBoundary,
                                                      nWalksForSolutionEstimates, evalPts, runSingleThreaded);
}

template <typename T, size_t DIM>
inline const std::vector<SamplePoint<T, DIM>>& BoundaryValueCachingSolver<T, DIM>::getAbsorbingBoundaryCache(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? absorbingBoundaryCacheNormalAligned : absorbingBoundaryCache;
}

template <typename T, size_t DIM>
inline const std::vector<SamplePoint<T, DIM>>& BoundaryValueCachingSolver<T, DIM>::getReflectingBoundaryCache(bool returnBoundaryNormalAligned) const
{
    return returnBoundaryNormalAligned ? reflectingBoundaryCacheNormalAligned : reflectingBoundaryCache;
}

template <typename T, size_t DIM>
inline const std::vector<SamplePoint<T, DIM>>& BoundaryValueCachingSolver<T, DIM>::getDomainCache() const
{
    return domainCache;
}

} // bvc

} // zombie

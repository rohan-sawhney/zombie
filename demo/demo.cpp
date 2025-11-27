// This file is the entry point for a 2D demo application demonstrating how to use Zombie.
// It reads a 'model problem' description from a JSON file, runs the WalkOnStars, BoundaryValueCaching
// or ReverseWalkonOnStars solvers, and writes the result to a PMF or PNG file.

#include "model_problem.h"
#include "grid.h"

using json = nlohmann::json;

template <size_t DIM>
void computeDistanceInfo(const std::vector<zombie::Vector<DIM>>& solveLocations,
                         const zombie::GeometricQueries<DIM>& queries,
                         bool solveDoubleSided, bool solveExterior,
                         std::vector<DistanceInfo>& distanceInfo)
{
    distanceInfo.resize(solveLocations.size());
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        zombie::Vector<DIM> pt = solveLocations[i];
        bool insideDomain = queries.insideDomain(pt);
        if (queries.domainIsWatertight && solveExterior) insideDomain = !insideDomain;
        distanceInfo[i].inValidSolveRegion = insideDomain || solveDoubleSided;
        distanceInfo[i].distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
        distanceInfo[i].distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);
    }
}

template <typename T, size_t DIM>
void createSamplePoints(const std::vector<zombie::Vector<DIM>>& solveLocations,
                        const std::vector<DistanceInfo>& distanceInfo,
                        std::vector<zombie::SamplePoint<T, DIM>>& samplePts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        if (distanceInfo[i].inValidSolveRegion) {
            zombie::Vector<DIM> pt = solveLocations[i];
            zombie::Vector<DIM> normal = zombie::Vector<DIM>::Zero();
            zombie::SampleType sampleType = zombie::SampleType::InDomain;
            zombie::EstimationQuantity estimationQuantity = zombie::EstimationQuantity::Solution;
            float pdf = 1.0f;
            float distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
            float distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;

            samplePts.emplace_back(zombie::SamplePoint<T, DIM>(pt, normal, sampleType,
                                                               estimationQuantity, pdf,
                                                               distToAbsorbingBoundary,
                                                               distToReflectingBoundary));
        }
    }
}

template <typename T, size_t DIM>
void runWalkOnStars(const json& solverConfig,
                    const zombie::GeometricQueries<DIM>& queries,
                    const zombie::PDE<T, DIM>& pde,
                    bool solveDoubleSided,
                    std::vector<zombie::SamplePoint<T, DIM>>& samplePts,
                    std::vector<zombie::SampleStatistics<T, DIM>>& sampleStatistics)
{
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // initialize solver and estimate solution
    ProgressBar pb(samplePts.size());
    std::function<void(int, int)> reportProgress = getReportProgressCallback(pb);

    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision,
                                      russianRouletteThreshold,
                                      splittingThreshold, maxWalkLength,
                                      stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres,
                                      solveDoubleSided,
                                      !disableGradientControlVariates,
                                      !disableGradientAntitheticVariates,
                                      useCosineSamplingForDirectionalDerivatives,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    std::vector<int> nWalksVector(samplePts.size(), nWalks);
    zombie::WalkOnStars<T, DIM> walkOnStars(queries);
    walkOnStars.solve(pde, walkSettings, nWalksVector, samplePts, sampleStatistics,
                      runSingleThreaded, reportProgress);
    pb.finish();
}

template<typename T, size_t DIM>
void runHarmonicCaching(const json&                                    solverConfig,
                        const zombie::GeometricQueries<DIM>&           queries,
                        const std::pair<Vector2, Vector2>&             bbox,
                        const zombie::PDE<T, DIM>&                     pde,
                        bool                                           solveDoubleSided,
                        std::vector<zombie::SamplePoint<T, DIM>>&      samplePts,
                        std::vector<zombie::SampleStatistics<T, DIM>>& sampleStatistics)
{
    // load config settings
    const float epsilonShellForAbsorbingBoundary  = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision               = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold          = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold                = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int nWalks                         = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength                  = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int nWalksNearBoundary             = getRequired<int>(solverConfig, "nWalksNearBoundary");
    const int stepsBeforeApplyingTikhonov    = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    int32_t nSamplesForSource      = getRequired<int32_t>(solverConfig, "nSamplesForSource");
    float minWeightForRecordLookup = getOptional<float>(solverConfig, "wmin", 0);
    float lambda                   = getRequired<float>(solverConfig, "lambda");
    int32_t numFourierOrders       = getOptional<int32_t>(solverConfig, "numFourierOrders", 10);

    const bool disableGradientControlVariates             = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates          = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution        = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution       = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution                   = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs                                  = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded                          = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // initialize solver and estimate solution
    // we have two passes.
    ProgressBar                   pb(samplePts.size() * 2);
    std::function<void(int, int)> reportProgress = getReportProgressCallback(pb);

    zombie::WalkSettings        walkSettings(epsilonShellForAbsorbingBoundary,
                                             epsilonShellForReflectingBoundary,
                                             silhouettePrecision,
                                             russianRouletteThreshold,
                                             splittingThreshold, maxWalkLength,
                                             stepsBeforeApplyingTikhonov,
                                             stepsBeforeUsingMaximalSpheres,
                                             solveDoubleSided,
                                             !disableGradientControlVariates,
                                             !disableGradientAntitheticVariates,
                                             useCosineSamplingForDirectionalDerivatives,
                                             ignoreAbsorbingBoundaryContribution,
                                             ignoreReflectingBoundaryContribution,
                                             ignoreSourceContribution, printLogs);
    std::vector<int>            nWalksVector(samplePts.size(), nWalks);
    zombie::WalkOnStars<T, DIM> walkOnStars(queries);
    
    zombie::hc::HarmonicCaching<T, DIM> hc{walkOnStars, bbox.first, bbox.second, minWeightForRecordLookup, nWalksNearBoundary, nSamplesForSource, lambda};
    hc.solve(pde, walkSettings, samplePts, sampleStatistics, runSingleThreaded, reportProgress);

    pb.finish();
}

template <typename T, size_t DIM>
void getSolution(const std::vector<DistanceInfo>& distanceInfo,
                 const std::vector<zombie::SampleStatistics<T, DIM>>& sampleStatistics,
                 std::vector<T>& solution)
{
    solution.resize(distanceInfo.size(), T(0.0f));
    int counter = 0;
    for (int i = 0; i < (int)distanceInfo.size(); i++) {
        if (distanceInfo[i].inValidSolveRegion) {
            solution[i] = sampleStatistics[counter++].getEstimatedSolution();
        }
    }
}

template <typename T, size_t DIM>
void createBvcEvaluationPoints(const std::vector<zombie::Vector<DIM>>& solveLocations,
                               const std::vector<DistanceInfo>& distanceInfo,
                               std::vector<zombie::bvc::EvaluationPoint<T, DIM>>& evalPts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        zombie::Vector<DIM> pt = solveLocations[i];
        zombie::Vector<DIM> normal = zombie::Vector<DIM>::Zero();
        zombie::SampleType sampleType = zombie::SampleType::InDomain;
        float distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
        float distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;

        evalPts.emplace_back(zombie::bvc::EvaluationPoint<T, DIM>(pt, normal, sampleType,
                                                                  distToAbsorbingBoundary,
                                                                  distToReflectingBoundary));
    }
}

template <typename T>
std::shared_ptr<zombie::BoundarySampler<T, 2>> createBoundarySampler(const std::vector<zombie::Vector2>& boundaryPositions,
                                                                     const std::vector<zombie::Vector2i>& boundaryIndices,
                                                                     const zombie::GeometricQueries<2>& queries)
{
    return zombie::createUniformLineSegmentBoundarySampler<T>(boundaryPositions, boundaryIndices,
                                                              queries.insideBoundingDomain);
}

template <typename T>
std::shared_ptr<zombie::BoundarySampler<T, 3>> createBoundarySampler(const std::vector<zombie::Vector3>& boundaryPositions,
                                                                     const std::vector<zombie::Vector3i>& boundaryIndices,
                                                                     const zombie::GeometricQueries<3>& queries)
{
    return zombie::createUniformTriangleBoundarySampler<T>(boundaryPositions, boundaryIndices,
                                                           queries.insideBoundingDomain);
}

template <typename T, size_t DIM>
void runBoundaryValueCaching(const json& solverConfig,
                             const std::vector<zombie::Vector<DIM>>& absorbingBoundaryPositions,
                             const std::vector<zombie::Vectori<DIM>>& absorbingBoundaryIndices,
                             const std::vector<zombie::Vector<DIM>>& reflectingBoundaryPositions,
                             const std::vector<zombie::Vectori<DIM>>& reflectingBoundaryIndices,
                             const zombie::GeometricQueries<DIM>& queries,
                             const zombie::PDE<T, DIM>& pde,
                             bool solveDoubleSided,
                             std::vector<zombie::bvc::EvaluationPoint<T, DIM>>& evalPts)
{
    // load config settings for wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // load config settings for boundary value caching
    const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
    const int nWalksForCachedGradientEstimates = getOptional<int>(solverConfig, "nWalksForCachedGradientEstimates", 640);
    const int absorbingBoundaryCacheSize = getOptional<int>(solverConfig, "absorbingBoundaryCacheSize", 1024);
    const int reflectingBoundaryCacheSize = getOptional<int>(solverConfig, "reflectingBoundaryCacheSize", 1024);
    int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);

    const bool useFiniteDifferencesForBoundaryDerivatives = getOptional<bool>(solverConfig, "useFiniteDifferencesForBoundaryDerivatives", false);

    const float robinCoeffCutoffForNormalDerivative = getOptional<float>(solverConfig, "robinCoeffCutoffForNormalDerivative",
                                                                         std::numeric_limits<float>::max());
    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary",
                                                                      5.0f*epsilonShellForAbsorbingBoundary);
    const float normalOffsetForReflectingBoundary = getOptional<float>(solverConfig, "normalOffsetForReflectingBoundary", 0.0f);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::shared_ptr<zombie::BoundarySampler<T, DIM>> absorbingBoundarySampler =
        createBoundarySampler<T>(absorbingBoundaryPositions, absorbingBoundaryIndices, queries);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);

    std::shared_ptr<zombie::BoundarySampler<T, DIM>> reflectingBoundarySampler =
        createBoundarySampler<T>(reflectingBoundaryPositions, reflectingBoundaryIndices, queries);
    reflectingBoundarySampler->initialize(normalOffsetForReflectingBoundary, solveDoubleSided);

    // initialize domain sampler
    std::function<bool(const zombie::Vector<DIM>&)> insideSolveRegionDomainSampler;
    float solveRegionVolume = 0.0f;
    if (solveDoubleSided) {
        insideSolveRegionDomainSampler = queries.insideBoundingDomain;
        solveRegionVolume = (queries.domainMax - queries.domainMin).prod();

    } else {
        insideSolveRegionDomainSampler = queries.insideDomain;
        solveRegionVolume = std::fabs(queries.computeDomainSignedVolume());
    }

    std::shared_ptr<zombie::DomainSampler<T, DIM>> domainSampler =
        zombie::createUniformDomainSampler<T, DIM>(insideSolveRegionDomainSampler,
                                                   queries.domainMin, queries.domainMax,
                                                   solveRegionVolume);
    if (ignoreSourceContribution) domainCacheSize = 0;

    // solve using boundary value caching
    int totalWork = 2*(absorbingBoundaryCacheSize + reflectingBoundaryCacheSize) + domainCacheSize;
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = getReportProgressCallback(pb);

    zombie::bvc::BoundaryValueCachingSolver<T, DIM> boundaryValueCaching(
        queries, absorbingBoundarySampler, reflectingBoundarySampler, domainSampler);

    // generate boundary and domain samples
    boundaryValueCaching.generateSamples(absorbingBoundaryCacheSize, reflectingBoundaryCacheSize,
                                         domainCacheSize, normalOffsetForAbsorbingBoundary,
                                         normalOffsetForReflectingBoundary, solveDoubleSided);

    // compute sample estimates
    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision,
                                      russianRouletteThreshold,
                                      splittingThreshold, maxWalkLength,
                                      stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres,
                                      solveDoubleSided,
                                      !disableGradientControlVariates,
                                      !disableGradientAntitheticVariates,
                                      useCosineSamplingForDirectionalDerivatives,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    boundaryValueCaching.computeSampleEstimates(pde, walkSettings,
                                                nWalksForCachedSolutionEstimates,
                                                nWalksForCachedGradientEstimates,
                                                robinCoeffCutoffForNormalDerivative,
                                                useFiniteDifferencesForBoundaryDerivatives,
                                                runSingleThreaded, reportProgress);

    // splat boundary sample estimates and domain data to evaluation points
    boundaryValueCaching.splat(pde, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);

    // estimate solution near boundary
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings,
                                                      normalOffsetForAbsorbingBoundary,
                                                      normalOffsetForReflectingBoundary,
                                                      nWalksForCachedSolutionEstimates,
                                                      evalPts, runSingleThreaded);
    pb.finish();
}

template <typename T, size_t DIM>
void getSolution(const std::vector<zombie::bvc::EvaluationPoint<T, DIM>>& evalPts,
                 std::vector<T>& solution)
{
    solution.resize(evalPts.size(), T(0.0f));
    for (int i = 0; i < (int)evalPts.size(); i++) {
        solution[i] = evalPts[i].getEstimatedSolution();
    }
}

template <typename T, size_t DIM>
void createRwsEvaluationPoints(const std::vector<zombie::Vector<DIM>>& solveLocations,
                               const std::vector<DistanceInfo>& distanceInfo,
                               std::vector<zombie::rws::EvaluationPoint<T, DIM>>& evalPts)
{
    for (int i = 0; i < (int)solveLocations.size(); i++) {
        zombie::Vector<DIM> pt = solveLocations[i];
        zombie::Vector<DIM> normal = zombie::Vector<DIM>::Zero();
        zombie::SampleType sampleType = zombie::SampleType::InDomain;
        float distToAbsorbingBoundary = distanceInfo[i].distToAbsorbingBoundary;
        float distToReflectingBoundary = distanceInfo[i].distToReflectingBoundary;

        evalPts.emplace_back(zombie::rws::EvaluationPoint<T, DIM>(pt, normal, sampleType,
                                                                  distToAbsorbingBoundary,
                                                                  distToReflectingBoundary));
    }
}

template <typename T, size_t DIM>
void runReverseWalkOnStars(const json& solverConfig,
                           const std::vector<zombie::Vector<DIM>>& absorbingBoundaryPositions,
                           const std::vector<zombie::Vectori<DIM>>& absorbingBoundaryIndices,
                           const std::vector<zombie::Vector<DIM>>& reflectingBoundaryPositions,
                           const std::vector<zombie::Vectori<DIM>>& reflectingBoundaryIndices,
                           const zombie::GeometricQueries<DIM>& queries,
                           const zombie::PDE<T, DIM>& pde,
                           bool solveDoubleSided,
                           std::vector<zombie::rws::EvaluationPoint<T, DIM>>& evalPts,
                           std::vector<int>& sampleCounts)
{
    // load config settings for reverse wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
    const float splittingThreshold = getOptional<float>(solverConfig, "splittingThreshold", std::numeric_limits<float>::max());

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);

    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // load config settings for reverse walk splatting
    int absorbingBoundarySampleCount = getOptional<int>(solverConfig, "absorbingBoundarySampleCount", 1024);
    int reflectingBoundarySampleCount = getOptional<int>(solverConfig, "reflectingBoundarySampleCount", 1024);
    int domainSampleCount = getOptional<int>(solverConfig, "domainSampleCount", 1024);

    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary",
                                                                      5.0f*epsilonShellForAbsorbingBoundary);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::shared_ptr<zombie::BoundarySampler<T, DIM>> absorbingBoundarySampler =
        createBoundarySampler<T>(absorbingBoundaryPositions, absorbingBoundaryIndices, queries);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);
    if (ignoreAbsorbingBoundaryContribution) absorbingBoundarySampleCount = 0;

    std::shared_ptr<zombie::BoundarySampler<T, DIM>> reflectingBoundarySampler =
        createBoundarySampler<T>(reflectingBoundaryPositions, reflectingBoundaryIndices, queries);
    reflectingBoundarySampler->initialize(0.0f, solveDoubleSided);
    if (ignoreReflectingBoundaryContribution) reflectingBoundarySampleCount = 0;

    // initialize domain sampler
    std::function<bool(const zombie::Vector<DIM>&)> insideSolveRegionDomainSampler;
    float solveRegionVolume = 0.0f;
    if (solveDoubleSided) {
        insideSolveRegionDomainSampler = queries.insideBoundingDomain;
        solveRegionVolume = (queries.domainMax - queries.domainMin).prod();

    } else {
        insideSolveRegionDomainSampler = queries.insideDomain;
        solveRegionVolume = std::fabs(queries.computeDomainSignedVolume());
    }

    std::shared_ptr<zombie::DomainSampler<T, DIM>> domainSampler =
        zombie::createUniformDomainSampler<T, DIM>(insideSolveRegionDomainSampler,
                                                   queries.domainMin, queries.domainMax,
                                                   solveRegionVolume);
    if (ignoreSourceContribution) domainSampleCount = 0;

    // solve using reverse walk on stars
    int totalWork = absorbingBoundarySampleCount + reflectingBoundarySampleCount + domainSampleCount;
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = getReportProgressCallback(pb);

    zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>> reverseWalkOnStars(
                                    queries, absorbingBoundarySampler, reflectingBoundarySampler, domainSampler);

    // generate boundary and domain samples
    reverseWalkOnStars.generateSamples(absorbingBoundarySampleCount, reflectingBoundarySampleCount,
                                       domainSampleCount, normalOffsetForAbsorbingBoundary, solveDoubleSided);

    // splat contributions to evaluation points
    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision,
                                      russianRouletteThreshold,
                                      splittingThreshold, maxWalkLength,
                                      stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres,
                                      solveDoubleSided, false, false, false,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    reverseWalkOnStars.solve(pde, walkSettings, normalOffsetForAbsorbingBoundary,
                             radiusClampForKernels, regularizationForKernels, evalPts,
                             true, runSingleThreaded, reportProgress);
    pb.finish();

    // save samples counts
    sampleCounts.resize(5);
    sampleCounts[0] = reverseWalkOnStars.getAbsorbingBoundarySampleCount(false);
    sampleCounts[1] = reverseWalkOnStars.getAbsorbingBoundarySampleCount(true);
    sampleCounts[2] = reverseWalkOnStars.getReflectingBoundarySampleCount(false);
    sampleCounts[3] = reverseWalkOnStars.getReflectingBoundarySampleCount(true);
    sampleCounts[4] = reverseWalkOnStars.getDomainSampleCount();
}

template <typename T, size_t DIM>
void getSolution(const std::vector<zombie::rws::EvaluationPoint<T, DIM>>& evalPts,
                 const std::vector<int>& sampleCounts,
                 std::vector<T>& solution)
{
    solution.resize(evalPts.size(), T(0.0f));
    int absorbingBoundarySampleCount = sampleCounts[0];
    int absorbingBoundaryNormalAlignedSampleCount = sampleCounts[1];
    int reflectingBoundarySampleCount = sampleCounts[2];
    int reflectingBoundaryNormalAlignedSampleCount = sampleCounts[3];
    int domainSampleCount = sampleCounts[4];

    for (int i = 0; i < (int)evalPts.size(); i++) {
        solution[i] = evalPts[i].getEstimatedSolution(absorbingBoundarySampleCount,
                                                      absorbingBoundaryNormalAlignedSampleCount,
                                                      reflectingBoundarySampleCount,
                                                      reflectingBoundaryNormalAlignedSampleCount,
                                                      domainSampleCount);
    }
}

template <typename T, size_t DIM>
void runSolver(const std::string& solverType, const json& config, 
               const std::pair<Vector2, Vector2>& bbox,
               const std::vector<zombie::Vector<DIM>>& absorbingBoundaryPositions,
               const std::vector<zombie::Vectori<DIM>>& absorbingBoundaryIndices,
               const std::vector<zombie::Vector<DIM>>& reflectingBoundaryPositions,
               const std::vector<zombie::Vectori<DIM>>& reflectingBoundaryIndices,
               const zombie::GeometricQueries<DIM>& queries,
               const zombie::PDE<T, DIM>& pde, bool solveDoubleSided,
               const std::vector<zombie::Vector<DIM>>& solveLocations,
               const std::vector<DistanceInfo>& distanceInfo,
               std::vector<T>& solution)
{
    if (solverType == "wost") {
        // create sample points to estimate solution at
        std::vector<zombie::SamplePoint<T, DIM>> samplePts;
        createSamplePoints<T, DIM>(solveLocations, distanceInfo, samplePts);

        // run walk on stars
        std::vector<zombie::SampleStatistics<T, DIM>> sampleStatistics;
        runWalkOnStars<T, DIM>(config, queries, pde, solveDoubleSided, samplePts, sampleStatistics);

        // extract solution from sample points
        getSolution<T, DIM>(distanceInfo, sampleStatistics, solution);

    } 
    else if (solverType == "hc") {
        // create sample points to estimate solution at
        std::vector<zombie::SamplePoint<T, DIM>> samplePts;
        createSamplePoints<T, DIM>(solveLocations, distanceInfo, samplePts);

        // run walk on stars
        std::vector<zombie::SampleStatistics<T, DIM>> sampleStatistics;
        runHarmonicCaching<T, DIM>(config, queries, bbox, pde, solveDoubleSided, samplePts, sampleStatistics);

        // extract solution from sample points
        getSolution<T, DIM>(distanceInfo, sampleStatistics, solution);
    }
    else if (solverType == "bvc") {
        // create evaluation points to estimate solution at
        std::vector<zombie::bvc::EvaluationPoint<T, DIM>> evalPts;
        createBvcEvaluationPoints<T, DIM>(solveLocations, distanceInfo, evalPts);

        // run boundary value caching
        runBoundaryValueCaching<T, DIM>(config, absorbingBoundaryPositions, absorbingBoundaryIndices,
                                        reflectingBoundaryPositions, reflectingBoundaryIndices,
                                        queries, pde, solveDoubleSided, evalPts);

        // extract solution from evaluation points
        getSolution<T, DIM>(evalPts, solution);

    } else if (solverType == "rws") {
        // ccreate evaluation points to estimate solution at
        std::vector<zombie::rws::EvaluationPoint<T, DIM>> evalPts;
        createRwsEvaluationPoints<T, DIM>(solveLocations, distanceInfo, evalPts);

        // run reverse walk on stars
        std::vector<int> sampleCounts;
        runReverseWalkOnStars<T, DIM>(config, absorbingBoundaryPositions, absorbingBoundaryIndices,
                                      reflectingBoundaryPositions, reflectingBoundaryIndices,
                                      queries, pde, solveDoubleSided, evalPts, sampleCounts);

        // extract solution from evaluation points
        getSolution<T, DIM>(evalPts, sampleCounts, solution);

    } else {
        std::cerr << "Unknown solver type: " << solverType << std::endl;
        exit(EXIT_FAILURE);
    }
}

int main(int argc, const char *argv[])
{
    if (argc != 2) {
        std::cerr << "must provide config filename" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    // load config settings
    json config = json::parse(configFile);
    const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");
    const std::string zombieDirectoryPath = "../../"; // local path to zombie directory

    // initialize the model problem
    ModelProblem modelProblem(modelProblemConfig, zombieDirectoryPath);
    const std::vector<Vector2i>& absorbingBoundaryIndices = modelProblem.getAbsorbingBoundaryIndices();
    const std::vector<Vector2i>& reflectingBoundaryIndices = modelProblem.getReflectingBoundaryIndices();
    const std::pair<Vector2, Vector2>& boundingBox = modelProblem.getBoundingBox();
    const zombie::GeometricQueries<2>& queries = modelProblem.getGeometricQueries();
    bool solveDoubleSided = modelProblem.solveDoubleSided();
    bool solveExterior = modelProblem.solveExterior();

    // create solve locations on a grid for this demo
    std::vector<Vector2> solveLocations;
    std::vector<DistanceInfo> distanceInfo;
    createGridPoints(outputConfig, boundingBox, solveLocations);
    computeDistanceInfo<2>(solveLocations, queries, solveDoubleSided, solveExterior, distanceInfo);

    // solve the model problem
    std::vector<float> solution;
    if (solveExterior) {
        const zombie::KelvinTransform<float, 2>& kelvinTransform = modelProblem.getKelvinTransform();
        const std::vector<Vector2>& invertedAbsorbingBoundaryPositions = modelProblem.getInvertedAbsorbingBoundaryPositions();
        const std::vector<Vector2>& invertedReflectingBoundaryPositions = modelProblem.getInvertedReflectingBoundaryPositions();
        const zombie::PDE<float, 2>& pdeInvertedDomain = modelProblem.getPDEInvertedDomain();
        const zombie::GeometricQueries<2>& queriesInvertedDomain = modelProblem.getGeometricQueriesInvertedDomain();

        // invert the solve locations and update the distance info
        int nSolveLocations = (int)solveLocations.size();
        std::vector<Vector2> invertedSolveLocations(nSolveLocations, Vector2::Zero());
        for (int i = 0; i < nSolveLocations; i++) {
            invertedSolveLocations[i] = kelvinTransform.transformPoint(solveLocations[i]);
        }
        std::vector<DistanceInfo> distanceInfoInvertedDomain;
        computeDistanceInfo<2>(invertedSolveLocations, queriesInvertedDomain,
                               solveDoubleSided, false, distanceInfoInvertedDomain);

        // run the solver on the inverted domain
        runSolver<float, 2>(solverType, solverConfig, boundingBox,
                            invertedAbsorbingBoundaryPositions, absorbingBoundaryIndices,
                            invertedReflectingBoundaryPositions, reflectingBoundaryIndices,
                            queriesInvertedDomain, pdeInvertedDomain, solveDoubleSided,
                            invertedSolveLocations, distanceInfoInvertedDomain, solution);

        // map the solution values back to the exterior domain
        for (int i = 0; i < nSolveLocations; i++) {
            solution[i] = kelvinTransform.transformSolutionEstimate(solution[i], invertedSolveLocations[i]);
        }

    } else {
        const std::vector<Vector2>& absorbingBoundaryPositions = modelProblem.getAbsorbingBoundaryPositions();
        const std::vector<Vector2>& reflectingBoundaryPositions = modelProblem.getReflectingBoundaryPositions();
        const zombie::PDE<float, 2>& pde = modelProblem.getPDE();

        // run the solver on the input domain
        runSolver<float, 2>(solverType, solverConfig,
                            boundingBox,
                            absorbingBoundaryPositions, absorbingBoundaryIndices,
                            reflectingBoundaryPositions, reflectingBoundaryIndices,
                            queries, pde, solveDoubleSided, solveLocations,
                            distanceInfo, solution);
    }

    // save the solution to disk
    saveGridValues(outputConfig, zombieDirectoryPath, distanceInfo, solution);
    return 0;
}

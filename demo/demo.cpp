// This file is the entry point for the 2D demo application demonstrating how to use Zombie.
// It reads a 'scene' description from a JSON file, runs the WalkOnStars or BoundaryValueCaching
// solver, and writes the result to a PMF or PNG file.

#include "scene.h"
#include "grid.h"

using json = nlohmann::json;

void runWalkOnStars(const Scene& scene, const json& solverConfig, const json& outputConfig) {
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int gridRes = getRequired<int>(outputConfig, "gridRes");

    const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
    const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
    const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    const std::pair<Vector2, Vector2>& bbox = scene.bbox;
    const zombie::GeometricQueries<2>& queries = scene.queries;
    const zombie::PDE<float, 2>& pde = scene.pde;
    bool solveDoubleSided = scene.isDoubleSided;

    // setup solution domain and set the estimation quantity to the PDE solution
    std::vector<zombie::SamplePoint<float, 2>> samplePts;
    createSolutionGrid(samplePts, queries, bbox.first, bbox.second, gridRes);

    std::vector<zombie::SampleEstimationData<2>> sampleEstimationData(samplePts.size());
    for (int i = 0; i < samplePts.size(); i++) {
        sampleEstimationData[i].nWalks = nWalks;
        sampleEstimationData[i].estimationQuantity = solveDoubleSided || queries.insideDomain(samplePts[i].pt, true) ?
                                                     zombie::EstimationQuantity::Solution:
                                                     zombie::EstimationQuantity::None;
    }

    // initialize solver and estimate solution
    ProgressBar pb(gridRes*gridRes);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision, russianRouletteThreshold,
                                      maxWalkLength, stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres, solveDoubleSided,
                                      !disableGradientControlVariates,
                                      !disableGradientAntitheticVariates,
                                      useCosineSamplingForDirectionalDerivatives,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    zombie::WalkOnStars<float, 2> walkOnStars(queries);
    walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, runSingleThreaded, reportProgress);
    pb.finish();

    // save to file
    saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);
}

void runBoundaryValueCaching(const Scene& scene, const json& solverConfig, const json& outputConfig) {
    // load config settings for wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int gridRes = getRequired<int>(outputConfig, "gridRes");

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
    const int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);

    const bool useFiniteDifferencesForBoundaryDerivatives = getOptional<bool>(solverConfig, "useFiniteDifferencesForBoundaryDerivatives", false);

    const float robinCoeffCutoffForNormalDerivative = getOptional<float>(solverConfig, "robinCoeffCutoffForNormalDerivative", std::numeric_limits<float>::max());
    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary", 5.0f*epsilonShellForAbsorbingBoundary);
    const float normalOffsetForReflectingBoundary = getOptional<float>(solverConfig, "normalOffsetForReflectingBoundary", 0.0f);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    const std::pair<Vector2, Vector2>& bbox = scene.bbox;
    const zombie::GeometricQueries<2>& queries = scene.queries;
    const zombie::PDE<float, 2>& pde = scene.pde;
    bool solveDoubleSided = scene.isDoubleSided;

    // setup solution domain and initialize evaluation points
    std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
        return !queries.outsideBoundingDomain(x);
    };
    std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
        return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x, true);
    };

    std::vector<zombie::bvc::EvaluationPoint<float, 2>> evalPts;
    createEvaluationGrid<zombie::bvc::EvaluationPoint<float, 2>>(evalPts, queries, bbox.first, bbox.second, gridRes);

    // generate boundary and domain samples
    std::vector<zombie::SamplePoint<float, 2>> absorbingBoundaryCache;
    std::vector<zombie::SamplePoint<float, 2>> absorbingBoundaryCacheNormalAligned;
    std::vector<zombie::SamplePoint<float, 2>> reflectingBoundaryCache;
    std::vector<zombie::SamplePoint<float, 2>> reflectingBoundaryCacheNormalAligned;
    std::vector<zombie::SamplePoint<float, 2>> domainCache;

    zombie::UniformLineSegmentBoundarySampler<float> absorbingBoundarySampler(
        scene.absorbingBoundaryVertices, scene.absorbingBoundarySegments, queries, insideSolveRegionBoundarySampler);
    absorbingBoundarySampler.initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);
    absorbingBoundarySampler.generateSamples(absorbingBoundarySampler.getSampleCount(absorbingBoundaryCacheSize, false),
                                             zombie::SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                             absorbingBoundaryCache, false);
    if (solveDoubleSided) {
        absorbingBoundarySampler.generateSamples(absorbingBoundarySampler.getSampleCount(absorbingBoundaryCacheSize, true),
                                                 zombie::SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                                 absorbingBoundaryCacheNormalAligned, true);
    }

    zombie::UniformLineSegmentBoundarySampler<float> reflectingBoundarySampler(
        scene.reflectingBoundaryVertices, scene.reflectingBoundarySegments, queries, insideSolveRegionBoundarySampler);
    reflectingBoundarySampler.initialize(normalOffsetForReflectingBoundary, solveDoubleSided);
    reflectingBoundarySampler.generateSamples(reflectingBoundarySampler.getSampleCount(reflectingBoundaryCacheSize, false),
                                              zombie::SampleType::OnReflectingBoundary, normalOffsetForReflectingBoundary,
                                              reflectingBoundaryCache, false);
    if (solveDoubleSided) {
        reflectingBoundarySampler.generateSamples(reflectingBoundarySampler.getSampleCount(reflectingBoundaryCacheSize, true),
                                                  zombie::SampleType::OnReflectingBoundary, normalOffsetForReflectingBoundary,
                                                  reflectingBoundaryCacheNormalAligned, true);
    }

    if (!ignoreSourceContribution) {
        float regionVolume = solveDoubleSided ? (bbox.second - bbox.first).prod() :
                                                std::fabs(queries.computeSignedDomainVolume());
        zombie::UniformDomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
                                                             bbox.first, bbox.second, regionVolume);
        domainSampler.generateSamples(domainCacheSize, domainCache);
    }

    // estimate solution on the boundary and set source values in the interior
    int totalWork = absorbingBoundaryCache.size() +
                    absorbingBoundaryCacheNormalAligned.size() +
                    reflectingBoundaryCache.size() +
                    reflectingBoundaryCacheNormalAligned.size() +
                    domainCache.size();
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::WalkOnStars<float, 2> walkOnStars(queries);
    zombie::bvc::BoundaryValueCaching<float, 2> boundaryValueCaching(queries, walkOnStars);
    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision, russianRouletteThreshold,
                                      maxWalkLength, stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres, solveDoubleSided,
                                      !disableGradientControlVariates,
                                      !disableGradientAntitheticVariates,
                                      useCosineSamplingForDirectionalDerivatives,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                                  nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  absorbingBoundaryCache, useFiniteDifferencesForBoundaryDerivatives,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                                  nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  absorbingBoundaryCacheNormalAligned, useFiniteDifferencesForBoundaryDerivatives,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                                  nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  reflectingBoundaryCache, useFiniteDifferencesForBoundaryDerivatives,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.computeBoundaryEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                                  nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                  reflectingBoundaryCacheNormalAligned, useFiniteDifferencesForBoundaryDerivatives,
                                                  runSingleThreaded, reportProgress);
    boundaryValueCaching.setSourceValues(pde, domainCache, runSingleThreaded);

    // splat solution to evaluation points
    boundaryValueCaching.splat(pde, absorbingBoundaryCache, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, absorbingBoundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, reflectingBoundaryCache, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, reflectingBoundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.splat(pde, domainCache, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, true, normalOffsetForAbsorbingBoundary,
                                                      nWalksForCachedSolutionEstimates, evalPts, runSingleThreaded);
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, false, normalOffsetForReflectingBoundary,
                                                      nWalksForCachedSolutionEstimates, evalPts, runSingleThreaded);
    pb.finish();

    // save to file
    saveEvaluationGrid(evalPts, pde, queries, solveDoubleSided, outputConfig);
}

void runReverseWalkSplatter(const Scene& scene, const json& solverConfig, const json& outputConfig) {
    // load config settings for reverse wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "stepsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "stepsBeforeUsingMaximalSpheres", maxWalkLength);
    const int gridRes = getRequired<int>(outputConfig, "gridRes");

    const bool ignoreAbsorbingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreAbsorbingBoundaryContribution", false);
    const bool ignoreReflectingBoundaryContribution = getOptional<bool>(solverConfig, "ignoreReflectingBoundaryContribution", false);
    const bool ignoreSourceContribution = getOptional<bool>(solverConfig, "ignoreSourceContribution", false);
    const bool printLogs = getOptional<bool>(solverConfig, "printLogs", false);
    const bool runSingleThreaded = getOptional<bool>(solverConfig, "runSingleThreaded", false);

    // load config settings for reverse walk splatting
    const int absorbingBoundarySampleCount = getOptional<int>(solverConfig, "absorbingBoundarySampleCount", 1024);
    const int reflectingBoundarySampleCount = getOptional<int>(solverConfig, "reflectingBoundarySampleCount", 1024);
    const int domainSampleCount = getOptional<int>(solverConfig, "domainSampleCount", 1024);

    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary", 5.0f*epsilonShellForAbsorbingBoundary);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    const std::pair<Vector2, Vector2>& bbox = scene.bbox;
    const zombie::GeometricQueries<2>& queries = scene.queries;
    const zombie::PDE<float, 2>& pde = scene.pde;
    bool solveDoubleSided = scene.isDoubleSided;

    // setup solution domain and initialize evaluation points
    std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
        return !queries.outsideBoundingDomain(x);
    };
    std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
        return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x, true);
    };

    std::vector<zombie::rws::EvaluationPoint<float, 2>> evalPts;
    createEvaluationGrid<zombie::rws::EvaluationPoint<float, 2>>(evalPts, queries, bbox.first, bbox.second, gridRes);

    // generate boundary and domain samples
    std::vector<zombie::SamplePoint<float, 2>> absorbingBoundarySamplePts;
    std::vector<zombie::SamplePoint<float, 2>> absorbingBoundaryNormalAlignedSamplePts;
    std::vector<zombie::SamplePoint<float, 2>> reflectingBoundarySamplePts;
    std::vector<zombie::SamplePoint<float, 2>> reflectingBoundaryNormalAlignedSamplePts;
    std::vector<zombie::SamplePoint<float, 2>> domainSamplePts;

    if (!ignoreAbsorbingBoundaryContribution) {
        zombie::UniformLineSegmentBoundarySampler<float> absorbingBoundarySampler(
            scene.absorbingBoundaryVertices, scene.absorbingBoundarySegments, queries, insideSolveRegionBoundarySampler);
        absorbingBoundarySampler.initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);
        absorbingBoundarySampler.generateSamples(absorbingBoundarySampler.getSampleCount(absorbingBoundarySampleCount, false),
                                                 zombie::SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                                 absorbingBoundarySamplePts, false);
        if (solveDoubleSided) {
            absorbingBoundarySampler.generateSamples(absorbingBoundarySampler.getSampleCount(absorbingBoundarySampleCount, true),
                                                     zombie::SampleType::OnAbsorbingBoundary, normalOffsetForAbsorbingBoundary,
                                                     absorbingBoundaryNormalAlignedSamplePts, true);
        }
    }

    if (!ignoreReflectingBoundaryContribution) {
        zombie::UniformLineSegmentBoundarySampler<float> reflectingBoundarySampler(
            scene.reflectingBoundaryVertices, scene.reflectingBoundarySegments, queries, insideSolveRegionBoundarySampler);
        reflectingBoundarySampler.initialize(0.0f, solveDoubleSided);
        reflectingBoundarySampler.generateSamples(reflectingBoundarySampler.getSampleCount(reflectingBoundarySampleCount, false),
                                                  zombie::SampleType::OnReflectingBoundary, 0.0f,
                                                  reflectingBoundarySamplePts, false);
        if (solveDoubleSided) {
            reflectingBoundarySampler.generateSamples(reflectingBoundarySampler.getSampleCount(reflectingBoundarySampleCount, true),
                                                      zombie::SampleType::OnReflectingBoundary, 0.0f,
                                                      reflectingBoundaryNormalAlignedSamplePts, true);
        }
    }

    if (!ignoreSourceContribution) {
        float regionVolume = solveDoubleSided ? (bbox.second - bbox.first).prod() :
                                                std::fabs(queries.computeSignedDomainVolume());
        zombie::UniformDomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
                                                             bbox.first, bbox.second, regionVolume);
        domainSampler.generateSamples(domainSampleCount, domainSamplePts);
    }

    // initialize nearest neigbhbor finder for evaluation points and assign
    // solution value to evaluation points on the absorbing boundary
    std::vector<Vector2> evalPtPositions;
    for (auto& evalPt: evalPts) {
        evalPtPositions.push_back(evalPt.pt);

        if (evalPt.type == zombie::SampleType::OnAbsorbingBoundary) {
            evalPt.totalAbsorbingBoundaryContribution = pde.dirichlet(evalPt.pt, false);
        }
    }
    zombie::NearestNeighborFinder<2> nearestNeighborFinder;
    nearestNeighborFinder.buildAccelerationStructure(evalPtPositions);

    // bind splat contribution callback
    zombie::SplatContributionCallback<float, 2> splatContribution =
        std::bind(&zombie::rws::splatContribution<float, 2, zombie::NearestNeighborFinder<2>>,
        std::placeholders::_1, std::placeholders::_2, std::cref(queries),
        std::cref(nearestNeighborFinder), std::cref(pde), normalOffsetForAbsorbingBoundary,
        radiusClampForKernels, regularizationForKernels, std::ref(evalPts));

    // estimate solution at evaluation points
    int totalWork = absorbingBoundarySamplePts.size() +
                    absorbingBoundaryNormalAlignedSamplePts.size() +
                    reflectingBoundarySamplePts.size() +
                    reflectingBoundaryNormalAlignedSamplePts.size() +
                    domainSamplePts.size();
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision, russianRouletteThreshold,
                                      maxWalkLength, stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres,
                                      solveDoubleSided, false, false, false,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    zombie::ReverseWalkOnStars<float, 2> reverseWalkOnStars(queries, splatContribution);
    reverseWalkOnStars.solve(pde, walkSettings, absorbingBoundarySamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, absorbingBoundaryNormalAlignedSamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, reflectingBoundarySamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, reflectingBoundaryNormalAlignedSamplePts, runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, domainSamplePts, runSingleThreaded, reportProgress);
    pb.finish();

    // save to file
    saveEvaluationGrid(evalPts, absorbingBoundarySamplePts.size(), absorbingBoundaryNormalAlignedSamplePts.size(),
                       reflectingBoundarySamplePts.size(), reflectingBoundaryNormalAlignedSamplePts.size(),
                       domainSamplePts.size(), pde, queries, solveDoubleSided, outputConfig);
};

int main(int argc, const char *argv[]) {
    if (argc != 2) {
        std::cerr << "must provide config filename" << std::endl;
        exit(EXIT_FAILURE);
    }

    std::ifstream configFile(argv[1]);
    if (!configFile.is_open()) {
        std::cerr << "Error opening file: " << argv[1] << std::endl;
        return EXIT_FAILURE;
    }

    json config = json::parse(configFile);
    const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
    const json sceneConfig = getRequired<json>(config, "scene");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");

    Scene scene(sceneConfig);
    if (solverType == "wost") {
        runWalkOnStars(scene, solverConfig, outputConfig);

    } else if (solverType == "bvc") {
        runBoundaryValueCaching(scene, solverConfig, outputConfig);

    } else if (solverType == "rws") {
        runReverseWalkSplatter(scene, solverConfig, outputConfig);

    } else {
        std::cerr << "Unknown solver type: " << solverType << std::endl;
        return EXIT_FAILURE;
    }
}

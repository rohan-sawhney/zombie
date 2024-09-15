// This file is the entry point for the 2D demo application demonstrating how to use Zombie.
// It reads a 'scene' description from a JSON file, runs the WalkOnStars or BoundaryValueCaching
// solver, and writes the result to a PMF or PNG file.

#include <zombie/variance_reduction/boundary_value_caching.h>
#include <zombie/variance_reduction/reverse_walk_splatter.h>
#include <zombie/utils/nearest_neighbor_finder.h>
#include <zombie/utils/progress.h>
#include "grid.h"
#include "scene.h"

using json = nlohmann::json;

void runWalkOnStars(const Scene& scene, const json& solverConfig, const json& outputConfig) {
    // load config settings
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

    const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
    const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
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
        sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt, true) || solveDoubleSided ?
                                                     zombie::EstimationQuantity::Solution:
                                                     zombie::EstimationQuantity::None;
    }

    // initialize solver and estimate solution
    ProgressBar pb(gridRes*gridRes);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::WalkSettings<float> walkSettings(0.0f, epsilonShellForAbsorbingBoundary,
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
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
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
    const int boundaryCacheSize = getOptional<int>(solverConfig, "boundaryCacheSize", 1024);
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
    std::function<bool(const Vector2&)> onReflectingBoundary = [&scene](const Vector2& x) -> bool {
        return scene.onReflectingBoundary(x);
    };

    std::vector<zombie::bvc::EvaluationPoint<float, 2>> evalPts;
    createEvaluationGrid<zombie::bvc::EvaluationPoint<float, 2>>(evalPts, queries, bbox.first, bbox.second, gridRes);

    // generate boundary and domain samples
    std::vector<zombie::SamplePoint<float, 2>> boundaryCache;
    std::vector<zombie::SamplePoint<float, 2>> boundaryCacheNormalAligned;
    std::vector<zombie::SamplePoint<float, 2>> domainCache;

    zombie::BoundarySampler<float, 2> boundarySampler(scene.vertices, scene.segments, queries,
                                                      insideSolveRegionBoundarySampler,
                                                      onReflectingBoundary);
    boundarySampler.initialize(normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary,
                               solveDoubleSided);
    boundarySampler.generateSamples(boundaryCacheSize, normalOffsetForAbsorbingBoundary,
                                    normalOffsetForReflectingBoundary, solveDoubleSided,
                                    0.0f, boundaryCache, boundaryCacheNormalAligned);

    if (!ignoreSourceContribution) {
        zombie::DomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
                                                      bbox.first, bbox.second,
                                                      scene.getSolveRegionVolume());
        domainSampler.generateSamples(pde, domainCacheSize, domainCache);
    }

    // estimate solution on the boundary
    int totalWork = 2*(boundaryCache.size() + boundaryCacheNormalAligned.size()) + domainCache.size();
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::WalkOnStars<float, 2> walkOnStars(queries);
    zombie::bvc::BoundaryValueCaching<float, 2> bvc(queries, walkOnStars);
    zombie::WalkSettings<float> walkSettings(0.0f, epsilonShellForAbsorbingBoundary,
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
    bvc.computeBoundaryEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                 nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                 boundaryCache, useFiniteDifferencesForBoundaryDerivatives,
                                 runSingleThreaded, reportProgress);
    bvc.computeBoundaryEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                 nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                 boundaryCacheNormalAligned, useFiniteDifferencesForBoundaryDerivatives,
                                 runSingleThreaded, reportProgress);

    // splat solution to evaluation points
    bvc.splat(pde, boundaryCache, radiusClampForKernels, regularizationForKernels,
              robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
              normalOffsetForReflectingBoundary, evalPts, reportProgress);
    bvc.splat(pde, boundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
              robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
              normalOffsetForReflectingBoundary, evalPts, reportProgress);
    bvc.splat(pde, domainCache, radiusClampForKernels, regularizationForKernels,
              robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
              normalOffsetForReflectingBoundary, evalPts, reportProgress);
    bvc.estimateSolutionNearBoundary(pde, walkSettings, true, normalOffsetForAbsorbingBoundary,
                                     nWalksForCachedSolutionEstimates, evalPts, runSingleThreaded);
    bvc.estimateSolutionNearBoundary(pde, walkSettings, false, normalOffsetForReflectingBoundary,
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
    const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", 0);
    const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
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
    std::function<bool(const Vector2&)> returnFalse = [](const Vector2& x) -> bool {
        return false;
    };
    std::function<bool(const Vector2&)> returnTrue = [](const Vector2& x) -> bool {
        return true;
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
        zombie::BoundarySampler<float, 2> absorbingBoundarySampler(
            scene.absorbingBoundaryVertices, scene.absorbingBoundarySegments,
            queries, insideSolveRegionBoundarySampler, returnFalse);
        absorbingBoundarySampler.initialize(normalOffsetForAbsorbingBoundary, 0.0f, solveDoubleSided);
        absorbingBoundarySampler.generateSamples(absorbingBoundarySampleCount,
                                                 normalOffsetForAbsorbingBoundary, 0.0f,
                                                 solveDoubleSided, 0.0f, absorbingBoundarySamplePts,
                                                 absorbingBoundaryNormalAlignedSamplePts);
    }

    if (!ignoreReflectingBoundaryContribution) {
        zombie::BoundarySampler<float, 2> reflectingBoundarySampler(
            scene.reflectingBoundaryVertices, scene.reflectingBoundarySegments,
            queries, insideSolveRegionBoundarySampler, returnTrue);
        reflectingBoundarySampler.initialize(0.0f, 0.0f, solveDoubleSided);
        reflectingBoundarySampler.generateSamples(reflectingBoundarySampleCount, 0.0f, 0.0f,
                                                  solveDoubleSided, 0.0f, reflectingBoundarySamplePts,
                                                  reflectingBoundaryNormalAlignedSamplePts);
    }

    if (!ignoreSourceContribution) {
        zombie::DomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
                                                      bbox.first, bbox.second,
                                                      scene.getSolveRegionVolume());
        domainSampler.generateSamples(pde, domainSampleCount, domainSamplePts);
    }

    // initialize nearest neigbhbor finder for evaluation points and assign
    // solution value to evaluation points on the absorbing boundary
    std::vector<Vector2> evalPtPositions;
    for (auto& evalPt: evalPts) {
        evalPtPositions.push_back(evalPt.pt);

        if (evalPt.type == zombie::SampleType::OnAbsorbingBoundary) {
            evalPt.totalAbsorbingBoundaryContribution = pde.dirichlet(evalPt.pt);
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

    zombie::WalkSettings<float> walkSettings(0.0f, epsilonShellForAbsorbingBoundary,
                                             epsilonShellForReflectingBoundary,
                                             silhouettePrecision, russianRouletteThreshold,
                                             maxWalkLength, stepsBeforeApplyingTikhonov,
                                             stepsBeforeUsingMaximalSpheres,
                                             solveDoubleSided, false, false, false,
                                             ignoreAbsorbingBoundaryContribution,
                                             ignoreReflectingBoundaryContribution,
                                             ignoreSourceContribution, printLogs);
    zombie::ReverseWalkOnStars<float, 2> reverseWalkOnStars(queries, splatContribution);
    reverseWalkOnStars.solve(pde, walkSettings, absorbingBoundarySamplePts,
                             runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, absorbingBoundaryNormalAlignedSamplePts,
                             runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, reflectingBoundarySamplePts,
                             runSingleThreaded, reportProgress);
    reverseWalkOnStars.solve(pde, walkSettings, reflectingBoundaryNormalAlignedSamplePts,
                             runSingleThreaded, reportProgress);
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

// This file is the entry point for the 2D demo application demonstrating how to use Zombie.
// It reads a 'model problem' description from a JSON file, runs the WalkOnStars, BoundaryValueCaching
// or ReverseWalkoOnStars solvers, and writes the result to a PMF or PNG file.

#include "model_problem.h"
#include "grid.h"

using json = nlohmann::json;

void runWalkOnStars(const ModelProblem& modelProblem, const json& solverConfig, const json& outputConfig)
{
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

    const zombie::GeometricQueries<2>& queries = modelProblem.queries;
    const zombie::PDE<float, 2>& pde = modelProblem.pde;
    bool solveDoubleSided = modelProblem.solveDoubleSided;

    // setup solution domain
    std::vector<zombie::SamplePoint<float, 2>> samplePts;
    createSolutionGrid(samplePts, queries, solveDoubleSided, gridRes);

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
    std::vector<int> nWalksVector(samplePts.size(), nWalks);
    zombie::WalkOnStars<float, 2> walkOnStars(queries);
    walkOnStars.solve(pde, walkSettings, nWalksVector, samplePts, runSingleThreaded, reportProgress);
    pb.finish();

    // save to file
    saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);
}

void runBoundaryValueCaching(const ModelProblem& modelProblem, const json& solverConfig, const json& outputConfig)
{
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

    const zombie::GeometricQueries<2>& queries = modelProblem.queries;
    const zombie::PDE<float, 2>& pde = modelProblem.pde;
    bool solveDoubleSided = modelProblem.solveDoubleSided;

    // initialize evaluation points
    std::vector<zombie::bvc::EvaluationPoint<float, 2>> evalPts;
    createEvaluationGrid<zombie::bvc::EvaluationPoint<float, 2>>(evalPts, queries, gridRes);

    // initialize boundary and domain samplers
    std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
        return !queries.outsideBoundingDomain(x);
    };

    std::unique_ptr<zombie::BoundarySampler<float, 2>> absorbingBoundarySampler =
        zombie::createUniformLineSegmentBoundarySampler<float>(
            modelProblem.absorbingBoundaryVertices, modelProblem.absorbingBoundarySegments,
            queries, insideSolveRegionBoundarySampler);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);

    std::unique_ptr<zombie::BoundarySampler<float, 2>> reflectingBoundarySampler =
        zombie::createUniformLineSegmentBoundarySampler<float>(
            modelProblem.reflectingBoundaryVertices, modelProblem.reflectingBoundarySegments,
            queries, insideSolveRegionBoundarySampler);
    reflectingBoundarySampler->initialize(normalOffsetForReflectingBoundary, solveDoubleSided);

    std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = {};
    std::unique_ptr<zombie::DomainSampler<float, 2>> domainSampler = nullptr;
    if (!ignoreSourceContribution) {
        insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
            return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x, true);
        };

        float regionVolume = solveDoubleSided ? (queries.domainMax - queries.domainMin).prod() :
                                                std::fabs(queries.computeDomainSignedVolume());
        domainSampler = zombie::createUniformDomainSampler<float, 2>(
            queries, insideSolveRegionDomainSampler, queries.domainMin, queries.domainMax, regionVolume);
    }

    // solve using boundary value caching
    int totalWork = 2.0f*(absorbingBoundaryCacheSize + reflectingBoundaryCacheSize) + domainCacheSize;
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::bvc::BoundaryValueCachingSolver<float, 2> boundaryValueCaching(
        queries, absorbingBoundarySampler, reflectingBoundarySampler, domainSampler);

    // generate boundary and domain samples
    boundaryValueCaching.generateSamples(absorbingBoundaryCacheSize, reflectingBoundaryCacheSize,
                                         domainCacheSize, normalOffsetForAbsorbingBoundary,
                                         normalOffsetForReflectingBoundary, solveDoubleSided);

    // compute sample estimates
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
    boundaryValueCaching.computeSampleEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
                                                nWalksForCachedGradientEstimates, robinCoeffCutoffForNormalDerivative,
                                                useFiniteDifferencesForBoundaryDerivatives, runSingleThreaded, reportProgress);

    // splat boundary sample estimates and domain data to evaluation points
    boundaryValueCaching.splat(pde, radiusClampForKernels, regularizationForKernels,
                               robinCoeffCutoffForNormalDerivative, normalOffsetForAbsorbingBoundary,
                               normalOffsetForReflectingBoundary, evalPts, reportProgress);

    // estimate solution near boundary
    boundaryValueCaching.estimateSolutionNearBoundary(pde, walkSettings, normalOffsetForAbsorbingBoundary,
                                                      normalOffsetForReflectingBoundary, nWalksForCachedSolutionEstimates,
                                                      evalPts, runSingleThreaded);
    pb.finish();

    // save to file
    saveEvaluationGrid(evalPts, pde, queries, solveDoubleSided, outputConfig);
}

void runReverseWalkOnStars(const ModelProblem& modelProblem, const json& solverConfig, const json& outputConfig)
{
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

    const zombie::GeometricQueries<2>& queries = modelProblem.queries;
    const zombie::PDE<float, 2>& pde = modelProblem.pde;
    bool solveDoubleSided = modelProblem.solveDoubleSided;

    // initialize evaluation points
    std::vector<zombie::rws::EvaluationPoint<float, 2>> evalPts;
    createEvaluationGrid<zombie::rws::EvaluationPoint<float, 2>>(evalPts, queries, gridRes);

    // initialize boundary and domain samplers
    std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = {};
    if (!ignoreAbsorbingBoundaryContribution || !ignoreReflectingBoundaryContribution) {
        insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
            return !queries.outsideBoundingDomain(x);
        };
    }

    std::unique_ptr<zombie::BoundarySampler<float, 2>> absorbingBoundarySampler = nullptr;
    if (!ignoreAbsorbingBoundaryContribution) {
        absorbingBoundarySampler = zombie::createUniformLineSegmentBoundarySampler<float>(
            modelProblem.absorbingBoundaryVertices, modelProblem.absorbingBoundarySegments,
            queries, insideSolveRegionBoundarySampler);
        absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);
    }

    std::unique_ptr<zombie::BoundarySampler<float, 2>> reflectingBoundarySampler = nullptr;
    if (!ignoreReflectingBoundaryContribution) {
        reflectingBoundarySampler = zombie::createUniformLineSegmentBoundarySampler<float>(
            modelProblem.reflectingBoundaryVertices, modelProblem.reflectingBoundarySegments,
            queries, insideSolveRegionBoundarySampler);
        reflectingBoundarySampler->initialize(0.0f, solveDoubleSided);
    }

    std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = {};
    std::unique_ptr<zombie::DomainSampler<float, 2>> domainSampler = nullptr;
    if (!ignoreSourceContribution) {
        insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
            return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x, true);
        };

        float regionVolume = solveDoubleSided ? (queries.domainMax - queries.domainMin).prod() :
                                                std::fabs(queries.computeDomainSignedVolume());
        domainSampler = zombie::createUniformDomainSampler<float, 2>(
            queries, insideSolveRegionDomainSampler, queries.domainMin, queries.domainMax, regionVolume);
    }

    // solve using reverse walk on stars
    int totalWork = absorbingBoundarySampleCount + reflectingBoundarySampleCount + domainSampleCount;
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

    zombie::rws::ReverseWalkOnStarsSolver<float, 2, zombie::NearestNeighborFinder<2>> reverseWalkOnStars(
        pde, queries, absorbingBoundarySampler, reflectingBoundarySampler, domainSampler,
        normalOffsetForAbsorbingBoundary, radiusClampForKernels, regularizationForKernels, evalPts);

    // generate boundary and domain samples
    reverseWalkOnStars.generateSamples(absorbingBoundarySampleCount, reflectingBoundarySampleCount,
                                       domainSampleCount, solveDoubleSided);

    // splat contributions to evaluation points
    zombie::WalkSettings walkSettings(epsilonShellForAbsorbingBoundary,
                                      epsilonShellForReflectingBoundary,
                                      silhouettePrecision, russianRouletteThreshold,
                                      maxWalkLength, stepsBeforeApplyingTikhonov,
                                      stepsBeforeUsingMaximalSpheres,
                                      solveDoubleSided, false, false, false,
                                      ignoreAbsorbingBoundaryContribution,
                                      ignoreReflectingBoundaryContribution,
                                      ignoreSourceContribution, printLogs);
    reverseWalkOnStars.solve(walkSettings, runSingleThreaded, reportProgress);
    pb.finish();

    // save to file
    saveEvaluationGrid(evalPts, reverseWalkOnStars.getAbsorbingBoundarySamplePts(false).size(),
                       reverseWalkOnStars.getAbsorbingBoundarySamplePts(true).size(),
                       reverseWalkOnStars.getReflectingBoundarySamplePts(false).size(),
                       reverseWalkOnStars.getReflectingBoundarySamplePts(true).size(),
                       reverseWalkOnStars.getDomainSamplePts().size(), pde, queries,
                       solveDoubleSided, outputConfig);
};

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

    json config = json::parse(configFile);
    const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");

    ModelProblem modelProblem(modelProblemConfig);
    if (solverType == "wost") {
        runWalkOnStars(modelProblem, solverConfig, outputConfig);

    } else if (solverType == "bvc") {
        runBoundaryValueCaching(modelProblem, solverConfig, outputConfig);

    } else if (solverType == "rws") {
        runReverseWalkOnStars(modelProblem, solverConfig, outputConfig);

    } else {
        std::cerr << "Unknown solver type: " << solverType << std::endl;
        return EXIT_FAILURE;
    }
}

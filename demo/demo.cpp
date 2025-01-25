// This file is the entry point for a 2D demo application demonstrating how to use Zombie.
// It reads a 'model problem' description from a JSON file, runs the WalkOnStars, BoundaryValueCaching
// or ReverseWalkonOnStars solvers, and writes the result to a PMF or PNG file.

#include "model_problem.h"
#include "grid.h"

using json = nlohmann::json;

void runWalkOnStars(const json& solverConfig,
                    const zombie::GeometricQueries<2>& queries,
                    const zombie::PDE<float, 2>& pde,
                    bool solveDoubleSided,
                    std::vector<zombie::SamplePoint<float, 2>>& samplePts)
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
}

void runBoundaryValueCaching(const json& solverConfig,
                             const std::vector<Vector2>& absorbingBoundaryVertices,
                             const std::vector<Vector2i>& absorbingBoundarySegments,
                             const std::vector<Vector2>& reflectingBoundaryVertices,
                             const std::vector<Vector2i>& reflectingBoundarySegments,
                             const zombie::GeometricQueries<2>& queries,
                             const zombie::PDE<float, 2>& pde,
                             bool solveDoubleSided,
                             std::vector<zombie::bvc::EvaluationPoint<float, 2>>& evalPts)
{
    // load config settings for wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

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

    const float robinCoeffCutoffForNormalDerivative = getOptional<float>(solverConfig, "robinCoeffCutoffForNormalDerivative", std::numeric_limits<float>::max());
    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary", 5.0f*epsilonShellForAbsorbingBoundary);
    const float normalOffsetForReflectingBoundary = getOptional<float>(solverConfig, "normalOffsetForReflectingBoundary", 0.0f);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::shared_ptr<zombie::BoundarySampler<float, 2>> absorbingBoundarySampler =
        zombie::createUniformLineSegmentBoundarySampler<float>(
            absorbingBoundaryVertices, absorbingBoundarySegments,
            queries, queries.insideBoundingDomain);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);

    std::shared_ptr<zombie::BoundarySampler<float, 2>> reflectingBoundarySampler =
        zombie::createUniformLineSegmentBoundarySampler<float>(
            reflectingBoundaryVertices, reflectingBoundarySegments,
            queries, queries.insideBoundingDomain);
    reflectingBoundarySampler->initialize(normalOffsetForReflectingBoundary, solveDoubleSided);

    // initialize domain sampler
    std::function<bool(const Vector2&)> insideSolveRegionDomainSampler;
    float solveRegionVolume = 0.0f;
    if (solveDoubleSided) {
        insideSolveRegionDomainSampler = queries.insideBoundingDomain;
        solveRegionVolume = (queries.domainMax - queries.domainMin).prod();

    } else {
        insideSolveRegionDomainSampler = queries.insideDomain;
        solveRegionVolume = std::fabs(queries.computeDomainSignedVolume());
    }

    std::shared_ptr<zombie::DomainSampler<float, 2>> domainSampler =
        zombie::createUniformDomainSampler<float, 2>(queries, insideSolveRegionDomainSampler,
                                                     queries.domainMin, queries.domainMax,
                                                     solveRegionVolume);
    if (ignoreSourceContribution) domainCacheSize = 0;

    // solve using boundary value caching
    int totalWork = 2.0f*(absorbingBoundaryCacheSize + reflectingBoundaryCacheSize) + domainCacheSize;
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = getReportProgressCallback(pb);

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
}

void runReverseWalkOnStars(const json& solverConfig,
                           const std::vector<Vector2>& absorbingBoundaryVertices,
                           const std::vector<Vector2i>& absorbingBoundarySegments,
                           const std::vector<Vector2>& reflectingBoundaryVertices,
                           const std::vector<Vector2i>& reflectingBoundarySegments,
                           const zombie::GeometricQueries<2>& queries,
                           const zombie::PDE<float, 2>& pde,
                           bool solveDoubleSided,
                           std::vector<zombie::rws::EvaluationPoint<float, 2>>& evalPts,
                           std::vector<int>& sampleCounts)
{
    // load config settings for reverse wost
    const float epsilonShellForAbsorbingBoundary = getOptional<float>(solverConfig, "epsilonShellForAbsorbingBoundary", 1e-3f);
    const float epsilonShellForReflectingBoundary = getOptional<float>(solverConfig, "epsilonShellForReflectingBoundary", 1e-3f);
    const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
    const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

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

    const float normalOffsetForAbsorbingBoundary = getOptional<float>(solverConfig, "normalOffsetForAbsorbingBoundary", 5.0f*epsilonShellForAbsorbingBoundary);
    const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 0.0f);
    const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

    // initialize boundary samplers
    std::shared_ptr<zombie::BoundarySampler<float, 2>> absorbingBoundarySampler =
        zombie::createUniformLineSegmentBoundarySampler<float>(
            absorbingBoundaryVertices, absorbingBoundarySegments,
            queries, queries.insideBoundingDomain);
    absorbingBoundarySampler->initialize(normalOffsetForAbsorbingBoundary, solveDoubleSided);
    if (ignoreAbsorbingBoundaryContribution) absorbingBoundarySampleCount = 0;

    std::shared_ptr<zombie::BoundarySampler<float, 2>> reflectingBoundarySampler =
        zombie::createUniformLineSegmentBoundarySampler<float>(
            reflectingBoundaryVertices, reflectingBoundarySegments,
            queries, queries.insideBoundingDomain);
    reflectingBoundarySampler->initialize(0.0f, solveDoubleSided);
    if (ignoreReflectingBoundaryContribution) reflectingBoundarySampleCount = 0;

    // initialize domain sampler
    std::function<bool(const Vector2&)> insideSolveRegionDomainSampler;
    float solveRegionVolume = 0.0f;
    if (solveDoubleSided) {
        insideSolveRegionDomainSampler = queries.insideBoundingDomain;
        solveRegionVolume = (queries.domainMax - queries.domainMin).prod();

    } else {
        insideSolveRegionDomainSampler = queries.insideDomain;
        solveRegionVolume = std::fabs(queries.computeDomainSignedVolume());
    }

    std::shared_ptr<zombie::DomainSampler<float, 2>> domainSampler =
        zombie::createUniformDomainSampler<float, 2>(queries, insideSolveRegionDomainSampler,
                                                     queries.domainMin, queries.domainMax,
                                                     solveRegionVolume);
    if (ignoreSourceContribution) domainSampleCount = 0;

    // solve using reverse walk on stars
    int totalWork = absorbingBoundarySampleCount + reflectingBoundarySampleCount + domainSampleCount;
    ProgressBar pb(totalWork);
    std::function<void(int, int)> reportProgress = getReportProgressCallback(pb);

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

    // save samples counts
    sampleCounts.resize(5);
    sampleCounts[0] = reverseWalkOnStars.getAbsorbingBoundarySampleCount(false);
    sampleCounts[1] = reverseWalkOnStars.getAbsorbingBoundarySampleCount(true);
    sampleCounts[2] = reverseWalkOnStars.getReflectingBoundarySampleCount(false);
    sampleCounts[3] = reverseWalkOnStars.getReflectingBoundarySampleCount(true);
    sampleCounts[4] = reverseWalkOnStars.getDomainSampleCount();
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

    // load config settings
    json config = json::parse(configFile);
    const std::string solverType = getOptional<std::string>(config, "solverType", "wost");
    const json modelProblemConfig = getRequired<json>(config, "modelProblem");
    const json solverConfig = getRequired<json>(config, "solver");
    const json outputConfig = getRequired<json>(config, "output");
    const int gridRes = getRequired<int>(outputConfig, "gridRes");

    // initialize model problem
    ModelProblem modelProblem(modelProblemConfig);
    const std::vector<Vector2>& absorbingBoundaryVertices = modelProblem.absorbingBoundaryVertices;
    const std::vector<Vector2i>& absorbingBoundarySegments = modelProblem.absorbingBoundarySegments;
    const std::vector<Vector2>& reflectingBoundaryVertices = modelProblem.reflectingBoundaryVertices;
    const std::vector<Vector2i>& reflectingBoundarySegments = modelProblem.reflectingBoundarySegments;
    const zombie::GeometricQueries<2>& queries = modelProblem.queries;
    const zombie::PDE<float, 2>& pde = modelProblem.pde;
    bool solveDoubleSided = modelProblem.solveDoubleSided;

    if (solverType == "wost") {
        // create sample points on grid to compute solution on
        std::vector<zombie::SamplePoint<float, 2>> samplePts;
        createSolutionGrid(samplePts, queries, solveDoubleSided, gridRes);

        // run walk on stars
        runWalkOnStars(solverConfig, queries, pde, solveDoubleSided, samplePts);

        // save solution to disk
        saveSolutionGrid(samplePts, queries, pde, solveDoubleSided, outputConfig);

    } else if (solverType == "bvc") {
        // create evaluation points on grid to compute solution on
        std::vector<zombie::bvc::EvaluationPoint<float, 2>> evalPts;
        createEvaluationGrid<zombie::bvc::EvaluationPoint<float, 2>>(evalPts, queries, gridRes);

        // run boundary value caching
        runBoundaryValueCaching(solverConfig, absorbingBoundaryVertices, absorbingBoundarySegments,
                                reflectingBoundaryVertices, reflectingBoundarySegments,
                                queries, pde, solveDoubleSided, evalPts);

        // save solution to disk
        saveEvaluationGrid(evalPts, queries, pde, solveDoubleSided, outputConfig);

    } else if (solverType == "rws") {
        // create evaluation points on grid to compute solution on
        std::vector<zombie::rws::EvaluationPoint<float, 2>> evalPts;
        createEvaluationGrid<zombie::rws::EvaluationPoint<float, 2>>(evalPts, queries, gridRes);

        // run reverse walk on stars
        std::vector<int> sampleCounts;
        runReverseWalkOnStars(solverConfig, absorbingBoundaryVertices, absorbingBoundarySegments,
                              reflectingBoundaryVertices, reflectingBoundarySegments,
                              queries, pde, solveDoubleSided, evalPts, sampleCounts);

        // save solution to disk
        saveEvaluationGrid(evalPts, sampleCounts, queries, pde, solveDoubleSided, outputConfig);

    } else {
        std::cerr << "Unknown solver type: " << solverType << std::endl;
        return EXIT_FAILURE;
    }
}

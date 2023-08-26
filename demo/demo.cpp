#include <zombie/point_estimation/walk_on_stars.h>
#include <zombie/boundary_value_caching/splatter.h>
#include <zombie/utils/progress.h>
#include "grid.h"
#include "scene.h"

using json = nlohmann::json;

void runWalkOnStars(const Scene& scene, const json& solverConfig, const json& outputConfig) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);

	const int nWalks = getOptional<int>(solverConfig, "nWalks", 128);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);

	fcpw::BoundingBox<2> bbox = scene.bbox;
	const zombie::GeometricQueries<2>& queries = scene.queries;
	const zombie::PDE<float, 2>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;

	// setup solution domain
	std::vector<zombie::SamplePoint<float, 2>> samplePts;
	createSolutionGrid(samplePts, queries, bbox.pMin, bbox.pMax, gridRes);

	std::vector<zombie::SampleEstimationData<2>> sampleEstimationData(samplePts.size());
	for (int i = 0; i < samplePts.size(); i++) {
		sampleEstimationData[i].nWalks = nWalks;
		sampleEstimationData[i].estimationQuantity = queries.insideDomain(samplePts[i].pt) || solveDoubleSided ?
													 zombie::EstimationQuantity::Solution:
													 zombie::EstimationQuantity::None;
	}

	// initialize solver and estimate solution
	ProgressBar pb(gridRes*gridRes);
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);

	zombie::WalkOnStars<float, 2> walkOnStars(queries);
	walkOnStars.solve(pde, walkSettings, sampleEstimationData, samplePts, false, reportProgress);
	pb.finish();

	// save to file
	saveSolutionGrid(samplePts, pde, queries, solveDoubleSided, outputConfig);
}

void runBoundaryValueCaching(const Scene& scene, const json& solverConfig, const json& outputConfig) {
	// load configuration settings
	const bool disableGradientControlVariates = getOptional<bool>(solverConfig, "disableGradientControlVariates", false);
	const bool disableGradientAntitheticVariates = getOptional<bool>(solverConfig, "disableGradientAntitheticVariates", false);
	const bool useCosineSamplingForDirectionalDerivatives = getOptional<bool>(solverConfig, "useCosineSamplingForDirectionalDerivatives", false);
	const bool useFiniteDifferencesForBoundaryDerivatives = getOptional<bool>(solverConfig, "useFiniteDifferencesForBoundaryDerivatives", false);
	const bool ignoreDirichlet = getOptional<bool>(solverConfig, "ignoreDirichlet", false);
	const bool ignoreNeumann = getOptional<bool>(solverConfig, "ignoreNeumann", false);
	const bool ignoreSource = getOptional<bool>(solverConfig, "ignoreSource", false);

	const int nWalksForCachedSolutionEstimates = getOptional<int>(solverConfig, "nWalksForCachedSolutionEstimates", 128);
	const int nWalksForCachedGradientEstimates = getOptional<int>(solverConfig, "nWalksForCachedGradientEstimates", 640);
	const int maxWalkLength = getOptional<int>(solverConfig, "maxWalkLength", 1024);
	const int stepsBeforeApplyingTikhonov = getOptional<int>(solverConfig, "setpsBeforeApplyingTikhonov", maxWalkLength);
	const int stepsBeforeUsingMaximalSpheres = getOptional<int>(solverConfig, "setpsBeforeUsingMaximalSpheres", maxWalkLength);
	const int boundaryCacheSize = getOptional<int>(solverConfig, "boundaryCacheSize", 1024);
	const int domainCacheSize = getOptional<int>(solverConfig, "domainCacheSize", 1024);
	const int gridRes = getRequired<int>(outputConfig, "gridRes");

	const float epsilonShell = getOptional<float>(solverConfig, "epsilonShell", 1e-3f);
	const float minStarRadius = getOptional<float>(solverConfig, "minStarRadius", 1e-3f);
	const float silhouettePrecision = getOptional<float>(solverConfig, "silhouettePrecision", 1e-3f);
	const float russianRouletteThreshold = getOptional<float>(solverConfig, "russianRouletteThreshold", 0.0f);
	const float normalOffsetForCachedDirichletSamples = getOptional<float>(solverConfig, "normalOffsetForCachedDirichletSamples", 5.0f*epsilonShell);
	const float radiusClampForKernels = getOptional<float>(solverConfig, "radiusClampForKernels", 1e-3f);
	const float regularizationForKernels = getOptional<float>(solverConfig, "regularizationForKernels", 0.0f);

	fcpw::BoundingBox<2> bbox = scene.bbox;
	const zombie::GeometricQueries<2>& queries = scene.queries;
	const zombie::PDE<float, 2>& pde = scene.pde;
	bool solveDoubleSided = scene.isDoubleSided;

	// setup solution domain
	std::function<bool(const Vector2&)> insideSolveRegionBoundarySampler = [&queries](const Vector2& x) -> bool {
		return !queries.outsideBoundingDomain(x);
	};
	std::function<bool(const Vector2&)> insideSolveRegionDomainSampler = [&queries, solveDoubleSided](const Vector2& x) -> bool {
		return solveDoubleSided ? !queries.outsideBoundingDomain(x) : queries.insideDomain(x);
	};
	std::function<bool(const Vector2&)> onNeumannBoundary = [&scene](const Vector2 &x) -> bool {
		return scene.onNeumannBoundary(x);
	};

	std::vector<zombie::SamplePoint<float, 2>> boundaryCache;
	std::vector<zombie::SamplePoint<float, 2>> boundaryCacheNormalAligned;
	std::vector<zombie::SamplePoint<float, 2>> domainCache;
	std::vector<zombie::EvaluationPoint<float, 2>> evalPts;
	createEvaluationGrid(evalPts, queries, bbox.pMin, bbox.pMax, gridRes);

	// initialize solver and generate samples
	zombie::WalkOnStars<float, 2> walkOnStars(queries);
	zombie::BoundarySampler<float, 2> boundarySampler(scene.vertices, scene.segments, queries,
													  walkOnStars, insideSolveRegionBoundarySampler,
													  onNeumannBoundary);
	zombie::DomainSampler<float, 2> domainSampler(queries, insideSolveRegionDomainSampler,
												  bbox.pMin, bbox.pMax, scene.getSolveRegionVolume());

	boundarySampler.initialize(normalOffsetForCachedDirichletSamples, solveDoubleSided);
	boundarySampler.generateSamples(boundaryCacheSize, normalOffsetForCachedDirichletSamples,
									solveDoubleSided, 0.0f, boundaryCache, boundaryCacheNormalAligned);
	if (!ignoreSource) domainSampler.generateSamples(pde, domainCacheSize, domainCache);

	// estimate solution on the boundary
	int totalWork = 2.0*(boundaryCache.size() + boundaryCacheNormalAligned.size()) + domainCacheSize;
	ProgressBar pb(totalWork);
	std::function<void(int, int)> reportProgress = [&pb](int i, int tid) -> void { pb.report(i, tid); };

	zombie::WalkSettings<float> walkSettings(0.0f, epsilonShell, minStarRadius,
											 silhouettePrecision, russianRouletteThreshold,
											 maxWalkLength, stepsBeforeApplyingTikhonov,
											 stepsBeforeUsingMaximalSpheres, solveDoubleSided,
											 !disableGradientControlVariates,
											 !disableGradientAntitheticVariates,
											 useCosineSamplingForDirectionalDerivatives,
											 ignoreDirichlet, ignoreNeumann, ignoreSource, false);
	boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
									 nWalksForCachedGradientEstimates, boundaryCache,
									 useFiniteDifferencesForBoundaryDerivatives,
									 false, reportProgress);
	boundarySampler.computeEstimates(pde, walkSettings, nWalksForCachedSolutionEstimates,
									 nWalksForCachedGradientEstimates, boundaryCacheNormalAligned,
									 useFiniteDifferencesForBoundaryDerivatives,
									 false, reportProgress);

	// splat solution to evaluation points
	zombie::Splatter<float, 2> splatter(queries, walkOnStars);
	splatter.splat(pde, boundaryCache, radiusClampForKernels, regularizationForKernels,
				   normalOffsetForCachedDirichletSamples, evalPts, reportProgress);
	splatter.splat(pde, boundaryCacheNormalAligned, radiusClampForKernels, regularizationForKernels,
				   normalOffsetForCachedDirichletSamples, evalPts, reportProgress);
	splatter.splat(pde, domainCache, radiusClampForKernels, regularizationForKernels,
				   normalOffsetForCachedDirichletSamples, evalPts, reportProgress);
	splatter.estimatePointwiseNearDirichletBoundary(pde, walkSettings, normalOffsetForCachedDirichletSamples,
													nWalksForCachedSolutionEstimates, evalPts, false);
	pb.finish();

	// save to file
	saveEvaluationGrid(evalPts, pde, queries, scene.isDoubleSided, outputConfig);
}

int main(int argc, const char *argv[]) {
	if (argc != 2) {
		std::cerr << "must provide config filename" << std::endl;
		abort();
	}

	std::ifstream configFile(argv[1]);
	if (!configFile.is_open()) {
		std::cerr << "Error opening file: " << argv[1] << std::endl;
		return 1;
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
	}
}

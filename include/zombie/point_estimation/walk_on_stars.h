#pragma once

#include <zombie/core/pde.h>
#include <zombie/core/geometric_queries.h>
#include <zombie/core/distributions.h>
#include "tbb/parallel_for.h"
#include "tbb/blocked_range.h"

#define RADIUS_SHRINK_PERCENTAGE 0.99f

namespace zombie {

enum class EstimationQuantity {
	Solution,
	SolutionAndGradient,
	None
};

enum class SampleType {
	InDomain, // applies to both interior and exterior sample points for closed domains
	OnDirichletBoundary,
	OnNeumannBoundary
};

enum class WalkCompletionCode {
	ReachedDirichletBoundary,
	TerminatedWithRussianRoulette,
	ExceededMaxWalkLength,
	EscapedDomain
};

template <typename T, int DIM>
struct SamplePoint;

template <int DIM>
struct SampleEstimationData;

template <typename T>
struct WalkSettings;

// NOTE: For data with multiple channels (e.g., 2D or 3D positions, rgb etc.), use
// Eigen::Array (in place of Eigen::VectorXf) as it supports component wise operations
template <typename T, int DIM>
class SampleStatistics;

template <typename T, int DIM>
struct WalkState;

template <typename T, int DIM>
class WalkOnStars {
public:
	// constructor
	WalkOnStars(const GeometricQueries<DIM>& queries_,
				std::function<void(WalkState<T, DIM>&)> getTerminalContribution_={}):
				queries(queries_), getTerminalContribution(getTerminalContribution_) {}

	// solves the given PDE at the input point; NOTE: assumes the point does not
	// lie on the boundary when estimating the gradient
	void solve(const PDE<T, DIM>& pde,
			   const WalkSettings<T>& walkSettings,
			   const SampleEstimationData<DIM>& estimationData,
			   SamplePoint<T, DIM>& samplePt) const {
		if (estimationData.estimationQuantity != EstimationQuantity::None) {
			if (estimationData.estimationQuantity == EstimationQuantity::SolutionAndGradient) {
				estimateSolutionAndGradient(pde, walkSettings, estimationData.directionForDerivative,
											estimationData.nWalks, samplePt);

			} else {
				estimateSolution(pde, walkSettings, estimationData.nWalks, samplePt);
			}
		}
	}

	// solves the given PDE at the input points (in parallel by default); NOTE:
	// assumes points do not lie on the boundary when estimating gradients
	void solve(const PDE<T, DIM>& pde,
			   const WalkSettings<T>& walkSettings,
			   const std::vector<SampleEstimationData<DIM>>& estimationData,
			   std::vector<SamplePoint<T, DIM>>& samplePts,
			   bool runSingleThreaded=false,
			   std::function<void(int, int)> reportProgress={}) const {
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

private:
	// performs a single reflecting random walk starting at the input point
	WalkCompletionCode walk(const PDE<T, DIM>& pde,
							const WalkSettings<T>& walkSettings,
							float dirichletDist, float firstSphereRadius,
							bool flipNormalOrientation, pcg32& sampler,
							std::unique_ptr<GreensFnBall<DIM>>& greensFn,
							WalkState<T, DIM>& state) const {
		// recursively perform a random walk till it reaches the Dirichlet boundary
		bool firstStep = true;
		float randNumsForNeumannSampling[DIM];

		while (dirichletDist > walkSettings.epsilonShell) {
			// compute the star radius
			float starRadius;
			if (firstStep && firstSphereRadius > 0.0f) {
				starRadius = firstSphereRadius;

			} else {
				// for problems with double-sided boundary conditions, flip the current
				// normal orientation if the geometry is front-facing
				flipNormalOrientation = false;
				if (walkSettings.solveDoubleSided && state.onNeumannBoundary) {
					if (state.prevDistance > 0.0f && state.prevDirection.dot(state.currentNormal) < 0.0f) {
						state.currentNormal *= -1.0f;
						flipNormalOrientation = true;
					}
				}

				if (walkSettings.stepsBeforeUsingMaximalSpheres <= state.walkLength) {
					starRadius = dirichletDist;

				} else {
					// NOTE: using dirichletDist as the maximum radius for the closest silhouette
					// query can result in a smaller than maximal star-shaped region: should ideally
					// use the distance to the closest visible Dirichlet point
					starRadius = queries.computeStarRadius(state.currentPt, walkSettings.minStarRadius,
														   dirichletDist, walkSettings.silhouettePrecision,
														   flipNormalOrientation);

					// shrink the radius slightly for numerical robustness---using a conservative
					// distance does not impact correctness
					if (walkSettings.minStarRadius <= dirichletDist) {
						starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius, walkSettings.minStarRadius);
					}
				}
			}

			// update the ball center and radius
			greensFn->updateBall(state.currentPt, starRadius);

			// sample a direction uniformly
			Vector<DIM> direction = sampleUnitSphereUniform<DIM>(sampler);

			// perform hemispherical sampling if on the Neumann boundary, which cancels
			// the alpha term in our integral expression
			if (state.onNeumannBoundary && state.currentNormal.dot(direction) > 0.0f) {
				direction *= -1.0f;
			}

			// check if there is an intersection with the Neumann boundary along the ray:
			// currentPt + starRadius * direction
			IntersectionPoint<DIM> intersectionPt;
			bool intersectedNeumann = queries.intersectWithNeumann(state.currentPt, state.currentNormal, direction,
																   starRadius, state.onNeumannBoundary, intersectionPt);

			// check if there is no intersection with the Neumann boundary
			if (!intersectedNeumann) {
				// apply small offset to the current pt for numerical robustness if it on
				// the Neumann boundary---the same offset is applied during ray intersections
				Vector<DIM> currentPt = state.onNeumannBoundary ?
										queries.offsetPointAlongDirection(state.currentPt, -state.currentNormal) :
										state.currentPt;

				// set intersectionPt to a point on the spherical arc of the ball
				intersectionPt.pt = currentPt + starRadius*direction;
				intersectionPt.dist = starRadius;
			}

			if (!walkSettings.ignoreNeumannContribution) {
				// compute the non-zero Neumann contribution inside the star-shaped region;
				// define the Neumann value to be zero outside this region
				BoundarySample<DIM> neumannSample;
				for (int i = 0; i < DIM; i++) randNumsForNeumannSampling[i] = sampler.nextFloat();
				if (queries.sampleNeumann(state.currentPt, starRadius, randNumsForNeumannSampling, neumannSample)) {
					Vector<DIM> directionToSample = neumannSample.pt - state.currentPt;
					float distToSample = directionToSample.norm();
					float alpha = state.onNeumannBoundary ? 2.0f : 1.0f;
					bool estimateBoundaryNormalAligned = false;

					if (walkSettings.solveDoubleSided) {
						// normalize the direction to the sample, and flip the sample normal
						// orientation if the geometry is front-facing; NOTE: using a precision
						// parameter since unlike direction sampling, samples can lie on the same
						// halfplane as the current walk location
						directionToSample /= distToSample;
						if (flipNormalOrientation) {
							neumannSample.normal *= -1.0f;
							estimateBoundaryNormalAligned = true;

						} else if (directionToSample.dot(neumannSample.normal) < -walkSettings.silhouettePrecision) {
							bool flipNeumannSampleNormal = true;
							if (alpha > 1.0f) {
								// on concave boundaries, we want to sample back-facing neumann
								// values on front-facing geometry below the hemisphere, so we
								// avoid flipping the normal orientation in this case
								flipNeumannSampleNormal = directionToSample.dot(state.currentNormal) <
														  -walkSettings.silhouettePrecision;
							}

							if (flipNeumannSampleNormal) {
								neumannSample.normal *= -1.0f;
								estimateBoundaryNormalAligned = true;
							}
						}
					}

					if (neumannSample.pdf > 0.0f && distToSample < starRadius &&
						!queries.intersectsWithNeumann(state.currentPt, neumannSample.pt, state.currentNormal,
													   neumannSample.normal, state.onNeumannBoundary, true)) {
						float G = greensFn->evaluate(state.currentPt, neumannSample.pt);
						T h = walkSettings.solveDoubleSided ?
							  pde.neumannDoubleSided(neumannSample.pt, estimateBoundaryNormalAligned) :
							  pde.neumann(neumannSample.pt);
						state.totalNeumannContribution += state.throughput*alpha*G*h/neumannSample.pdf;
					}
				}
			}

			if (!walkSettings.ignoreSourceContribution) {
				// compute the source contribution inside the star-shaped region;
				// define the source value to be zero outside this region
				float sourcePdf;
				Vector<DIM> sourcePt = greensFn->sampleVolume(direction, sampler, sourcePdf);
				if (greensFn->r <= intersectionPt.dist) {
					// NOTE: hemispherical sampling causes the alpha term to cancel when
					// currentPt is on the Neumann boundary; in this case, the green's function
					// norm remains unchanged even though our domain is a hemisphere;
					// for double-sided problems in watertight domains, both the current pt
					// and source pt lie either inside or outside the domain by construction
					T sourceContribution = greensFn->norm()*pde.source(sourcePt);
					state.totalSourceContribution += state.throughput*sourceContribution;
				}
			}

			// update walk position
			state.prevDistance = intersectionPt.dist;
			state.prevDirection = direction;
			state.currentPt = intersectionPt.pt;
			state.currentNormal = intersectionPt.normal; // NOTE: stale unless intersectedNeumann is true
			state.onNeumannBoundary = intersectedNeumann;

			// check if the current pt lies outside the domain; for interior problems,
			// this tests for walks that escape due to numerical error
			if (!state.onNeumannBoundary && queries.outsideBoundingDomain(state.currentPt)) {
				if (walkSettings.printLogs) {
					std::cout << "Walk escaped domain!" << std::endl;
				}

				return WalkCompletionCode::EscapedDomain;
			}

			// update the walk throughput and use russian roulette to decide whether
			// to terminate the walk
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
			if (pde.absorption > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == state.walkLength)  {
				greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorption);
			}

			// compute the distance to the dirichlet boundary
			dirichletDist = queries.computeDistToDirichlet(state.currentPt, false);
			firstStep = false;
		}

		return WalkCompletionCode::ReachedDirichletBoundary;
	}

	void setTerminalContribution(WalkCompletionCode code, const PDE<T, DIM>& pde,
								 const WalkSettings<T>& walkSettings,
								 WalkState<T, DIM>& state) const {
		if (code == WalkCompletionCode::ReachedDirichletBoundary && !walkSettings.ignoreDirichletContribution) {
			// project the walk position to the Dirichlet boundary and grab the known boundary value
			float signedDistance;
			queries.projectToDirichlet(state.currentPt, state.currentNormal,
									   signedDistance, walkSettings.solveDoubleSided);
			state.terminalContribution = walkSettings.solveDoubleSided ?
										 pde.dirichletDoubleSided(state.currentPt, signedDistance > 0.0f) :
										 pde.dirichlet(state.currentPt);

		} else if (code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution) {
			// get the user-specified terminal contribution
			getTerminalContribution(state);

		} else {
			// terminated with russian roulette or ignoring Dirichlet boundary values
			state.terminalContribution = walkSettings.initVal;
		}
	}

	// estimates only the solution of the given PDE at the input point
	void estimateSolution(const PDE<T, DIM>& pde,
						  const WalkSettings<T>& walkSettings,
						  int nWalks, SamplePoint<T, DIM>& samplePt) const {
		// initialize statistics if there are no previous estimates
		bool hasPrevEstimates = samplePt.statistics != nullptr;
		if (!hasPrevEstimates) {
			samplePt.statistics = std::make_shared<SampleStatistics<T, DIM>>(walkSettings.initVal);
		}

		// check if the sample pt is on the Dirichlet boundary
		if (samplePt.type == SampleType::OnDirichletBoundary) {
			if (!hasPrevEstimates) {
				// record the known boundary value
				T totalContribution = walkSettings.initVal;
				if (!walkSettings.ignoreDirichletContribution) {
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

		} else if (samplePt.dirichletDist <= walkSettings.epsilonShell) {
			// run just a single walk since the sample pt is inside the epsilon shell
			nWalks = 1;
		}

		// for problems with double-sided boundary conditions, initialize the direction
		// of approach for walks, and flip the current normal orientation if the geometry
		// is front-facing
		Vector<DIM> currentNormal = samplePt.normal;
		Vector<DIM> prevDirection = samplePt.normal;
		float prevDistance = std::numeric_limits<float>::max();
		bool flipNormalOrientation = false;

		if (walkSettings.solveDoubleSided && samplePt.type == SampleType::OnNeumannBoundary) {
			if (samplePt.estimateBoundaryNormalAligned) {
				currentNormal *= -1.0f;
				prevDirection *= -1.0f;
				flipNormalOrientation = true;
			}
		}

		// precompute the first sphere radius for all walks
		if (!hasPrevEstimates) {
			if (samplePt.dirichletDist > walkSettings.epsilonShell && walkSettings.stepsBeforeUsingMaximalSpheres != 0) {
				// compute the star radius; NOTE: using dirichletDist as the maximum radius for
				// the closest silhouette query can result in a smaller than maximal star-shaped
				// region: should ideally use the distance to the closest visible Dirichlet point
				float starRadius = queries.computeStarRadius(samplePt.pt, walkSettings.minStarRadius,
															 samplePt.dirichletDist, walkSettings.silhouettePrecision,
															 flipNormalOrientation);

				// shrink the radius slightly for numerical robustness---using a conservative
				// distance does not impact correctness
				if (walkSettings.minStarRadius <= samplePt.dirichletDist) {
					starRadius = std::max(RADIUS_SHRINK_PERCENTAGE*starRadius, walkSettings.minStarRadius);
				}

				samplePt.firstSphereRadius = starRadius;

			} else {
				samplePt.firstSphereRadius = samplePt.dirichletDist;
			}
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
			WalkState<T, DIM> state(samplePt.pt, currentNormal, prevDirection, prevDistance,
									1.0f, samplePt.type == SampleType::OnNeumannBoundary, 0,
									walkSettings.initVal);

			// perform walk
			WalkCompletionCode code = walk(pde, walkSettings, samplePt.dirichletDist,
										   samplePt.firstSphereRadius, flipNormalOrientation,
										   samplePt.sampler, greensFn, state);

			if ((code == WalkCompletionCode::ReachedDirichletBoundary ||
				 code == WalkCompletionCode::TerminatedWithRussianRoulette) ||
				(code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution)) {
				// compute the walk contribution
				setTerminalContribution(code, pde, walkSettings, state);
				T totalContribution = state.throughput*state.terminalContribution +
									  state.totalNeumannContribution +
									  state.totalSourceContribution;

				// update statistics
				samplePt.statistics->addSolutionEstimate(totalContribution);
				samplePt.statistics->addWalkLength(state.walkLength);
			}
		}
	}

	// estimates the solution and gradient of the given PDE at the input point;
	// NOTE: assumes the point does not lie on the boundary; the directional derivative
	// can be accessed through samplePt.statistics->getEstimatedDerivative()
	void estimateSolutionAndGradient(const PDE<T, DIM>& pde,
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

		// use the distance to the boundary as the first sphere radius for all walks;
		// shrink the radius slightly for numerical robustness---using a conservative
		// distance does not impact correctness
		float boundaryDist = std::min(samplePt.dirichletDist, samplePt.neumannDist);
		samplePt.firstSphereRadius = RADIUS_SHRINK_PERCENTAGE*boundaryDist;

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

				state.prevDistance = greensFn->R;
				state.prevDirection = (greensFn->ySurf - state.currentPt)/greensFn->R;
				state.currentPt = greensFn->ySurf;
				state.throughput *= greensFn->poissonKernel()/boundaryPdf;
				state.boundaryGradientDirection = greensFn->poissonKernelGradient()/(boundaryPdf*state.throughput);

				// compute the distance to the Dirichlet boundary
				float dirichletDist = queries.computeDistToDirichlet(state.currentPt, false);

				// perform walk
				samplePt.sampler.seed(seed);
				WalkCompletionCode code = walk(pde, walkSettings, dirichletDist, 0.0f,
											   false, samplePt.sampler, greensFn, state);

				if ((code == WalkCompletionCode::ReachedDirichletBoundary ||
					 code == WalkCompletionCode::TerminatedWithRussianRoulette) ||
					(code == WalkCompletionCode::ExceededMaxWalkLength && getTerminalContribution)) {
					// compute the walk contribution
					setTerminalContribution(code, pde, walkSettings, state);
					T totalContribution = state.throughput*state.terminalContribution +
										  state.totalNeumannContribution +
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

	// members
	const GeometricQueries<DIM>& queries;
	std::function<void(WalkState<T, DIM>&)> getTerminalContribution;
};

template <typename T, int DIM>
struct SamplePoint {
	// constructor
	SamplePoint(const Vector<DIM>& pt_, const Vector<DIM>& normal_, SampleType type_,
				float pdf_, float dirichletDist_, float neumannDist_, T initVal_):
				pt(pt_), normal(normal_), type(type_), pdf(pdf_),
				dirichletDist(dirichletDist_),
				neumannDist(neumannDist_),
				firstSphereRadius(0.0f),
				estimateBoundaryNormalAligned(false) {
		reset(initVal_);
	}

	// resets solution data
	void reset(T initVal) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		sampler = pcg32(seed);
		statistics = nullptr;
		solution = initVal;
		normalDerivative = initVal;
		source = initVal;
	}

	// members
	pcg32 sampler;
	Vector<DIM> pt;
	Vector<DIM> normal;
	SampleType type;
	float pdf;
	float dirichletDist;
	float neumannDist;
	float firstSphereRadius; // populated by WalkOnStars
	bool estimateBoundaryNormalAligned;
	std::shared_ptr<SampleStatistics<T, DIM>> statistics; // populated by WalkOnStars
	T solution, normalDerivative, source; // not populated by WalkOnStars, but available for downstream use (e.g. boundary value caching)
};

template <int DIM>
struct SampleEstimationData {
	// constructors
	SampleEstimationData(): nWalks(0), estimationQuantity(EstimationQuantity::None),
							directionForDerivative(Vector<DIM>::Zero()) {
		directionForDerivative(0) = 1.0f;
	}
	SampleEstimationData(int nWalks_, EstimationQuantity estimationQuantity_,
						 Vector<DIM> directionForDerivative_=Vector<DIM>::Zero()):
						 nWalks(nWalks_), estimationQuantity(estimationQuantity_),
						 directionForDerivative(directionForDerivative_) {}

	// members
	int nWalks;
	EstimationQuantity estimationQuantity;
	Vector<DIM> directionForDerivative; // needed only for computing direction derivatives
};

template <typename T>
struct WalkSettings {
	// constructors
	WalkSettings(T initVal_, float epsilonShell_, float minStarRadius_,
				 int maxWalkLength_, bool solveDoubleSided_):
				 initVal(initVal_),
				 epsilonShell(epsilonShell_),
				 minStarRadius(minStarRadius_),
				 silhouettePrecision(1e-3f),
				 russianRouletteThreshold(0.0f),
				 maxWalkLength(maxWalkLength_),
				 stepsBeforeApplyingTikhonov(maxWalkLength_),
				 stepsBeforeUsingMaximalSpheres(maxWalkLength_),
				 solveDoubleSided(solveDoubleSided_),
				 useGradientControlVariates(true),
				 useGradientAntitheticVariates(true),
				 useCosineSamplingForDerivatives(false),
				 ignoreDirichletContribution(false),
				 ignoreNeumannContribution(false),
				 ignoreSourceContribution(false),
				 printLogs(false) {}
	WalkSettings(T initVal_, float epsilonShell_, float minStarRadius_,
				 float silhouettePrecision_, float russianRouletteThreshold_,
				 int maxWalkLength_, int stepsBeforeApplyingTikhonov_,
				 int stepsBeforeUsingMaximalSpheres_, bool solveDoubleSided_,
				 bool useGradientControlVariates_, bool useGradientAntitheticVariates_,
				 bool useCosineSamplingForDerivatives_, bool ignoreDirichletContribution_,
				 bool ignoreNeumannContribution_, bool ignoreSourceContribution_,
				 bool printLogs_):
				 initVal(initVal_),
				 epsilonShell(epsilonShell_),
				 minStarRadius(minStarRadius_),
				 silhouettePrecision(silhouettePrecision_),
				 russianRouletteThreshold(russianRouletteThreshold_),
				 maxWalkLength(maxWalkLength_),
				 stepsBeforeApplyingTikhonov(stepsBeforeApplyingTikhonov_),
				 stepsBeforeUsingMaximalSpheres(stepsBeforeUsingMaximalSpheres_),
				 solveDoubleSided(solveDoubleSided_),
				 useGradientControlVariates(useGradientControlVariates_),
				 useGradientAntitheticVariates(useGradientAntitheticVariates_),
				 useCosineSamplingForDerivatives(useCosineSamplingForDerivatives_),
				 ignoreDirichletContribution(ignoreDirichletContribution_),
				 ignoreNeumannContribution(ignoreNeumannContribution_),
				 ignoreSourceContribution(ignoreSourceContribution_),
				 printLogs(printLogs_) {}

	// members
	T initVal;
	float epsilonShell;
	float minStarRadius;
	float silhouettePrecision;
	float russianRouletteThreshold;
	int maxWalkLength;
	int stepsBeforeApplyingTikhonov;
	int stepsBeforeUsingMaximalSpheres;
	bool solveDoubleSided; // NOTE: this flag should be set to true if domain is open
	bool useGradientControlVariates;
	bool useGradientAntitheticVariates;
	bool useCosineSamplingForDerivatives;
	bool ignoreDirichletContribution;
	bool ignoreNeumannContribution;
	bool ignoreSourceContribution;
	bool printLogs;
};

template <typename T, int DIM>
class SampleStatistics {
public:
	// constructor
	SampleStatistics(T initVal) {
		reset(initVal);
	}

	// resets statistics
	void reset(T initVal) {
		solutionMean = initVal;
		solutionM2 = initVal;
		for (int i = 0; i < DIM; i++) {
			gradientMean[i] = initVal;
			gradientM2[i] = initVal;
		}
		totalFirstSourceContribution = initVal;
		totalDerivativeContribution = initVal;
		nSolutionEstimates = 0;
		nGradientEstimates = 0;
		totalWalkLength = 0;
	}

	// adds solution estimate to running sum
	void addSolutionEstimate(const T& estimate) {
		nSolutionEstimates += 1;
		update(estimate, solutionMean, solutionM2, nSolutionEstimates);
	}

	// adds gradient estimate to running sum
	void addGradientEstimate(const T *boundaryEstimate, const T *sourceEstimate) {
		nGradientEstimates += 1;
		for (int i = 0; i < DIM; i++) {
			update(boundaryEstimate[i] + sourceEstimate[i], gradientMean[i],
				   gradientM2[i], nGradientEstimates);
		}
	}

	// adds gradient estimate to running sum
	void addGradientEstimate(const T *estimate) {
		nGradientEstimates += 1;
		for (int i = 0; i < DIM; i++) {
			update(estimate[i], gradientMean[i], gradientM2[i], nGradientEstimates);
		}
	}

	// adds source contribution for the first step to running sum
	void addFirstSourceContribution(const T& contribution) {
		totalFirstSourceContribution += contribution;
	}

	// adds derivative contribution to running sum
	void addDerivativeContribution(const T& contribution) {
		totalDerivativeContribution += contribution;
	}

	// adds walk length to running sum
	void addWalkLength(int length) {
		totalWalkLength += length;
	}

	// returns estimated solution
	T getEstimatedSolution() const {
		return solutionMean;
	}

	// returns variance of estimated solution
	T getEstimatedSolutionVariance() const {
		int N = std::max(1, nSolutionEstimates - 1);
		return solutionM2/N;
	}

	// returns estimated gradient
	const T* getEstimatedGradient() const {
		return gradientMean;
	}

	// returns variance of estimated gradient
	std::vector<T> getEstimatedGradientVariance() const {
		int N = std::max(1, nGradientEstimates - 1);
		std::vector<T> variance(DIM);

		for (int i = 0; i < DIM; i++) {
			variance[i] = gradientM2[i]/N;
		}

		return variance;
	}

	// returns mean source contribution for the first step
	T getMeanFirstSourceContribution() const {
		int N = std::max(1, nSolutionEstimates);
		return totalFirstSourceContribution/N;
	}

	// returns estimated derivative
	T getEstimatedDerivative() const {
		int N = std::max(1, nSolutionEstimates);
		return totalDerivativeContribution/N;
	}

	// returns number of solution estimates
	int getSolutionEstimateCount() const {
		return nSolutionEstimates;
	}

	// returns number of gradient estimates
	int getGradientEstimateCount() const {
		return nGradientEstimates;
	}

	// returns mean walk length
	float getMeanWalkLength() const {
		int N = std::max(1, nSolutionEstimates);
		return (float)totalWalkLength/N;
	}

private:
	// updates statistics
	void update(const T& estimate, T& mean, T& M2, int N) {
		T delta = estimate - mean;
		mean += delta/N;
		T delta2 = estimate - mean;
		M2 += delta*delta2;
	}

	// members
	T solutionMean, solutionM2;
	T gradientMean[DIM], gradientM2[DIM];
	T totalFirstSourceContribution;
	T totalDerivativeContribution;
	int nSolutionEstimates, nGradientEstimates;
	int totalWalkLength;
};

template <typename T, int DIM>
struct WalkState {
	// constructor
	WalkState(const Vector<DIM>& currentPt_, const Vector<DIM>& currentNormal_,
			  const Vector<DIM>& prevDirection_, float prevDistance_, float throughput_,
			  bool onNeumannBoundary_, int walkLength_, T initVal_):
			  currentPt(currentPt_),
			  currentNormal(currentNormal_),
			  prevDirection(prevDirection_),
			  sourceGradientDirection(Vector<DIM>::Zero()),
			  boundaryGradientDirection(Vector<DIM>::Zero()),
			  prevDistance(prevDistance_),
			  throughput(throughput_),
			  onNeumannBoundary(onNeumannBoundary_),
			  terminalContribution(initVal_),
			  totalNeumannContribution(initVal_),
			  totalSourceContribution(initVal_),
			  firstSourceContribution(initVal_),
			  walkLength(walkLength_) {}

	// members
	Vector<DIM> currentPt;
	Vector<DIM> currentNormal;
	Vector<DIM> prevDirection;
	Vector<DIM> sourceGradientDirection;
	Vector<DIM> boundaryGradientDirection;
	float prevDistance;
	float throughput;
	bool onNeumannBoundary;
	T terminalContribution;
	T totalNeumannContribution;
	T totalSourceContribution;
	T firstSourceContribution;
	int walkLength;
};

} // zombie

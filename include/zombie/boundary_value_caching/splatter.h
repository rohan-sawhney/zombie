#pragma once

#include <zombie/boundary_value_caching/boundary_sampler.h>
#include <zombie/boundary_value_caching/domain_sampler.h>

namespace zombie {

template <typename T, int DIM>
struct EvaluationPoint;

template <int DIM>
float computeGreensFnRegularization(float r)
{
	return 1.0f;
}

template <int DIM>
float computePoissonKernelRegularization(float r)
{
	return 1.0f;
}

template <>
float computeGreensFnRegularization<3>(float r)
{
	// source: https://arxiv.org/pdf/1508.00265.pdf
	return std::erf(r);
}

template <>
float computePoissonKernelRegularization<2>(float r)
{
	// source: https://epubs.siam.org/doi/abs/10.1137/S0036142999362845
	return 1.0f - std::exp(-r*r);
}

template <>
float computePoissonKernelRegularization<3>(float r)
{
	// source: https://arxiv.org/pdf/1508.00265.pdf
	return std::erf(r) - 2.0f*r*std::exp(-r*r)/std::sqrt(M_PI);
}

// FUTURE: bias correction/compensation
template <typename T, int DIM>
class Splatter {
public:
	// constructor
	Splatter(const GeometricQueries<DIM>& queries_,
			 const WalkOnStars<T, DIM>& walkOnStars_):
			 queries(queries_), walkOnStars(walkOnStars_) {}

	// splats sample pt data to the input evaluation pt
	void splat(const PDE<T, DIM>& pde,
			   const SamplePoint<T, DIM>& samplePt,
			   float radiusClamp,
			   float kernelRegularization,
			   float dirichletDistCutoff,
			   EvaluationPoint<T, DIM>& evalPt) const {
		// don't evaluate if the distance to Dirichlet boundary is smaller than the cutoff distance
		if (evalPt.dirichletDist < dirichletDistCutoff) return;

		// initialize the greens function
		std::unique_ptr<GreensFnFreeSpace<DIM>> greensFn = nullptr;
		if (pde.absorption > 0.0f) {
			greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.absorption);

		} else {
			greensFn = std::make_unique<HarmonicGreensFnFreeSpace<DIM>>();
		}

		greensFn->updatePole(evalPt.pt);

		// evaluate
		if (samplePt.type == SampleType::OnDirichletBoundary ||
			samplePt.type == SampleType::OnNeumannBoundary) {
			splatBoundaryData(samplePt, greensFn, radiusClamp, kernelRegularization, evalPt);

		} else {
			splatSourceData(samplePt, greensFn, radiusClamp, kernelRegularization, evalPt);
		}
	}

	// splats sample pt data to the input evaluation pt
	void splat(const PDE<T, DIM>& pde,
			   const std::vector<SamplePoint<T, DIM>>& samplePts,
			   float radiusClamp,
			   float kernelRegularization,
			   float dirichletDistCutoff,
			   EvaluationPoint<T, DIM>& evalPt) const {
		// don't evaluate if the distance to Dirichlet boundary is smaller than the cutoff distance
		if (evalPt.dirichletDist < dirichletDistCutoff) return;

		// initialize the greens function
		std::unique_ptr<GreensFnFreeSpace<DIM>> greensFn = nullptr;
		if (pde.absorption > 0.0f) {
			greensFn = std::make_unique<YukawaGreensFnFreeSpace<DIM>>(pde.absorption);

		} else {
			greensFn = std::make_unique<HarmonicGreensFnFreeSpace<DIM>>();
		}

		greensFn->updatePole(evalPt.pt);

		// evaluate
		for (int i = 0; i < (int)samplePts.size(); i++) {
			if (samplePts[i].type == SampleType::OnDirichletBoundary ||
				samplePts[i].type == SampleType::OnNeumannBoundary) {
				splatBoundaryData(samplePts[i], greensFn, radiusClamp, kernelRegularization, evalPt);

			} else {
				splatSourceData(samplePts[i], greensFn, radiusClamp, kernelRegularization, evalPt);
			}
		}
	}

	// splats sample pt data to the input evaluation pts
	void splat(const PDE<T, DIM>& pde,
			   const SamplePoint<T, DIM>& samplePt,
			   float radiusClamp,
			   float kernelRegularization,
			   float dirichletDistCutoff,
			   std::vector<EvaluationPoint<T, DIM>>& evalPts,
			   bool runSingleThreaded=false) const {
		int nEvalPoints = (int)evalPts.size();
		if (runSingleThreaded) {
			for (int i = 0; i < nEvalPoints; i++) {
				splat(pde, samplePt, radiusClamp, kernelRegularization, dirichletDistCutoff, evalPts[i]);
			}

		} else {
			auto run = [&](const tbb::blocked_range<int>& range) {
				for (int i = range.begin(); i < range.end(); ++i) {
					splat(pde, samplePt, radiusClamp, kernelRegularization, dirichletDistCutoff, evalPts[i]);
				}
			};

			tbb::blocked_range<int> range(0, nEvalPoints);
			tbb::parallel_for(range, run);
		}
	}

	// splats sample pt data to the input evaluation pts
	void splat(const PDE<T, DIM>& pde,
			   const std::vector<SamplePoint<T, DIM>>& samplePts,
			   float radiusClamp,
			   float kernelRegularization,
			   float dirichletDistCutoff,
			   std::vector<EvaluationPoint<T, DIM>>& evalPts,
			   std::function<void(int, int)> reportProgress={}) const {
		const int reportGranularity = 100;
		for (int i = 0; i < (int)samplePts.size(); i++) {
			splat(pde, samplePts[i], radiusClamp, kernelRegularization, dirichletDistCutoff, evalPts);
			if (reportProgress && (i + 1)%reportGranularity == 0) reportProgress(reportGranularity, 0);
		}
		if (reportProgress) reportProgress(samplePts.size()%reportGranularity, 0);
	}

	// estimates the solution at the input evaluation pt near the Dirichlet boundary
	void estimatePointwiseNearDirichletBoundary(const PDE<T, DIM>& pde,
												const WalkSettings<T>& walkSettings,
												float dirichletDistCutoff, int nWalks,
												EvaluationPoint<T, DIM>& evalPt) const {
		if (evalPt.dirichletDist < dirichletDistCutoff) {
			// NOTE: When the evaluation pt is on the Dirichlet boundary, this setup
			// evaluates the inward boundary normal aligned solution
			SamplePoint<T, DIM> samplePt(evalPt.pt, evalPt.normal, evalPt.type, 1.0f,
										 evalPt.dirichletDist, evalPt.neumannDist,
										 walkSettings.initVal);
			SampleEstimationData<DIM> estimationData(nWalks, EstimationQuantity::Solution);
			walkOnStars.solve(pde, walkSettings, estimationData, samplePt);

			// update statistics
			evalPt.reset(walkSettings.initVal);
			T solutionEstimate = samplePt.statistics->getEstimatedSolution();
			evalPt.boundaryStatistics->addSolutionEstimate(solutionEstimate);
		}
	}

	// estimates the solution at the input evaluation pts near the Dirichlet boundary
	void estimatePointwiseNearDirichletBoundary(const PDE<T, DIM>& pde,
												const WalkSettings<T>& walkSettings,
												float dirichletDistCutoff, int nWalks,
												std::vector<EvaluationPoint<T, DIM>>& evalPts,
												bool runSingleThreaded=false) const {
		int nEvalPoints = (int)evalPts.size();
		if (runSingleThreaded) {
			for (int i = 0; i < nEvalPoints; i++) {
				estimatePointwiseNearDirichletBoundary(pde, walkSettings, dirichletDistCutoff,
													   nWalks, evalPts[i]);
			}

		} else {
			auto run = [&](const tbb::blocked_range<int>& range) {
				for (int i = range.begin(); i < range.end(); ++i) {
					estimatePointwiseNearDirichletBoundary(pde, walkSettings, dirichletDistCutoff,
														   nWalks, evalPts[i]);
				}
			};

			tbb::blocked_range<int> range(0, nEvalPoints);
			tbb::parallel_for(range, run);
		}
	}

private:
	// splats boundary sample data
	void splatBoundaryData(const SamplePoint<T, DIM>& samplePt,
						   const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
						   float radiusClamp, float kernelRegularization,
						   EvaluationPoint<T, DIM>& evalPt) const {
		// compute the contribution of the boundary sample
		const T& solution = samplePt.solution;
		const T& normalDerivative = samplePt.normalDerivative;
		const Vector<DIM>& pt = samplePt.pt;
		Vector<DIM> n = samplePt.normal*(samplePt.estimateBoundaryNormalAligned ? -1.0f : 1.0f);
		float pdf = samplePt.pdf;

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
			G *= computeGreensFnRegularization<DIM>(r);
			P *= computePoissonKernelRegularization<DIM>(r);
		}

		float alpha = evalPt.type == SampleType::OnDirichletBoundary ||
					  evalPt.type == SampleType::OnNeumannBoundary ?
					  2.0f : 1.0f;
		T solutionEstimate = alpha*(G*normalDerivative - P*solution)/pdf;

		T gradientEstimate[DIM];
		if (alpha > 1.0f) alpha = 0.0f; // FUTURE: estimate gradient on the boundary
		for (int i = 0; i < DIM; i++) {
			gradientEstimate[i] = alpha*(dG[i]*normalDerivative - dP[i]*solution)/pdf;
		}

		// update statistics
		if (samplePt.estimateBoundaryNormalAligned) {
			evalPt.boundaryNormalAlignedStatistics->addSolutionEstimate(solutionEstimate);
			evalPt.boundaryNormalAlignedStatistics->addGradientEstimate(gradientEstimate);

		} else {
			evalPt.boundaryStatistics->addSolutionEstimate(solutionEstimate);
			evalPt.boundaryStatistics->addGradientEstimate(gradientEstimate);
		}
	}

	// splats source sample data
	void splatSourceData(const SamplePoint<T, DIM>& samplePt,
						 const std::unique_ptr<GreensFnFreeSpace<DIM>>& greensFn,
						 float radiusClamp, float kernelRegularization,
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
			G *= computeGreensFnRegularization<DIM>(r);
		}

		float alpha = evalPt.type == SampleType::OnDirichletBoundary ||
					  evalPt.type == SampleType::OnNeumannBoundary ?
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

	// members
	const GeometricQueries<DIM>& queries;
	const WalkOnStars<T, DIM>& walkOnStars;
};

template <typename T, int DIM>
struct EvaluationPoint {
	// constructor
	EvaluationPoint(const Vector<DIM>& pt_, const Vector<DIM>& normal_, SampleType type_,
					float dirichletDist_, float neumannDist_, T initVal_):
					pt(pt_), normal(normal_), type(type_),
					dirichletDist(dirichletDist_),
					neumannDist(neumannDist_) {
		boundaryStatistics = std::make_unique<SampleStatistics<T, DIM>>(initVal_);
		boundaryNormalAlignedStatistics = std::make_unique<SampleStatistics<T, DIM>>(initVal_);
		sourceStatistics = std::make_unique<SampleStatistics<T, DIM>>(initVal_);
	}

	// returns estimated solution
	T getEstimatedSolution() const {
		T solution = boundaryStatistics->getEstimatedSolution();
		solution += boundaryNormalAlignedStatistics->getEstimatedSolution();
		solution += sourceStatistics->getEstimatedSolution();

		return solution;
	}

	// returns estimated gradient
	void getEstimatedGradient(std::vector<T>& gradient) const {
		gradient.resize(DIM);
		for (int i = 0; i < DIM; i++) {
			gradient[i] = boundaryStatistics->getEstimatedGradient()[i];
			gradient[i] += boundaryNormalAlignedStatistics->getEstimatedGradient()[i];
			gradient[i] += sourceStatistics->getEstimatedGradient()[i];
		}
	}

	// resets statistics
	void reset(T initVal) {
		boundaryStatistics->reset(initVal);
		boundaryNormalAlignedStatistics->reset(initVal);
		sourceStatistics->reset(initVal);
	}

	// members
	Vector<DIM> pt;
	Vector<DIM> normal;
	SampleType type;
	float dirichletDist;
	float neumannDist;
	std::unique_ptr<SampleStatistics<T, DIM>> boundaryStatistics;
	std::unique_ptr<SampleStatistics<T, DIM>> boundaryNormalAlignedStatistics;
	std::unique_ptr<SampleStatistics<T, DIM>> sourceStatistics;
};

} // zombie

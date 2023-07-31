#pragma once

#include <zombie/core/pde.h>
#include <zombie/core/geometric_queries.h>
#include <zombie/core/sampling.h>

namespace zombie {

// FUTURE:
// - improve stratification, since it helps reduce clumping/singular artifacts
// - sample points in the domain in proportion to source values
template <typename T, int DIM>
class DomainSampler {
public:
	// constructor
	DomainSampler(const GeometricQueries<DIM>& queries_,
				  const std::function<bool(const Vector<DIM>&)>& insideSolveRegion_,
				  const Vector<DIM>& solveRegionMin_,
				  const Vector<DIM>& solveRegionMax_,
				  float solveRegionVolume_):
				  queries(queries_),
				  insideSolveRegion(insideSolveRegion_),
				  solveRegionMin(solveRegionMin_),
				  solveRegionMax(solveRegionMax_),
				  solveRegionVolume(solveRegionVolume_) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		sampler = pcg32(seed);
	}

	// generates uniformly distributed sample points inside the solve region;
	// NOTE: may not generate exactly the requested number of samples when the
	// solve region volume does not match the volume of its bounding extents
	void generateSamples(const PDE<T, DIM>& pde, int nTotalSamples,
						 std::vector<SamplePoint<T, DIM>>& samplePts) {
		// initialize sample points
		samplePts.clear();
		Vector<DIM> regionExtent = solveRegionMax - solveRegionMin;
		float pdf = 1.0f/solveRegionVolume;

		// generate stratified samples
		std::vector<float> stratifiedSamples;
		int nStratifiedSamples = nTotalSamples;
		if (solveRegionVolume > 0.0f) nStratifiedSamples *= regionExtent.prod()*pdf;
		generateStratifiedSamples<DIM>(stratifiedSamples, nStratifiedSamples, sampler);

		// generate sample points inside the solve region
		for (int i = 0; i < nStratifiedSamples; i++) {
			Vector<DIM> randomVector = Vector<DIM>::Zero();
			for (int j = 0; j < DIM; j++) randomVector[j] = stratifiedSamples[DIM*i + j];
			Vector<DIM> pt = (solveRegionMin.array() + regionExtent.array()*randomVector.array()).matrix();
			if (!insideSolveRegion(pt)) continue;

			T source = pde.source(pt);
			float dirichletDist = queries.computeDistToDirichlet(pt, false);
			float neumannDist = queries.computeDistToNeumann(pt, false);
			samplePts.emplace_back(SamplePoint<T, DIM>(pt, Vector<DIM>::Zero(), SampleType::InDomain,
													   pdf, dirichletDist, neumannDist, source));
		}
	}

private:
	// members
	pcg32 sampler;
	const GeometricQueries<DIM>& queries;
	const std::function<bool(const Vector<DIM>&)>& insideSolveRegion;
	const Vector<DIM>& solveRegionMin;
	const Vector<DIM>& solveRegionMax;
	float solveRegionVolume;
};

} // zombie

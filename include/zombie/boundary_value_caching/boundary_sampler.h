#pragma once

#include <zombie/point_estimation/walk_on_stars.h>
#include <unordered_map>

namespace zombie {

template <int DIM>
inline Vector<DIM> lineSegmentNormal(const Vector<DIM>& pa, const Vector<DIM>& pb, bool normalize)
{
	std::cerr << "lineSegmentNormal not implemented for DIM: " << DIM << std::endl;
	return Vector<DIM>::Zero();
}

template <>
inline Vector2 lineSegmentNormal<2>(const Vector2& pa, const Vector2& pb, bool normalize)
{
	Vector2 s = pb - pa;
	Vector2 n(s[1], -s[0]);

	return normalize ? n.normalized() : n;
}

template <int DIM>
inline float lineSegmentSurfaceArea(const Vector<DIM>& pa, const Vector<DIM>& pb)
{
	std::cerr << "lineSegmentSurfaceArea not implemented for DIM: " << DIM << std::endl;
	return 0.0f;
}

template <>
inline float lineSegmentSurfaceArea<2>(const Vector2& pa, const Vector2& pb)
{
	return lineSegmentNormal<2>(pa, pb, false).norm();
}

template <int DIM>
inline Vector<DIM> triangleNormal(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc, bool normalize)
{
	std::cerr << "triangleNormal not implemented for DIM: " << DIM << std::endl;
	return Vector<DIM>::Zero();
}

template <>
inline Vector3 triangleNormal<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc, bool normalize)
{
	Vector3 n = (pb - pa).cross(pc - pa);

	return normalize ? n.normalized() : n;
}

template <int DIM>
inline float triangleSurfaceArea(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc)
{
	std::cerr << "triangleSurfaceArea not implemented for DIM: " << DIM << std::endl;
	return 0.0f;
}

template <>
inline float triangleSurfaceArea<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc)
{
	return 0.5f*triangleNormal<3>(pa, pb, pc, false).norm();
}

template <int DIM>
inline float triangleAngle(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc)
{
	std::cerr << "triangleAngle not implemented for DIM: " << DIM << std::endl;
	return 0.0f;
}

template <>
inline float triangleAngle<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc)
{
	Vector3 u = (pb - pa).normalized();
	Vector3 v = (pc - pa).normalized();

	return std::acos(std::max(-1.0f, std::min(1.0f, u.dot(v))));
}

// NOTE: currently specialized to line segments in 2D and triangles in 3D
// FUTURE:
// - improve stratification, since it helps reduce clumping/singular artifacts
// - sample points on the boundary in proportion to dirichlet and neumann values
template <typename T, int DIM>
class BoundarySampler {
public:
	// constructor
	BoundarySampler(const std::vector<Vector<DIM>>& positions_,
					const std::vector<std::vector<size_t>>& indices_,
					const GeometricQueries<DIM>& queries_,
					const WalkOnStars<T, DIM>& walkOnStars_,
					const std::function<bool(const Vector<DIM>&)>& insideSolveRegion_,
					const std::function<bool(const Vector<DIM>&)>& onNeumannBoundary_):
					positions(positions_), indices(indices_), queries(queries_),
					walkOnStars(walkOnStars_), insideSolveRegion(insideSolveRegion_),
					onNeumannBoundary(onNeumannBoundary_), boundaryArea(0.0f),
					boundaryAreaNormalAligned(0.0f) {
		unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
		sampler = pcg32(seed);
	}

	// initialize sampler
	void initialize(float normalOffsetForDirichletSamples, bool solveDoubleSided,
					bool computeWeightedNormals=false) {
		// compute normals
		computeNormals(computeWeightedNormals);

		// build a cdf table for the boundary with Dirichlet vertices displaced along inward normals
		buildCDFTable(cdfTable, boundaryArea, -1.0f*normalOffsetForDirichletSamples);

		if (solveDoubleSided) {
			// build a cdf table for the boundary with Dirichlet vertices displaced along outward normals
			buildCDFTable(cdfTableNormalAligned, boundaryAreaNormalAligned, normalOffsetForDirichletSamples);
		}
	}

	// generates uniformly distributed sample points on the boundary
	void generateSamples(int nTotalSamples, float normalOffsetForDirichletSamples,
						 bool solveDoubleSided, T initVal,
						 std::vector<SamplePoint<T, DIM>>& samplePts,
						 std::vector<SamplePoint<T, DIM>>& samplePtsNormalAligned) {
		if (solveDoubleSided) {
			// decide sample count split based on boundary areas
			float totalBoundaryArea = boundaryArea + boundaryAreaNormalAligned;
			int nSamples = std::ceil(nTotalSamples*boundaryArea/totalBoundaryArea);
			int nSamplesNormalAligned = std::ceil(nTotalSamples*boundaryAreaNormalAligned/totalBoundaryArea);

			// generate samples
			generateSamples(cdfTable, nSamples, boundaryArea, -1.0f*normalOffsetForDirichletSamples,
							initVal, samplePts);
			generateSamples(cdfTableNormalAligned, nSamplesNormalAligned, boundaryAreaNormalAligned,
							normalOffsetForDirichletSamples, initVal, samplePtsNormalAligned);

		} else {
			generateSamples(cdfTable, nTotalSamples, boundaryArea, -1.0f*normalOffsetForDirichletSamples,
							initVal, samplePts);
		}
	}

	// solves the given PDE at the generated sample points
	void computeEstimates(const PDE<T, DIM>& pde, const WalkSettings<T>& walkSettings,
						  int nWalksForSolutionEstimates, int nWalksForGradientEstimates,
						  std::vector<SamplePoint<T, DIM>>& samplePts,
						  bool useFiniteDifferences=false, bool runSingleThreaded=false,
						  std::function<void(int,int)> reportProgress={}) const {
		// initialize estimation quantities
		int nSamples = (int)samplePts.size();
		std::vector<SampleEstimationData<DIM>> estimationData(nSamples);

		for (int i = 0; i < nSamples; i++) {
			SamplePoint<T, DIM>& samplePt = samplePts[i];

			if (samplePt.type == SampleType::OnDirichletBoundary) {
				if (useFiniteDifferences) {
					estimationData[i].estimationQuantity = EstimationQuantity::Solution;
					samplePt.type = SampleType::InDomain;

				} else {
					Vector<DIM> normal = samplePt.normal;
					if (walkSettings.solveDoubleSided && samplePt.estimateBoundaryNormalAligned) {
						normal *= -1.0f;
					}

					estimationData[i].estimationQuantity = EstimationQuantity::SolutionAndGradient;
					estimationData[i].directionForDerivative = normal;
				}

				estimationData[i].nWalks = nWalksForGradientEstimates;

			} else if (samplePt.type == SampleType::OnNeumannBoundary) {
				estimationData[i].estimationQuantity = EstimationQuantity::Solution;
				estimationData[i].nWalks = nWalksForSolutionEstimates;

			} else {
				std::cerr << "BoundarySampler::computeEstimates(): Invalid sample type!" << std::endl;
				exit(EXIT_FAILURE);
			}
		}

		// compute estimates
		walkOnStars.solve(pde, walkSettings, estimationData, samplePts, runSingleThreaded, reportProgress);

		// set estimated boundary data
		for (int i = 0; i < nSamples; i++) {
			SamplePoint<T, DIM>& samplePt = samplePts[i];
			samplePt.solution = samplePt.statistics->getEstimatedSolution();

			if (samplePt.type == SampleType::OnNeumannBoundary) {
				if (!walkSettings.ignoreNeumannContribution) {
					samplePt.normalDerivative = walkSettings.solveDoubleSided ?
												pde.neumannDoubleSided(samplePt.pt, samplePt.estimateBoundaryNormalAligned) :
												pde.neumann(samplePt.pt);
				}

			} else {
				if (useFiniteDifferences) {
					// use biased gradient estimates
					float signedDistance;
					Vector<DIM> normal;
					Vector<DIM> pt = samplePt.pt;
					queries.projectToDirichlet(pt, normal, signedDistance, walkSettings.solveDoubleSided);
					T dirichlet = walkSettings.solveDoubleSided ?
								  pde.dirichletDoubleSided(pt, signedDistance > 0.0f) :
								  pde.dirichlet(pt);

					samplePt.normalDerivative = dirichlet - samplePt.solution;
					samplePt.normalDerivative /= std::fabs(signedDistance);
					samplePt.type = SampleType::OnDirichletBoundary;

				} else {
					// use unbiased gradient estimates
					samplePt.normalDerivative = samplePt.statistics->getEstimatedDerivative();
				}
			}
		}
	}

private:
	// computes normals
	void computeNormals(bool computeWeighted) {
		int nPrimitives = (int)indices.size();
		int nPositions = (int)positions.size();
		normals.clear();
		normals.resize(nPositions, Vector<DIM>::Zero());

		for (int i = 0; i < nPrimitives; i++) {
			const std::vector<size_t>& index = indices[i];

			if (DIM == 2) {
				const Vector<DIM>& pa = positions[index[0]];
				const Vector<DIM>& pb = positions[index[1]];
				Vector<DIM> n = lineSegmentNormal<DIM>(pa, pb, !computeWeighted);

				normals[index[0]] += n;
				normals[index[1]] += n;

			} else if (DIM == 3) {
				const Vector<DIM>& pa = positions[index[0]];
				const Vector<DIM>& pb = positions[index[1]];
				const Vector<DIM>& pc = positions[index[2]];
				Vector<DIM> n = triangleNormal<DIM>(pa, pb, pc, true);

				for (int j = 0; j < 3; j++) {
					const Vector<DIM>& p0 = positions[index[(j + 0)%3]];
					const Vector<DIM>& p1 = positions[index[(j + 1)%3]];
					const Vector<DIM>& p2 = positions[index[(j + 2)%3]];
					float angle = computeWeighted ? triangleAngle<DIM>(p0, p1, p2) : 1.0f;

					normals[index[j]] += angle*n;
				}
			}
		}

		for (int i = 0; i < nPositions; i++) {
			normals[i].normalize();
		}
	}

	// builds a cdf table for sampling; FUTURE: to get truly unbiased results, introduce
	// additional primitives at the Neumann-Dirichlet interface that can be sampled
	void buildCDFTable(CDFTable& table, float& totalArea, float normalOffset) {
		int nPrimitives = (int)indices.size();
		std::vector<float> weights(nPrimitives, 0.0f);

		for (int i = 0; i < nPrimitives; i++) {
			const std::vector<size_t>& index = indices[i];

			if (DIM == 2) {
				Vector<DIM> pa = positions[index[0]];
				Vector<DIM> pb = positions[index[1]];
				Vector<DIM> pMid = (pa + pb)/2.0f;
				Vector<DIM> n = lineSegmentNormal<DIM>(pa, pb, true);

				// don't generate any samples on the boundary outside the solve region
				if (insideSolveRegion(pMid + normalOffset*n)) {
					if (!onNeumannBoundary(pMid)) {
						pa += normalOffset*normals[index[0]];
						pb += normalOffset*normals[index[1]];
					}

					weights[i] = lineSegmentSurfaceArea<DIM>(pa, pb);
				}

			} else if (DIM == 3) {
				Vector<DIM> pa = positions[index[0]];
				Vector<DIM> pb = positions[index[1]];
				Vector<DIM> pc = positions[index[2]];
				Vector<DIM> pMid = (pa + pb + pc)/3.0f;
				Vector<DIM> n = triangleNormal<DIM>(pa, pb, pc, true);

				// don't generate any samples on the boundary outside the solve region
				if (insideSolveRegion(pMid + normalOffset*n)) {
					if (!onNeumannBoundary(pMid)) {
						pa += normalOffset*normals[index[0]];
						pb += normalOffset*normals[index[1]];
						pc += normalOffset*normals[index[2]];
					}

					weights[i] = triangleSurfaceArea<DIM>(pa, pb, pc);
				}
			}
		}

		totalArea = table.build(weights);
	}

	// generates uniformly distributed sample points on the boundary
	void generateSamples(const CDFTable& table, int nSamples,
						 float totalArea, float normalOffset, T initVal,
						 std::vector<SamplePoint<T, DIM>>& samplePts) {
		samplePts.clear();
		float pdf = 1.0f/totalArea;

		if (totalArea > 0.0f) {
			// generate stratified samples for CDF table sampling
			std::vector<float> stratifiedSamples;
			generateStratifiedSamples<1>(stratifiedSamples, nSamples, sampler);

			// count the number of times a mesh face is sampled from the CDF table
			std::unordered_map<int, int> indexCount;
			for (int i = 0; i < nSamples; i++) {
				float u = stratifiedSamples[i];
				int offset = table.sample(u);

				if (indexCount.find(offset) == indexCount.end()) {
					indexCount[offset] = 1;

				} else {
					indexCount[offset]++;
				}
			}

			for (auto& kv: indexCount) {
				// generate samples for selected mesh face
				std::vector<float> indexSamples;
				const std::vector<size_t>& index = indices[kv.first];
				if (kv.second == 1) {
					for (int i = 0; i < DIM - 1; i++) {
						indexSamples.emplace_back(sampler.nextFloat());
					}

				} else {
					generateStratifiedSamples<DIM - 1>(indexSamples, kv.second, sampler);
				}

				for (int i = 0; i < kv.second; i++) {
					// generate sample point
					Vector<DIM> pt = Vector<DIM>::Zero();
					Vector<DIM> normal = Vector<DIM>::Zero();
					SampleType sampleType = SampleType::OnNeumannBoundary;

					if (DIM == 2) {
						Vector<DIM> pa = positions[index[0]];
						Vector<DIM> pb = positions[index[1]];
						Vector<DIM> pMid = (pa + pb)/2.0f;

						if (!onNeumannBoundary(pMid)) {
							sampleType = SampleType::OnDirichletBoundary;
							pa += normalOffset*normals[index[0]];
							pb += normalOffset*normals[index[1]];
						}

						sampleLineSegmentUniformly<DIM>(pa, pb, &indexSamples[(DIM - 1)*i], pt, normal);

					} else if (DIM == 3) {
						Vector<DIM> pa = positions[index[0]];
						Vector<DIM> pb = positions[index[1]];
						Vector<DIM> pc = positions[index[2]];
						Vector<DIM> pMid = (pa + pb + pc)/3.0f;

						if (!onNeumannBoundary(pMid)) {
							sampleType = SampleType::OnDirichletBoundary;
							pa += normalOffset*normals[index[0]];
							pb += normalOffset*normals[index[1]];
							pc += normalOffset*normals[index[2]];
						}

						sampleTriangleUniformly<DIM>(pa, pb, pc, &indexSamples[(DIM - 1)*i], pt, normal);
					}

					float dirichletDist = queries.computeDistToDirichlet(pt, false);
					float neumannDist = queries.computeDistToNeumann(pt, false);
					samplePts.emplace_back(SamplePoint<T, DIM>(pt, normal, sampleType, pdf,
															   dirichletDist, neumannDist, initVal));
				}
			}

			if (normalOffset > 0.0f) {
				// invert the orientation of the boundary normals during estimation,
				// with Dirichlet vertices displaced along these normals
				for (int i = 0; i < nSamples; i++) {
					samplePts[i].estimateBoundaryNormalAligned = true;
				}
			}

		} else {
			std::cout << "CDF table is empty!" << std::endl;
		}
	}

	// members
	pcg32 sampler;
	const std::vector<Vector<DIM>>& positions;
	const std::vector<std::vector<size_t>>& indices;
	const GeometricQueries<DIM>& queries;
	const WalkOnStars<T, DIM>& walkOnStars;
	const std::function<bool(const Vector<DIM>&)>& insideSolveRegion;
	const std::function<bool(const Vector<DIM>&)>& onNeumannBoundary;
	std::vector<Vector<DIM>> normals;
	CDFTable cdfTable, cdfTableNormalAligned;
	float boundaryArea, boundaryAreaNormalAligned;
};

} // zombie

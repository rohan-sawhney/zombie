#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace zombie {

template<int DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <int DIM>
struct IntersectionPoint {
	// constructors
	IntersectionPoint(): pt(Vector<DIM>::Zero()), normal(Vector<DIM>::Zero()),
						 dist(std::numeric_limits<float>::max()) {}
	IntersectionPoint(const Vector<DIM>& pt_, const Vector<DIM>& normal_, float dist_):
					  pt(pt_), normal(normal_), dist(dist_) {}

	// members
	Vector<DIM> pt;
	Vector<DIM> normal;
	float dist;
};

template <int DIM>
struct BoundarySample {
	// constructors
	BoundarySample(): pt(Vector<DIM>::Zero()), normal(Vector<DIM>::Zero()), pdf(0.0f) {}
	BoundarySample(const Vector<DIM>& pt_, const Vector<DIM>& normal_, float pdf_):
				   pt(pt_), normal(normal_), pdf(pdf_) {}

	// members
	Vector<DIM> pt;
	Vector<DIM> normal;
	float pdf;
};

template <int DIM>
struct GeometricQueries {
	// constructor
	GeometricQueries(bool domainIsWatertight_): domainIsWatertight(domainIsWatertight_) {}

	// members
	bool domainIsWatertight;
	std::function<float(const Vector<DIM>&, bool)> computeDistToDirichlet;
	std::function<float(const Vector<DIM>&, bool)> computeDistToNeumann;
	std::function<float(const Vector<DIM>&, bool, bool&)> computeDistToBoundary;
	std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToDirichlet;
	std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToNeumann;
	std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool&, bool)> projectToBoundary;
	std::function<Vector<DIM>(const Vector<DIM>&, const Vector<DIM>&)> offsetPointAlongDirection;
	std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
					   float, bool, IntersectionPoint<DIM>&)> intersectWithDirichlet;
	std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
					   float, bool, IntersectionPoint<DIM>&)> intersectWithNeumann;
	std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
					   const Vector<DIM>&, bool, bool)> intersectsWithNeumann;
	std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
					   float, bool, bool, IntersectionPoint<DIM>&, bool&)> intersectWithBoundary;
	std::function<int(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
					  float, bool, bool, std::vector<IntersectionPoint<DIM>>&,
					  std::vector<bool>&)> intersectWithBoundaryAllHits;
	std::function<bool(const Vector<DIM>&, float, float *, BoundarySample<DIM>&)> sampleNeumann;
	std::function<float(const Vector<DIM>&, float, float, float, bool)> computeStarRadius;
	std::function<bool(const Vector<DIM>&)> insideDomain; // NOTE: specialized to watertight domains
	std::function<bool(const Vector<DIM>&)> outsideBoundingDomain;
};

} // zombie

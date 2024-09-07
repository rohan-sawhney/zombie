#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

namespace zombie {

template<size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <size_t DIM>
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

template <size_t DIM>
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

template <size_t DIM>
struct GeometricQueries {
    // constructor
    GeometricQueries(bool domainIsWatertight_): domainIsWatertight(domainIsWatertight_) {}

    // members
    bool domainIsWatertight;
    std::function<float(const Vector<DIM>&, bool)> computeDistToAbsorbingBoundary;
    std::function<float(const Vector<DIM>&, bool)> computeDistToReflectingBoundary;
    std::function<float(const Vector<DIM>&, bool)> computeDistToBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToAbsorbingBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToReflectingBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToBoundary;
    std::function<Vector<DIM>(const Vector<DIM>&, const Vector<DIM>&)> offsetPointAlongDirection;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, IntersectionPoint<DIM>&)> intersectAbsorbingBoundary;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, IntersectionPoint<DIM>&)> intersectReflectingBoundary;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, bool, IntersectionPoint<DIM>&)> intersectBoundary;
    std::function<int(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                      float, bool, bool, std::vector<IntersectionPoint<DIM>>&)> intersectBoundaryAllHits;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       const Vector<DIM>&, bool, bool)> intersectsWithReflectingBoundary;
    std::function<bool(const Vector<DIM>&, float, const Vector<DIM>&, BoundarySample<DIM>&)> sampleReflectingBoundary;
    std::function<float(const Vector<DIM>&, float, float, float, bool)> computeStarRadiusForReflectingBoundary;
    std::function<bool(const Vector<DIM>&)> insideDomain; // NOTE: specialized to watertight domains
    std::function<bool(const Vector<DIM>&)> outsideBoundingDomain;
    std::function<float()> computeSignedDomainVolume;
};

} // zombie

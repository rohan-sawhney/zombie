// This file defines an interface for performing geometric queries on a domain boundary,
// such as computing distances to the boundary, projecting points to the boundary, and
// intersecting rays with the boundary. As part of the problem setup, users of Zombie
// should populate the callback functions defined by the GeometricQueries interface for
// the boundary representation used in their application.
//
// For surface meshes in 2D and 3D, the FcpwBoundaryHandler class provides a convenient
// way to populate the GeometricQueries interface; refer to the 'populateGeometricQueries'
// function in fcpw_boundary_handler.h for details.

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

    // computes the distance to the boundary
    std::function<float(const Vector<DIM>&, bool)> computeDistToAbsorbingBoundary;
    std::function<float(const Vector<DIM>&, bool)> computeDistToReflectingBoundary;
    std::function<float(const Vector<DIM>&, bool)> computeDistToBoundary;

    // projects a point to the boundary
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToAbsorbingBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToReflectingBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToBoundary;

    // offsets a point along a direction
    std::function<Vector<DIM>(const Vector<DIM>&, const Vector<DIM>&)> offsetPointAlongDirection;

    // intersects a ray with the boundary
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, IntersectionPoint<DIM>&)> intersectAbsorbingBoundary;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, IntersectionPoint<DIM>&)> intersectReflectingBoundary;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, bool, IntersectionPoint<DIM>&)> intersectBoundary;
    std::function<int(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                      float, bool, bool, std::vector<IntersectionPoint<DIM>>&)> intersectBoundaryAllHits;

    // checks whether there is a line of sight between two points
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       const Vector<DIM>&, bool, bool)> intersectsWithReflectingBoundary;

    // samples a point on the boundary
    std::function<bool(const Vector<DIM>&, float, const Vector<DIM>&, BoundarySample<DIM>&)> sampleReflectingBoundary;

    // computes the radius of a star-shaped region on a reflecting boundary
    std::function<float(const Vector<DIM>&, float, float, float, bool)> computeStarRadiusForReflectingBoundary;

    // checks if a point is inside the domain (assuming it is watertight)
    std::function<bool(const Vector<DIM>&, bool)> insideDomain;

    // checks if a point is outside the bounding domain
    std::function<bool(const Vector<DIM>&)> outsideBoundingDomain;

    // computes the signed volume of a domain
    std::function<float()> computeSignedDomainVolume;
};

} // zombie

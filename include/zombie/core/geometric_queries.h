// This file defines an interface for performing geometric queries on a domain boundary,
// such as computing distances to the boundary, projecting points to the boundary, and
// intersecting rays with the boundary. As part of the problem setup, users of Zombie
// should populate the callback functions defined by the GeometricQueries interface for
// the boundary representation used in their application.
//
// For surface meshes in 2D and 3D, the FcpwBoundaryHandler class provides a
// convenient way to populate the GeometricQueries interface; refer to the
// 'populateGeometricQueriesForAbsorbingBoundary' and 'populateGeometricQueriesForReflectingBoundary'
// functions in fcpw_boundary_handler.h for details.

#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>

#define RAY_OFFSET 1e-6f

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
    // constructors
    GeometricQueries();
    GeometricQueries(bool domainIsWatertight_,
                     const Vector<DIM>& domainMin_,
                     const Vector<DIM>& domainMax_);

    // computes the distance to the boundary
    std::function<float(const Vector<DIM>&, bool)> computeDistToAbsorbingBoundary;
    std::function<float(const Vector<DIM>&, bool)> computeDistToReflectingBoundary;
    std::function<float(const Vector<DIM>&, bool)> computeDistToBoundary; // set automatically

    // projects a point to the boundary
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToAbsorbingBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToReflectingBoundary;
    std::function<bool(Vector<DIM>&, Vector<DIM>&, float&, bool)> projectToBoundary; // set automatically

    // offsets a point along a direction
    std::function<Vector<DIM>(const Vector<DIM>&, const Vector<DIM>&)> offsetPointAlongDirection; // set automatically

    // intersects a ray with the boundary
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, IntersectionPoint<DIM>&)> intersectAbsorbingBoundary;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, IntersectionPoint<DIM>&)> intersectReflectingBoundary;
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       float, bool, bool, IntersectionPoint<DIM>&)> intersectBoundary; // set automatically
    std::function<int(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                      float, bool, std::vector<IntersectionPoint<DIM>>&)> intersectAbsorbingBoundaryAllHits;
    std::function<int(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                      float, bool, std::vector<IntersectionPoint<DIM>>&)> intersectReflectingBoundaryAllHits;
    std::function<int(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                      float, bool, bool, std::vector<IntersectionPoint<DIM>>&)> intersectBoundaryAllHits; // set automatically

    // checks whether there is a line of sight between two points
    std::function<bool(const Vector<DIM>&, const Vector<DIM>&, const Vector<DIM>&,
                       const Vector<DIM>&, bool, bool)> intersectsWithReflectingBoundary;

    // samples a point on the boundary
    std::function<bool(const Vector<DIM>&, float, const Vector<DIM>&, BoundarySample<DIM>&)> sampleReflectingBoundary;

    // computes the radius of a star-shaped region on a reflecting boundary
    std::function<float(const Vector<DIM>&, float, float, float, bool)> computeStarRadiusForReflectingBoundary;

    // checks if a point is inside the domain (assuming it is watertight)
    std::function<bool(const Vector<DIM>&, bool)> insideDomain; // set automatically

    // checks if a point is outside the bounding domain
    std::function<bool(const Vector<DIM>&)> outsideBoundingDomain; // set automatically

    // computes the signed volume of the domain
    std::function<float()> computeAbsorbingBoundarySignedVolume;
    std::function<float()> computeReflectingBoundarySignedVolume;
    std::function<float()> computeDomainSignedVolume; // set automatically

    // members
    bool domainIsWatertight;
    Vector<DIM> domainMin, domainMax;

protected:
    // default implementation for populating queries
    void populate();
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <size_t DIM>
inline GeometricQueries<DIM>::GeometricQueries():
domainIsWatertight(true),
domainMin(Vector<DIM>::Constant(std::numeric_limits<float>::lowest())),
domainMax(Vector<DIM>::Constant(std::numeric_limits<float>::max()))
{
    populate();
}

template <size_t DIM>
inline GeometricQueries<DIM>::GeometricQueries(bool domainIsWatertight_,
                                               const Vector<DIM>& domainMin_,
                                               const Vector<DIM>& domainMax_):
domainIsWatertight(domainIsWatertight_),
domainMin(domainMin_),
domainMax(domainMax_)
{
    populate();
}

template <size_t DIM>
inline Vector<DIM> offsetPointAlongDirectionImpl(const Vector<DIM>& p, const Vector<DIM>& n)
{
    return p + RAY_OFFSET*n;
}

inline float intAsFloat(int a)
{
    union {int a; float b;} u;
    u.a = a;

    return u.b;
}

inline int floatAsInt(float a)
{
    union {float a; int b;} u;
    u.a = a;

    return u.b;
}

template <>
inline Vector2 offsetPointAlongDirectionImpl<2>(const Vector2& p, const Vector2& n)
{
    // source: https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf
    const float origin = 1.0f/32.0f;
    const float floatScale = 1.0f/65536.0f;
    const float intScale = 256.0f;

    Eigen::Vector2i nOffset(n(0)*intScale, n(1)*intScale);
    Eigen::Vector2f pOffset(intAsFloat(floatAsInt(p(0)) + (p(0) < 0 ? -nOffset(0) : nOffset(0))),
                            intAsFloat(floatAsInt(p(1)) + (p(1) < 0 ? -nOffset(1) : nOffset(1))));

    return Vector2(std::fabs(p(0)) < origin ? p(0) + floatScale*n(0) : pOffset(0),
                   std::fabs(p(1)) < origin ? p(1) + floatScale*n(1) : pOffset(1));
}

template <>
inline Vector3 offsetPointAlongDirectionImpl<3>(const Vector3& p, const Vector3& n)
{
    // source: https://link.springer.com/content/pdf/10.1007%2F978-1-4842-4427-2_6.pdf
    const float origin = 1.0f/32.0f;
    const float floatScale = 1.0f/65536.0f;
    const float intScale = 256.0f;

    Eigen::Vector3i nOffset(n(0)*intScale, n(1)*intScale, n(2)*intScale);
    Eigen::Vector3f pOffset(intAsFloat(floatAsInt(p(0)) + (p(0) < 0 ? -nOffset(0) : nOffset(0))),
                            intAsFloat(floatAsInt(p(1)) + (p(1) < 0 ? -nOffset(1) : nOffset(1))),
                            intAsFloat(floatAsInt(p(2)) + (p(2) < 0 ? -nOffset(2) : nOffset(2))));

    return Vector3(std::fabs(p(0)) < origin ? p(0) + floatScale*n(0) : pOffset(0),
                   std::fabs(p(1)) < origin ? p(1) + floatScale*n(1) : pOffset(1),
                   std::fabs(p(2)) < origin ? p(2) + floatScale*n(2) : pOffset(2));
}

template <size_t DIM>
inline void GeometricQueries<DIM>::populate()
{
    computeDistToAbsorbingBoundary = [this](const Vector<DIM>& x, bool computeSignedDistance) -> float {
        Vector<DIM> u = this->domainMin - x;
        Vector<DIM> v = x - this->domainMax;

        return u.cwiseMin(v).norm();
    };
    computeDistToReflectingBoundary = [](const Vector<DIM>& x, bool computeSignedDistance) -> float {
        return std::numeric_limits<float>::max();
    };
    computeDistToBoundary = [this](const Vector<DIM>& x, bool computeSignedDistance) -> float {
        float d1 = this->computeDistToAbsorbingBoundary(x, computeSignedDistance);
        float d2 = this->computeDistToReflectingBoundary(x, computeSignedDistance);

        return std::fabs(d1) < std::fabs(d2) ? d1 : d2;
    };
    projectToAbsorbingBoundary = [](Vector<DIM>& x, Vector<DIM>& normal,
                                    float& distance, bool computeSignedDistance) -> bool {
        distance = 0.0f;
        return false;
    };
    projectToReflectingBoundary = [](Vector<DIM>& x, Vector<DIM>& normal,
                                     float& distance, bool computeSignedDistance) -> bool {
        distance = 0.0f;
        return false;
    };
    projectToBoundary = [this](Vector<DIM>& x, Vector<DIM>& normal,
                               float& distance, bool computeSignedDistance) -> bool {
        distance = std::numeric_limits<float>::max();
        bool didProject = false;

        Vector<DIM> absorbingBoundaryPt = x;
        Vector<DIM> absorbingBoundaryNormal;
        float distanceToAbsorbingBoundary;
        if (this->projectToAbsorbingBoundary(absorbingBoundaryPt, absorbingBoundaryNormal,
                                             distanceToAbsorbingBoundary, computeSignedDistance)) {
            x = absorbingBoundaryPt;
            normal = absorbingBoundaryNormal;
            distance = distanceToAbsorbingBoundary;
            didProject = true;
        }

        Vector<DIM> reflectingBoundaryPt = x;
        Vector<DIM> reflectingBoundaryNormal;
        float distanceToReflectingBoundary;
        if (this->projectToReflectingBoundary(reflectingBoundaryPt, reflectingBoundaryNormal,
                                              distanceToReflectingBoundary, computeSignedDistance)) {
            if (std::fabs(distanceToReflectingBoundary) < std::fabs(distance)) {
                x = reflectingBoundaryPt;
                normal = reflectingBoundaryNormal;
                distance = distanceToReflectingBoundary;
            }

            didProject = true;
        }

        if (!didProject) distance = 0.0f;
        return didProject;
    };
    offsetPointAlongDirection = [](const Vector<DIM>& x, const Vector<DIM>& dir) -> Vector<DIM> {
        return offsetPointAlongDirectionImpl<DIM>(x, dir);
    };
    intersectAbsorbingBoundary = [](const Vector<DIM>& origin, const Vector<DIM>& normal,
                                    const Vector<DIM>& dir, float tMax, bool onAborbingBoundary,
                                    IntersectionPoint<DIM>& intersectionPt) -> bool {
        return false;
    };
    intersectReflectingBoundary = [](const Vector<DIM>& origin, const Vector<DIM>& normal,
                                     const Vector<DIM>& dir, float tMax, bool onReflectingBoundary,
                                     IntersectionPoint<DIM>& intersectionPt) -> bool {
        return false;
    };
    intersectBoundary = [this](const Vector<DIM>& origin, const Vector<DIM>& normal,
                               const Vector<DIM>& dir, float tMax,
                               bool onAborbingBoundary, bool onReflectingBoundary,
                               IntersectionPoint<DIM>& intersectionPt) -> bool {
        IntersectionPoint<DIM> absorbingBoundaryIntersectionPt;
        bool intersectedAbsorbingBoundary = this->intersectAbsorbingBoundary(
            origin, normal, dir, tMax, onAborbingBoundary, absorbingBoundaryIntersectionPt);

        IntersectionPoint<DIM> reflectingBoundaryIntersectionPt;
        bool intersectedReflectingBoundary = this->intersectReflectingBoundary(
            origin, normal, dir, tMax, onReflectingBoundary, reflectingBoundaryIntersectionPt);

        if (intersectedAbsorbingBoundary && intersectedReflectingBoundary) {
            if (absorbingBoundaryIntersectionPt.dist < reflectingBoundaryIntersectionPt.dist) {
                intersectionPt = absorbingBoundaryIntersectionPt;

            } else {
                intersectionPt = reflectingBoundaryIntersectionPt;
            }

        } else if (intersectedAbsorbingBoundary) {
            intersectionPt = absorbingBoundaryIntersectionPt;

        } else if (intersectedReflectingBoundary) {
            intersectionPt = reflectingBoundaryIntersectionPt;
        }

        return intersectedAbsorbingBoundary || intersectedReflectingBoundary;
    };
    intersectAbsorbingBoundaryAllHits = [](const Vector<DIM>& origin, const Vector<DIM>& normal,
                                           const Vector<DIM>& dir, float tMax, bool onAborbingBoundary,
                                           std::vector<IntersectionPoint<DIM>>& intersectionPts) -> int {
        intersectionPts.clear();
        return 0;
    };
    intersectReflectingBoundaryAllHits = [](const Vector<DIM>& origin, const Vector<DIM>& normal,
                                            const Vector<DIM>& dir, float tMax, bool onReflectingBoundary,
                                            std::vector<IntersectionPoint<DIM>>& intersectionPts) -> int {
        intersectionPts.clear();
        return 0;
    };
    intersectBoundaryAllHits = [this](const Vector<DIM>& origin, const Vector<DIM>& normal,
                                      const Vector<DIM>& dir, float tMax,
                                      bool onAborbingBoundary, bool onReflectingBoundary,
                                      std::vector<IntersectionPoint<DIM>>& intersectionPts) -> int {
        std::vector<IntersectionPoint<DIM>> absorbingBoundaryIntersectionPts;
        int nAbsorbingBoundaryIntersections = this->intersectAbsorbingBoundaryAllHits(
            origin, normal, dir, tMax, onAborbingBoundary, absorbingBoundaryIntersectionPts);

        std::vector<IntersectionPoint<DIM>> reflectingBoundaryIntersectionPts;
        int nReflectingBoundaryIntersections = this->intersectReflectingBoundaryAllHits(
            origin, normal, dir, tMax, onReflectingBoundary, reflectingBoundaryIntersectionPts);

        intersectionPts.clear();
        intersectionPts.insert(intersectionPts.end(), absorbingBoundaryIntersectionPts.begin(),
                               absorbingBoundaryIntersectionPts.end());
        intersectionPts.insert(intersectionPts.end(), reflectingBoundaryIntersectionPts.begin(),
                               reflectingBoundaryIntersectionPts.end());

        return nAbsorbingBoundaryIntersections + nReflectingBoundaryIntersections;
    };
    intersectsWithReflectingBoundary = [](const Vector<DIM>& xi, const Vector<DIM>& xj,
                                          const Vector<DIM>& ni, const Vector<DIM>& nj,
                                          bool offseti, bool offsetj) -> bool {
        return false;
    };
    sampleReflectingBoundary = [](const Vector<DIM>& x, float radius, const Vector<DIM>& randNums,
                                  BoundarySample<DIM>& boundarySample) -> bool {
        return false;
    };
    computeStarRadiusForReflectingBoundary = [](const Vector<DIM>& x, float minRadius, float maxRadius,
                                                float silhouettePrecision, bool flipNormalOrientation) -> float {
        return maxRadius;
    };
    insideDomain = [this](const Vector<DIM>& x, bool useRayIntersections) -> bool {
        if (!this->domainIsWatertight) return true;
        if (useRayIntersections) {
            bool isInside = true;
            Vector<DIM> zero = Vector<DIM>::Zero();
            for (size_t i = 0; i < DIM; i++) {
                Vector<DIM> dir = zero;
                dir(i) = 1.0f;
                std::vector<IntersectionPoint<DIM>> is;
                int hits = this->intersectBoundaryAllHits(x, zero, dir, std::numeric_limits<float>::max(),
                                                          false, false, is);
                isInside = isInside && (hits%2 == 1);
            }

            return isInside;
        }

        return this->computeDistToBoundary(x, true) < 0.0f;
    };
    outsideBoundingDomain = [this](const Vector<DIM>& x) -> bool {
        return (x.array() < this->domainMin.array()).any() ||
               (x.array() > this->domainMax.array()).any();
    };
    computeAbsorbingBoundarySignedVolume = []() -> float {
        return 0.0f;
    };
    computeReflectingBoundarySignedVolume = []() -> float {
        return 0.0f;
    };
    computeDomainSignedVolume = [this]() -> float {
        return this->computeAbsorbingBoundarySignedVolume() +
               this->computeReflectingBoundarySignedVolume();
    };
}

} // zombie

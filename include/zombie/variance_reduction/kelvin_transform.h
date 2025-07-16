// This file enables solving Poisson equations with Dirichlet, Neumann, and/or
// Robin boundary conditions in an exterior domain using the Kelvin transform.
// This transformation maps the exterior of a watertight domain to the interior
// of an inverted domain, ensuring that random walks terminate on the boundary
// of the inverted domain, rather than escaping to infinity in the exterior domain.
//
// The KelvinTransform class provides methods for performing the inversion, and
// setting up the PDE on the inverted domain. The user can then run any solver
// on the inverted domain, and map the solution and gradient estimates back
// to the exterior domain via the transform*Estimate methods.
//
// NOTE: the implementation assumes that the origin is contained within the
// domain boundary, and that the boundary normals are outward facing relative
// to the exterior domain.
//
// Resources:
// - Kelvin Transformations for Simulations on Infinite Domains [2021]

#pragma once

#include <zombie/core/geometry_helpers.h>
#include <zombie/core/pde.h>
#include <vector>

namespace zombie {

template <typename T, size_t DIM>
class KelvinTransform {
public:
    // constructor
    KelvinTransform(const Vector<DIM>& origin_=Vector<DIM>::Zero());

    // set and get origin
    void setOrigin(const Vector<DIM>& origin_);
    Vector<DIM> getOrigin() const;

    // applies the Kelvin transform to a point in the exterior domain
    Vector<DIM> transformPoint(const Vector<DIM>& x) const;

    // applies the Kelvin transform to a normal vector at a point
    // in the exterior domain
    Vector<DIM> transformNormal(const Vector<DIM>& x,
                                const Vector<DIM>& n) const;

    // sets up the PDE for the inverted domain given the PDE
    // for the exterior domain
    void transformPde(const PDE<T, DIM>& pdeExteriorDomain,
                      PDE<T, DIM>& pdeInvertedDomain) const;

    // returns the estimated solution in the exterior domain,
    // given the solution estimate at a transformed point
    T transformSolutionEstimate(const T& V,
                                const Vector<DIM>& y) const;

    // returns the estimated gradient in the exterior domain,
    // given solution and gradient estimates at a transformed point
    void transformGradientEstimate(const T& V,
                                   const std::vector<T>& dV,
                                   const Vector<DIM>& y,
                                   std::vector<T>& gradient) const;

    ///////////////////////////////////////////////////////////////////////////////////////////////
    // helper functions for 2D line segment meshes and 3D triangle meshes
    // NOTE: The discretization might require refinement for accurate results,
    // as a spherical inversion maps planar boundaries to curved boundaries.

    // applies the Kelvin transform to a set of points in the exterior domain
    void transformPoints(const std::vector<Vector<DIM>>& points,
                         std::vector<Vector<DIM>>& transformedPoints) const;

    // computes the modified Robin coefficients for the transformed reflecting boundary:
    // in 3D, a boundary with Neumann conditions typically has non-zero Robin coefficients
    // on the inverted domain, whereas in 2D it continues to have Neumann conditions
    void computeRobinCoefficients(const std::vector<Vector<DIM>>& transformedPoints,
                                  const std::vector<Vectori<DIM>>& indices,
                                  const std::vector<float>& minRobinCoeffValues,
                                  const std::vector<float>& maxRobinCoeffValues,
                                  std::vector<float>& transformedMinRobinCoeffValues,
                                  std::vector<float>& transformedMaxRobinCoeffValues) const;

protected:
    // member
    Vector<DIM> origin;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline KelvinTransform<T, DIM>::KelvinTransform(const Vector<DIM>& origin_):
origin(origin_)
{
    // do nothing
}

template <typename T, size_t DIM>
inline void KelvinTransform<T, DIM>::setOrigin(const Vector<DIM>& origin_)
{
    origin = origin_;
}

template <typename T, size_t DIM>
inline Vector<DIM> KelvinTransform<T, DIM>::getOrigin() const
{
    return origin;
}

template <typename T, size_t DIM>
inline Vector<DIM> KelvinTransform<T, DIM>::transformPoint(const Vector<DIM>& x) const
{
    Vector<DIM> xShifted = x - origin;
    return xShifted/xShifted.squaredNorm();
}

template <typename T, size_t DIM>
inline Vector<DIM> KelvinTransform<T, DIM>::transformNormal(const Vector<DIM>& x,
                                                            const Vector<DIM>& n) const
{
    Vector<DIM> xShifted = x - origin;
    return n - 2.0f*n.dot(xShifted)*xShifted/xShifted.squaredNorm();
}

template <typename T, size_t DIM>
inline void KelvinTransform<T, DIM>::transformPoints(const std::vector<Vector<DIM>>& points,
                                                     std::vector<Vector<DIM>>& transformedPoints) const
{
    int nPoints = (int)points.size();
    transformedPoints.resize(nPoints);

    for (int i = 0; i < nPoints; i++) {
        transformedPoints[i] = transformPoint(points[i]);
    }
}

template <size_t DIM>
inline float computeClosestPointOnPrimitive(const std::vector<Vector<DIM>>& positions,
                                            const std::vector<Vectori<DIM>>& indices,
                                            int elementIndex, const Vector<DIM>& x,
                                            Vector<DIM>& pt)
{
    std::cerr << "computeClosestPointOnPrimitive: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);

    return 0.0f;
}

template <>
inline float computeClosestPointOnPrimitive<2>(const std::vector<Vector2>& positions,
                                               const std::vector<Vector2i>& indices,
                                               int elementIndex, const Vector2& x,
                                               Vector2& pt)
{
    const Vector2& pa = positions[indices[elementIndex][0]];
    const Vector2& pb = positions[indices[elementIndex][1]];

    return computeClosestPointOnLineSegment(pa, pb, x, pt);
}

template <>
inline float computeClosestPointOnPrimitive<3>(const std::vector<Vector3>& positions,
                                               const std::vector<Vector3i>& indices,
                                               int elementIndex, const Vector3& x,
                                               Vector3& pt)
{
    const Vector3& pa = positions[indices[elementIndex][0]];
    const Vector3& pb = positions[indices[elementIndex][1]];
    const Vector3& pc = positions[indices[elementIndex][2]];

    return computeClosestPointOnTriangle(pa, pb, pc, x, pt);
}

template <size_t DIM>
inline float computeFarthestPointOnPrimitive(const std::vector<Vector<DIM>>& positions,
                                             const std::vector<Vectori<DIM>>& indices,
                                             int elementIndex, const Vector<DIM>& x,
                                             Vector<DIM>& pt)
{
    std::cerr << "computeFarthestPointOnPrimitive: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);

    return 0.0f;
}

template <>
inline float computeFarthestPointOnPrimitive<2>(const std::vector<Vector2>& positions,
                                                const std::vector<Vector2i>& indices,
                                                int elementIndex, const Vector2& x,
                                                Vector2& pt)
{
    const Vector2& pa = positions[indices[elementIndex][0]];
    const Vector2& pb = positions[indices[elementIndex][1]];

    return computeFarthestPointOnLineSegment(pa, pb, x, pt);
}

template <>
inline float computeFarthestPointOnPrimitive<3>(const std::vector<Vector3>& positions,
                                                const std::vector<Vector3i>& indices,
                                                int elementIndex, const Vector3& x,
                                                Vector3& pt)
{
    const Vector3& pa = positions[indices[elementIndex][0]];
    const Vector3& pb = positions[indices[elementIndex][1]];
    const Vector3& pc = positions[indices[elementIndex][2]];

    return computeFarthestPointOnTriangle(pa, pb, pc, x, pt);
}

template <size_t DIM>
inline Vector<DIM> computePrimitiveNormal(const std::vector<Vector<DIM>>& positions,
                                          const std::vector<Vectori<DIM>>& indices,
                                          int elementIndex)
{
    std::cerr << "computePrimitiveNormal: Unsupported dimension: " << DIM << std::endl;
    exit(EXIT_FAILURE);

    return Vector<DIM>::Zero();
}

template <>
inline Vector2 computePrimitiveNormal<2>(const std::vector<Vector2>& positions,
                                         const std::vector<Vector2i>& indices,
                                         int elementIndex)
{
    const Vector2& pa = positions[indices[elementIndex][0]];
    const Vector2& pb = positions[indices[elementIndex][1]];

    return computeLineSegmentNormal(pa, pb, true);
}

template <>
inline Vector3 computePrimitiveNormal<3>(const std::vector<Vector3>& positions,
                                         const std::vector<Vector3i>& indices,
                                         int elementIndex)
{
    const Vector3& pa = positions[indices[elementIndex][0]];
    const Vector3& pb = positions[indices[elementIndex][1]];
    const Vector3& pc = positions[indices[elementIndex][2]];

    return computeTriangleNormal(pa, pb, pc, true);
}

template <typename T, size_t DIM>
inline void KelvinTransform<T, DIM>::computeRobinCoefficients(const std::vector<Vector<DIM>>& transformedPoints,
                                                              const std::vector<Vectori<DIM>>& indices,
                                                              const std::vector<float>& minRobinCoeffValues,
                                                              const std::vector<float>& maxRobinCoeffValues,
                                                              std::vector<float>& transformedMinRobinCoeffValues,
                                                              std::vector<float>& transformedMaxRobinCoeffValues) const
{
    int nIndices = (int)indices.size();
    transformedMinRobinCoeffValues.resize(nIndices, 0.0f);
    transformedMaxRobinCoeffValues.resize(nIndices, 0.0f);
    float epsilon = std::numeric_limits<float>::epsilon();
    Vector<DIM> zero = Vector<DIM>::Zero();

    for (int i = 0; i < nIndices; i++) {
        Vector<DIM> closestPt, farthestPt;
        float distToClosestPt = computeClosestPointOnPrimitive<DIM>(transformedPoints, indices,
                                                                    i, zero, closestPt);
        float distToFarthestPt = computeFarthestPointOnPrimitive<DIM>(transformedPoints, indices,
                                                                      i, zero, farthestPt);
        float minShiftValue = 0.0f;
        float maxShiftValue = 0.0f;
        if (DIM == 3) {
            Vector<DIM> normal = computePrimitiveNormal<DIM>(transformedPoints, indices, i);
            float dotClosest = normal.dot(closestPt);
            float dotFarthest = normal.dot(farthestPt);
            minShiftValue = std::min(dotClosest, dotFarthest);
            maxShiftValue = std::max(dotClosest, dotFarthest);
        }

        float minValue = (minRobinCoeffValues[i] + minShiftValue)/(distToFarthestPt*distToFarthestPt);
        float maxValue = (maxRobinCoeffValues[i] + maxShiftValue)/(distToClosestPt*distToClosestPt);
        transformedMinRobinCoeffValues[i] = minValue*maxValue > epsilon ?
                                            std::min(std::fabs(minValue), std::fabs(maxValue)) : 0.0f;
        transformedMaxRobinCoeffValues[i] = std::max(std::fabs(minValue), std::fabs(maxValue));
    }
}

template <typename T, size_t DIM>
inline void KelvinTransform<T, DIM>::transformPde(const PDE<T, DIM>& pdeExteriorDomain,
                                                  PDE<T, DIM>& pdeInvertedDomain) const
{
    if (pdeExteriorDomain.absorptionCoeff > 0.0f) {
        std::cout << "KelvinTransform::transformPde(): non-zero absorption coefficients are not supported" << std::endl;
        exit(EXIT_FAILURE);
    }

    pdeInvertedDomain.source = [pdeExteriorDomain, this](const Vector<DIM>& y) -> T {
        float r2 = y.squaredNorm();
        Vector<DIM> x = y/r2 + this->origin;
        float rPow = r2*r2;
        if (DIM == 3) {
            rPow *= std::sqrt(r2);
        }

        return pdeExteriorDomain.source(x)/rPow;
    };
    pdeInvertedDomain.dirichlet = [pdeExteriorDomain, this](const Vector<DIM>& y,
                                                            bool _) -> T {
        float r2 = y.squaredNorm();
        Vector<DIM> x = y/r2 + this->origin;
        float rPow = 1.0f;
        if (DIM == 3) {
            rPow *= std::sqrt(r2);
        }

        return pdeExteriorDomain.dirichlet(x, false)/rPow;
    };
    pdeInvertedDomain.robin = [pdeExteriorDomain, this](const Vector<DIM>& y,
                                                        const Vector<DIM>& N,
                                                        bool _) -> T {
        float r2 = y.squaredNorm();
        Vector<DIM> x = y/r2 + this->origin;
        Vector<DIM> n = N - 2.0f*N.dot(y)*y/r2;
        float rPow = r2;
        if (DIM == 3) {
            rPow *= std::sqrt(r2);
        }

        return pdeExteriorDomain.robin(x, n, false)/rPow;
    };
    pdeInvertedDomain.robinCoeff = [pdeExteriorDomain, this](const Vector<DIM>& y,
                                                             const Vector<DIM>& N,
                                                             bool _) -> float {
        float r2 = y.squaredNorm();
        Vector<DIM> x = y/r2 + this->origin;
        float P = N.dot(y)/r2;
        float shift = DIM == 3 ? P : 0.0f;
        Vector<DIM> n = N - 2.0f*P*y;

        return pdeExteriorDomain.robinCoeff(x, n, false)/r2 + shift;
    };
    pdeInvertedDomain.hasReflectingBoundaryConditions = [pdeExteriorDomain, this](
                                                        const Vector<DIM>& y) -> bool {
        float r2 = y.squaredNorm();
        Vector<DIM> x = y/r2 + this->origin;

        return pdeExteriorDomain.hasReflectingBoundaryConditions(x);
    };
    pdeInvertedDomain.areRobinConditionsPureNeumann =
        DIM == 2 ? pdeExteriorDomain.areRobinConditionsPureNeumann : false;
    pdeInvertedDomain.areRobinCoeffsNonnegative =
        DIM == 2 ? pdeExteriorDomain.areRobinCoeffsNonnegative : false;
    pdeInvertedDomain.absorptionCoeff = 0.0f;
}

template <typename T, size_t DIM>
inline T KelvinTransform<T, DIM>::transformSolutionEstimate(const T& V,
                                                            const Vector<DIM>& y) const
{
    float rPow = DIM == 3 ? y.norm() : 1.0f;
    return rPow*V;
}

template <typename T, size_t DIM>
inline void KelvinTransform<T, DIM>::transformGradientEstimate(const T& V,
                                                               const std::vector<T>& dV,
                                                               const Vector<DIM>& y,
                                                               std::vector<T>& gradient) const
{
    std::vector<T> dU(DIM);
    float r2 = y.squaredNorm();
    float rPow = 1.0f;
    if (DIM == 3) {
        rPow *= std::sqrt(r2);
    }

    for (int i = 0; i < DIM; i++) {
        dU[i] = rPow*r2*dV[i];
        if (DIM == 3) {
            dU[i] += rPow*y(i)*V;
        }
    }

    gradient.resize(DIM);
    for (int i = 0; i < DIM; i++) {
        gradient[i] = T(0.0f);

        for (int j = 0; j < DIM; j++) {
            float R = -2.0f*y(i)*y(j)/r2;
            if (i == j) R += 1.0f;
            gradient[i] += R*dU[j];
        }
    }
}

} // zombie

// This file implements a dense sdf grid.

#pragma once

#include <zombie/core/geometric_queries.h>
#include <zombie/utils/dense_grid.h>

namespace zombie {

template <size_t DIM>
class SdfGrid: public DenseGrid<float, 1, DIM> {
public:
    // constructors
    SdfGrid(const Vector<DIM>& gridMin, const Vector<DIM>& gridMax);
    SdfGrid(const Eigen::VectorXf& sdfData, const Vectori<DIM>& gridShape,
            const Vector<DIM>& gridMin, const Vector<DIM>& gridMax);
    SdfGrid(std::function<Array<float, 1>(const Vector<DIM>&)> sdfDataCallback,
            const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
            const Vector<DIM>& gridMax);

    // utility functions
    Vector<DIM> computeGradient(const Vector<DIM>& x) const;
    Vector<DIM> computeNormal(const Vector<DIM>& x) const;
    void projectToZeroLevelSet(Vector<DIM>& x, Vector<DIM>& normal,
                               int maxIterations=8, float epsilon=1e-5f) const;
    bool intersectZeroLevelSet(const Vector<DIM>& origin, const Vector<DIM>& dir,
                               float tMax, IntersectionPoint<DIM>& intersectionPt,
                               int maxIterations=128, float epsilon=1e-5f) const;

private:
    // clamps the given point to the grid bounds
    Vector<DIM> clampInGrid(const Vector<DIM>& x) const;
};

enum class SdfOperation {
    Union,
    Intersection,
    Difference
};

template <typename SdfLeft, typename SdfRight, size_t DIM>
class SdfHierarchy {
public:
    // constructor
    SdfHierarchy(const SdfLeft& left_, const SdfRight& right_, SdfOperation operation_);

    // accessors
    Array<float, 1> operator()(const Vector<DIM>& x) const;

    // members
    const SdfLeft& left;
    const SdfRight& right;
    SdfOperation operation;
};

template <typename Sdf, size_t DIM>
void populateGeometricQueriesForDirichletBoundary(const Sdf& sdf, GeometricQueries<DIM>& geometricQueries);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <size_t DIM>
inline SdfGrid<DIM>::SdfGrid(const Vector<DIM>& gridMin, const Vector<DIM>& gridMax):
DenseGrid<float, 1, DIM>(gridMin, gridMax, true)
{
    // do nothing
}

template <size_t DIM>
inline SdfGrid<DIM>::SdfGrid(const Eigen::VectorXf& sdfData, const Vectori<DIM>& gridShape,
                             const Vector<DIM>& gridMin, const Vector<DIM>& gridMax):
DenseGrid<float, 1, DIM>(sdfData, gridShape, gridMin, gridMax, true)
{
    // do nothing
}

template <size_t DIM>
inline SdfGrid<DIM>::SdfGrid(std::function<Array<float, 1>(const Vector<DIM>&)> sdfDataCallback,
                             const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
                             const Vector<DIM>& gridMax):
DenseGrid<float, 1, DIM>(sdfDataCallback, gridShape, gridMin, gridMax, true)
{
    // do nothing
}

template <size_t DIM>
inline Vector<DIM> SdfGrid<DIM>::clampInGrid(const Vector<DIM>& x) const
{
    Vector<DIM> xClamped = x;
    for (int i = 0; i < DIM; i++) {
        float gridMin = this->origin[i] + 1e-6f;
        float gridMax = this->origin[i] + this->extent[i] - 1e-6f;
        xClamped[i] = std::clamp(xClamped[i], gridMin, gridMax);
    }

    return xClamped;
}

template <size_t DIM>
inline Vector<DIM> SdfGrid<DIM>::computeGradient(const Vector<DIM>& x) const
{
    // perform central differencing
    Vector<DIM> gradient = Vector<DIM>::Zero();
    for (int i = 0; i < DIM; i++) {
        float spacing = this->extent[i]/static_cast<float>(this->shape[i]);
        Vector<DIM> offset = Vector<DIM>::Zero();
        offset[i] = spacing;
        float sdfMinus = (*this)(clampInGrid(x - offset))(0);
        float sdfPlus = (*this)(clampInGrid(x + offset))(0);
        gradient[i] = (sdfPlus - sdfMinus)/(2.0f*spacing);
    }

    return gradient;
}

template <size_t DIM>
inline Vector<DIM> SdfGrid<DIM>::computeNormal(const Vector<DIM>& x) const
{
    Vector<DIM> dSdf = computeGradient(x);
    dSdf /= (dSdf.norm() + 1e-6f);

    return dSdf;
}

template <size_t DIM>
inline void SdfGrid<DIM>::projectToZeroLevelSet(Vector<DIM>& x, Vector<DIM>& normal,
                                                int maxIterations, float epsilon) const
{
    // apply steepest descent to project x to the zero level set
    for (int i = 0; i < maxIterations; i++) {
        float sdf = (*this)(x)(0);
        if (std::fabs(sdf) < epsilon) break;
        Vector<DIM> dSdf = computeGradient(x);
        Vector<DIM> step = dSdf*sdf/(dSdf.norm() + 1e-6f);
        x = clampInGrid(x - step);
    }

    normal = computeNormal(x);
}

template <size_t DIM>
inline bool SdfGrid<DIM>::intersectZeroLevelSet(const Vector<DIM>& origin, const Vector<DIM>& dir,
                                                float tMax, IntersectionPoint<DIM>& intersectionPt,
                                                int maxIterations, float epsilon) const
{
    // first intersect whether the ray intersects the grid bounds
    Vector<DIM> bMin = this->origin;
    Vector<DIM> bMax = this->origin + this->extent;
    Vector<DIM> invDir = dir.cwiseInverse();
    Vector<DIM> t0 = (bMin - origin).cwiseProduct(invDir);
    Vector<DIM> t1 = (bMax - origin).cwiseProduct(invDir);
    Vector<DIM> tNear = t0.cwiseMin(t1);
    Vector<DIM> tFar = t0.cwiseMax(t1);
    float tEnter = std::max(0.0f, tNear.maxCoeff());
    float tExit = std::min(tMax, tFar.minCoeff());
    if (tEnter > tExit) return false; // ray misses the grid entirely

    // sphere trace sdf field
    float tPrev = tEnter;
    float sdfPrev = 0.0f;
    float t = tPrev;

    for (int i = 0; i < maxIterations && t <= tExit; i++) {
        Vector<DIM> p = origin + dir*t;
        float sdf = (*this)(p)(0);

        // check if the ray intersects the zero level set
        if (std::fabs(sdf) < epsilon) {
            intersectionPt.pt = p;
            intersectionPt.normal = computeNormal(p);
            intersectionPt.dist = t;
            return true;
        }

        // perform a single secant refinement step if sign of sdf changes
        if (sdfPrev*sdf < 0.0f) {
            float alpha = sdfPrev/(sdfPrev - sdf);
            float tHit = tPrev + alpha*(t - tPrev);
            Vector<DIM> pHit = origin + dir*tHit;
            float sdfHit = (*this)(pHit)(0);
            Vector<DIM> dSdfHit = computeGradient(pHit);
            Vector<DIM> step = dSdfHit*sdfHit/(dSdfHit.norm() + 1e-6f);
            pHit -= step;

            intersectionPt.pt = pHit;
            intersectionPt.normal = computeNormal(pHit);
            intersectionPt.dist = (pHit - origin).dot(dir);
            return true;
        }

        // step forward
        tPrev = t;
        sdfPrev = sdf;
        t += std::fabs(sdf);
    }

    return false;
}

template <typename SdfLeft, typename SdfRight, size_t DIM>
inline SdfHierarchy<SdfLeft, SdfRight, DIM>::SdfHierarchy(const SdfLeft& left_,
                                                          const SdfRight& right_,
                                                          SdfOperation operation_):
left(left_),
right(right_),
operation(operation_)
{
    // do nothing
}

template <typename SdfLeft, typename SdfRight, size_t DIM>
inline Array<float, 1> SdfHierarchy<SdfLeft, SdfRight, DIM>::operator()(const Vector<DIM>& x) const
{
    Array<float, 1> leftSdf = left(x);
    Array<float, 1> rightSdf = right(x);

    if (operation == SdfOperation::Union) {
        return leftSdf.min(rightSdf);

    } else if (operation == SdfOperation::Intersection) {
        return leftSdf.max(rightSdf);

    } else if (operation == SdfOperation::Difference) {
        return leftSdf.max(-rightSdf);

    } else {
        std::cerr << "SdfHierarchy::operator(): Invalid operation." << std::endl;
        exit(EXIT_FAILURE);
    }

    return Array<float, 1>::Zero();
}

template <typename Sdf, size_t DIM>
void populateGeometricQueriesForDirichletBoundary(const Sdf& sdf, GeometricQueries<DIM>& geometricQueries)
{
    geometricQueries.hasNonEmptyAbsorbingBoundary = true;
    geometricQueries.computeDistToAbsorbingBoundary = [&sdf](const Vector<DIM>& x,
                                                             bool computeSignedDistance) -> float {
        return computeSignedDistance ? sdf(x)(0) : std::fabs(sdf(x)(0));
    };
    if constexpr (std::is_same<Sdf, SdfGrid<DIM>>::value) {
        geometricQueries.projectToAbsorbingBoundary = [&sdf](Vector<DIM>& x, Vector<DIM>& normal,
                                                             float& distance, bool computeSignedDistance) -> bool {
            distance = computeSignedDistance ? sdf(x)(0) : std::fabs(sdf(x)(0));
            sdf.projectToZeroLevelSet(x, normal);
            return true;
        };
        geometricQueries.intersectAbsorbingBoundary = [&sdf](const Vector<DIM>& origin, const Vector<DIM>& normal,
                                                             const Vector<DIM>& dir, float tMax, bool OnAbsorbingBoundary,
                                                             IntersectionPoint<DIM>& intersectionPt) -> bool {
            return sdf.intersectZeroLevelSet(origin, dir, tMax, intersectionPt);
        };

    } else {
        geometricQueries.projectToAbsorbingBoundary = [&sdf](Vector<DIM>& x, Vector<DIM>& normal,
                                                             float& distance, bool computeSignedDistance) -> bool {
            distance = computeSignedDistance ? sdf(x)(0) : std::fabs(sdf(x)(0));
            return false; // return false since x and normal are not updated
        };
    }
}

} // zombie

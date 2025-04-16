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
    geometricQueries.projectToAbsorbingBoundary = [&sdf](Vector<DIM>& x, Vector<DIM>& normal,
                                                         float& distance, bool computeSignedDistance) -> bool {
        distance = computeSignedDistance ? sdf(x)(0) : std::fabs(sdf(x)(0));
        return false; // return false since x and normal are not updated
    };
}

} // zombie

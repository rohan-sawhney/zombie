// This file implements a dense grid data structure for storing and querying data.
// Interpolation is supported for floating-point data types.

#pragma once

#include <algorithm>
#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "oneapi/tbb/parallel_for.h"

namespace zombie {

template <typename T, size_t CHANNELS>
using Array = Eigen::Array<T, CHANNELS, 1>;

template <typename T, size_t CHANNELS, size_t DIM>
class DenseGrid {
public:
    // constructors
    DenseGrid(const Vector<DIM>& gridMin, const Vector<DIM>& gridMax,
              bool enableInterpolation=false);
    DenseGrid(const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& gridData,
              const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
              const Vector<DIM>& gridMax, bool enableInterpolation=false);
    DenseGrid(std::function<Array<T, CHANNELS>(const Vector<DIM>&)> gridDataCallback,
              const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
              const Vector<DIM>& gridMax, bool enableInterpolation=false);

    // setters
    void set(const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& gridData,
             const Vectori<DIM>& gridShape);
    void set(std::function<Array<T, CHANNELS>(const Vector<DIM>&)> gridDataCallback,
             const Vectori<DIM>& gridShape);
    void set(int index, const Array<T, CHANNELS>& value);
    void set(const Vectori<DIM>& index, const Array<T, CHANNELS>& value);

    // getters
    Vector<DIM> get(int index) const;
    Vector<DIM> get(const Vectori<DIM>& index) const;

    // data accessors
    Array<T, CHANNELS> operator()(const Vector<DIM>& x) const;
    Array<T, CHANNELS> min() const;
    Array<T, CHANNELS> max() const;

    // members
    Eigen::Matrix<T, Eigen::Dynamic, CHANNELS> data;
    Vectori<DIM> shape;
    Vector<DIM> origin;
    Vector<DIM> extent;

protected:
    // returns local coordinates
    Vector<DIM> getLocalCoordinates(const Vector<DIM>& x) const;

    // member
    bool interpolationEnabled;
};

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&)> getDenseGridCallback0(
    const DenseGrid<S, CHANNELS, DIM>& grid);
template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&)> getDenseGridCallback0(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation=false);

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback1(
    const DenseGrid<S, CHANNELS, DIM>& grid);
template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback1(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation=false);

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback2(
    const DenseGrid<S, CHANNELS, DIM>& grid,
    const DenseGrid<S, CHANNELS, DIM>& gridBoundaryNormalAligned);
template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback2(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridDataBoundaryNormalAligned,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation=false);

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback3(
    const DenseGrid<S, CHANNELS, DIM>& grid);
template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback3(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation=false);

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback4(
    const DenseGrid<S, CHANNELS, DIM>& grid,
    const DenseGrid<S, CHANNELS, DIM>& gridBoundaryNormalAligned);
template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback4(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridDataBoundaryNormalAligned,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation=false);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t CHANNELS, size_t DIM>
inline DenseGrid<T, CHANNELS, DIM>::DenseGrid(const Vector<DIM>& gridMin, const Vector<DIM>& gridMax,
                                              bool enableInterpolation):
                                              shape(Vectori<DIM>::Zero()),
                                              origin(gridMin), extent(gridMax - gridMin),
                                              interpolationEnabled(enableInterpolation)
{
    // do nothing
}

template <typename T, size_t CHANNELS, size_t DIM>
inline DenseGrid<T, CHANNELS, DIM>::DenseGrid(const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& gridData,
                                              const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
                                              const Vector<DIM>& gridMax, bool enableInterpolation):
                                              origin(gridMin), extent(gridMax - gridMin),
                                              interpolationEnabled(enableInterpolation)
{
    set(gridData, gridShape);
}

template <typename T, size_t CHANNELS, size_t DIM>
inline DenseGrid<T, CHANNELS, DIM>::DenseGrid(std::function<Array<T, CHANNELS>(const Vector<DIM>&)> gridDataCallback,
                                              const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
                                              const Vector<DIM>& gridMax, bool enableInterpolation):
                                              origin(gridMin), extent(gridMax - gridMin),
                                              interpolationEnabled(enableInterpolation)
{
    set(gridDataCallback, gridShape);
}

template <typename T, size_t CHANNELS, size_t DIM>
inline void DenseGrid<T, CHANNELS, DIM>::set(const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& gridData,
                                             const Vectori<DIM>& gridShape)
{
    // check data consistency
    if (gridData.rows() != gridShape.prod()) {
        std::cerr << "DenseGrid::set(): Number of rows in data (" << gridData.rows()
                  << ") does not match the product of gridShape (" << gridShape.prod() << ")."
                  << std::endl;
        exit(EXIT_FAILURE);
    }

    // set data and shape
    data = gridData;
    shape = gridShape;
}

template <typename T, size_t CHANNELS, size_t DIM>
inline void DenseGrid<T, CHANNELS, DIM>::set(std::function<Array<T, CHANNELS>(const Vector<DIM>&)> gridDataCallback,
                                             const Vectori<DIM>& gridShape)
{
    // resize the data vector
    int gridSize = gridShape.prod();
    data = Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>::Zero(gridSize, CHANNELS);
    shape = gridShape;

    // populate the grid using the data callback
    auto run = [&](const tbb::blocked_range<int>& range) {
        for (int i = range.begin(); i < range.end(); ++i) {
            // compute the world-space position for this index
            Vector<DIM> x = get(i);

            // evaluate the callback at the computed position
            data.row(i) = gridDataCallback(x);
        }
    };

    tbb::blocked_range<int> range(0, gridSize);
    tbb::parallel_for(range, run);
}

template <typename T, size_t CHANNELS, size_t DIM>
inline void DenseGrid<T, CHANNELS, DIM>::set(int index, const Array<T, CHANNELS>& value)
{
    if (index < 0 || index >= shape.prod()) {
        std::cerr << "DenseGrid::set(): Index out of bounds." << std::endl;
        exit(EXIT_FAILURE);
    }

    data.row(index) = value;
}

template <typename T, size_t CHANNELS, size_t DIM>
inline void DenseGrid<T, CHANNELS, DIM>::set(const Vectori<DIM>& index, const Array<T, CHANNELS>& value)
{
    // compute the linear index
    size_t flatIndex = 0;
    size_t stride = 1;
    for (int i = DIM - 1; i >= 0; i--) {
        if (index[i] < 0 || index[i] >= shape[i]) {
            std::cerr << "DenseGrid::set(): Index out of bounds." << std::endl;
            exit(EXIT_FAILURE);
        }

        flatIndex += index[i]*stride;
        stride *= shape[i];
    }

    // set the value
    data.row(flatIndex) = value;
}

template <typename T, size_t CHANNELS, size_t DIM>
inline Vector<DIM> DenseGrid<T, CHANNELS, DIM>::get(int index) const
{
    if (index < 0 || index >= shape.prod()) {
        std::cerr << "DenseGrid::get(): Index out of bounds." << std::endl;
        exit(EXIT_FAILURE);
    }

    Vector<DIM> x = Vector<DIM>::Zero();
    for (int i = DIM - 1; i >= 0; i--) {
        float spacing = (shape[i] == 0) ? 0.0f : extent[i]/static_cast<float>(shape[i]);
        x[i] = origin[i] + (static_cast<float>(index%shape[i]) + 0.5f)*spacing;
        index /= shape[i];
    }

    return x;
}

template <typename T, size_t CHANNELS, size_t DIM>
inline Vector<DIM> DenseGrid<T, CHANNELS, DIM>::get(const Vectori<DIM>& index) const
{
    Vector<DIM> x = Vector<DIM>::Zero();
    for (int i = 0; i < DIM; i++) {
        if (index[i] < 0 || index[i] >= shape[i]) {
            std::cerr << "DenseGrid::get(): Index out of bounds." << std::endl;
            exit(EXIT_FAILURE);
        }

        float spacing = (shape[i] == 0) ? 0.0f : extent[i]/static_cast<float>(shape[i]);
        x[i] = origin[i] + (static_cast<float>(index[i]) + 0.5f)*spacing;
    }

    return x;
}

template <typename T, size_t CHANNELS>
inline Array<T, CHANNELS> interpolate(const DenseGrid<T, CHANNELS, 2>& grid,
                                      const Vector2& xLocal)
{
    // perform bilinear interpolation
    const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& data = grid.data;
    const Vector2i& shape = grid.shape;

    float u = shape[0] == 1 ? 0.0f : xLocal[0]*shape[0] - 0.5f;
    float v = shape[1] == 1 ? 0.0f : xLocal[1]*shape[1] - 0.5f;
    int i0 = std::clamp(static_cast<int>(std::floor(u)), 0, shape[0] - 1);
    int i1 = std::clamp(i0 + 1, 0, shape[0] - 1);
    int j0 = std::clamp(static_cast<int>(std::floor(v)), 0, shape[1] - 1);
    int j1 = std::clamp(j0 + 1, 0, shape[1] - 1);

    Array<T, CHANNELS> f00 = data.row(i0*shape[1] + j0);
    Array<T, CHANNELS> f01 = data.row(i0*shape[1] + j1);
    Array<T, CHANNELS> f10 = data.row(i1*shape[1] + j0);
    Array<T, CHANNELS> f11 = data.row(i1*shape[1] + j1);
    float x = u - static_cast<float>(i0);
    float y = v - static_cast<float>(j0);

    return f00*(1.0f - x)*(1.0f - y) + f10*x*(1.0f - y) + f01*(1.0f - x)*y + f11*x*y;
}

template <typename T, size_t CHANNELS>
inline Array<T, CHANNELS> interpolate(const DenseGrid<T, CHANNELS, 3>& grid,
                                      const Vector3& xLocal)
{
    // perform trilinear interpolation
    const Eigen::Matrix<T, Eigen::Dynamic, CHANNELS>& data = grid.data;
    const Vector3i& shape = grid.shape;

    float u = shape[0] == 1 ? 0.0f : xLocal[0]*shape[0] - 0.5f;
    float v = shape[1] == 1 ? 0.0f : xLocal[1]*shape[1] - 0.5f;
    float w = shape[2] == 1 ? 0.0f : xLocal[2]*shape[2] - 0.5f;
    int i0 = std::clamp(static_cast<int>(std::floor(u)), 0, shape[0] - 1);
    int i1 = std::clamp(i0 + 1, 0, shape[0] - 1);
    int j0 = std::clamp(static_cast<int>(std::floor(v)), 0, shape[1] - 1);
    int j1 = std::clamp(j0 + 1, 0, shape[1] - 1);
    int k0 = std::clamp(static_cast<int>(std::floor(w)), 0, shape[2] - 1);
    int k1 = std::clamp(k0 + 1, 0, shape[2] - 1);

    Array<T, CHANNELS> f000 = data.row((i0*shape[1] + j0)*shape[2] + k0);
    Array<T, CHANNELS> f001 = data.row((i0*shape[1] + j0)*shape[2] + k1);
    Array<T, CHANNELS> f010 = data.row((i0*shape[1] + j1)*shape[2] + k0);
    Array<T, CHANNELS> f011 = data.row((i0*shape[1] + j1)*shape[2] + k1);
    Array<T, CHANNELS> f100 = data.row((i1*shape[1] + j0)*shape[2] + k0);
    Array<T, CHANNELS> f101 = data.row((i1*shape[1] + j0)*shape[2] + k1);
    Array<T, CHANNELS> f110 = data.row((i1*shape[1] + j1)*shape[2] + k0);
    Array<T, CHANNELS> f111 = data.row((i1*shape[1] + j1)*shape[2] + k1);
    float x = u - static_cast<float>(i0);
    float y = v - static_cast<float>(j0);
    float z = w - static_cast<float>(k0);

    return f000*(1.0f - x)*(1.0f - y)*(1.0f - z) + f100*x*(1.0f - y)*(1.0f - z) +
           f010*(1.0f - x)*y*(1.0f - z) + f110*x*y*(1.0f - z) +
           f001*(1.0f - x)*(1.0f - y)*z + f101*x*(1.0f - y)*z +
           f011*(1.0f - x)*y*z + f111*x*y*z;
}

template <typename T, size_t CHANNELS, size_t DIM>
inline Array<T, CHANNELS> DenseGrid<T, CHANNELS, DIM>::operator()(const Vector<DIM>& x) const
{
    // convert input point to local coordinates
    Vector<DIM> xLocal = getLocalCoordinates(x);

    if constexpr (std::is_floating_point<T>::value) {
        // perform interpolation if enabled
        if (interpolationEnabled) {
            return interpolate(*this, xLocal);
        }
    }

    // perform nearest neighbor (texel centre) lookup
    size_t flatIndex = 0;
    size_t stride = 1;
    for (int i = DIM - 1; i >= 0; i--) {
        int index = std::clamp(static_cast<int>(std::floor(xLocal[i]*shape[i])), 0, shape[i] - 1);
        flatIndex += index*stride;
        stride *= shape[i];
    }

    return data.row(flatIndex);
}

template <typename T, size_t CHANNELS, size_t DIM>
inline Array<T, CHANNELS> DenseGrid<T, CHANNELS, DIM>::min() const
{
    return data.colwise().minCoeff().transpose().array();
}

template <typename T, size_t CHANNELS, size_t DIM>
inline Array<T, CHANNELS> DenseGrid<T, CHANNELS, DIM>::max() const
{
    return data.colwise().maxCoeff().transpose().array();
}

template <typename T, size_t CHANNELS, size_t DIM>
inline Vector<DIM> DenseGrid<T, CHANNELS, DIM>::getLocalCoordinates(const Vector<DIM>& x) const
{
    return (x - origin).array()/extent.array();
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&)> getDenseGridCallback0(
    const DenseGrid<S, CHANNELS, DIM>& grid)
{
    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [&grid](const Vector<DIM>& a) -> T { return (grid(a))(0); };

    } else {
        return [&grid](const Vector<DIM>& a) -> T { return grid(a); };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&)> getDenseGridCallback0(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation)
{
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> grid =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridData, gridShape, gridMin, gridMax, enableInterpolation);

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [grid](const Vector<DIM>& a) -> T { return ((*grid)(a))(0); };

    } else {
        return [grid](const Vector<DIM>& a) -> T { return (*grid)(a); };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback1(
    const DenseGrid<S, CHANNELS, DIM>& grid)
{
    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [&grid](const Vector<DIM>& a, bool b) -> T { return (grid(a))(0); };

    } else {
        return [&grid](const Vector<DIM>& a, bool b) -> T { return grid(a); };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback1(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation)
{
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> grid =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridData, gridShape, gridMin, gridMax, enableInterpolation);

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [grid](const Vector<DIM>& a, bool b) -> T { return ((*grid)(a))(0); };

    } else {
        return [grid](const Vector<DIM>& a, bool b) -> T { return (*grid)(a); };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback2(
    const DenseGrid<S, CHANNELS, DIM>& grid,
    const DenseGrid<S, CHANNELS, DIM>& gridBoundaryNormalAligned)
{
    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [&grid, &gridBoundaryNormalAligned](const Vector<DIM>& a, bool b) -> T {
            return b ? (gridBoundaryNormalAligned(a))(0) : (grid(a))(0);
        };

    } else {
        return [&grid, &gridBoundaryNormalAligned](const Vector<DIM>& a, bool b) -> T {
            return b ? gridBoundaryNormalAligned(a) : grid(a);
        };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, bool)> getDenseGridCallback2(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridDataBoundaryNormalAligned,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation)
{
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> grid =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridData, gridShape, gridMin, gridMax, enableInterpolation);
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> gridBoundaryNormalAligned =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridDataBoundaryNormalAligned, gridShape, gridMin, gridMax, enableInterpolation);

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [grid, gridBoundaryNormalAligned](const Vector<DIM>& a, bool b) -> T {
            return b ? ((*gridBoundaryNormalAligned)(a))(0) : ((*grid)(a))(0);
        };

    } else {
        return [grid, gridBoundaryNormalAligned](const Vector<DIM>& a, bool b) -> T {
            return b ? (*gridBoundaryNormalAligned)(a) : (*grid)(a);
        };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback3(
    const DenseGrid<S, CHANNELS, DIM>& grid)
{
    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [&grid](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T { return (grid(a))(0); };

    } else {
        return [&grid](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T { return grid(a); };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback3(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation)
{
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> grid =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridData, gridShape, gridMin, gridMax, enableInterpolation);

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [grid](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T { return ((*grid)(a))(0); };

    } else {
        return [grid](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T { return (*grid)(a); };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback4(
    const DenseGrid<S, CHANNELS, DIM>& grid,
    const DenseGrid<S, CHANNELS, DIM>& gridBoundaryNormalAligned)
{
    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [&grid, &gridBoundaryNormalAligned](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T {
            return c ? (gridBoundaryNormalAligned(a))(0) : (grid(a))(0);
        };

    } else {
        return [&grid, &gridBoundaryNormalAligned](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T {
            return c ? gridBoundaryNormalAligned(a) : grid(a);
        };
    }
}

template <typename T, typename S, size_t CHANNELS, size_t DIM>
std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> getDenseGridCallback4(
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridData,
    const Eigen::Matrix<S, Eigen::Dynamic, CHANNELS>& gridDataBoundaryNormalAligned,
    const Vectori<DIM>& gridShape, const Vector<DIM>& gridMin,
    const Vector<DIM>& gridMax, bool enableInterpolation)
{
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> grid =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridData, gridShape, gridMin, gridMax, enableInterpolation);
    std::shared_ptr<DenseGrid<S, CHANNELS, DIM>> gridBoundaryNormalAligned =
        std::make_shared<DenseGrid<S, CHANNELS, DIM>>(
            gridDataBoundaryNormalAligned, gridShape, gridMin, gridMax, enableInterpolation);

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        return [grid, gridBoundaryNormalAligned](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T {
            return c ? ((*gridBoundaryNormalAligned)(a))(0) : ((*grid)(a))(0);
        };

    } else {
        return [grid, gridBoundaryNormalAligned](const Vector<DIM>& a, const Vector<DIM>& b, bool c) -> T {
            return c ? (*gridBoundaryNormalAligned)(a) : (*grid)(a);
        };
    }
}

} // zombie

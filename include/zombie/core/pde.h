#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>

namespace zombie {

template<size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <typename T, size_t DIM>
struct PDE {
    // constructor
    PDE(): absorption(0.0f), source({}), dirichlet({}),
           neumann({}), robin({}), robinCoeff({}), 
           dirichletDoubleSided({}), neumannDoubleSided({}),
           robinDoubleSided({}), robinCoeffDoubleSided({}) {
        hasNonZeroRobinCoeff = [this](const Vector<DIM>& x) {
            if (robinCoeffDoubleSided) {
                return robinCoeffDoubleSided(x, true) > 0.0f ||
                       robinCoeffDoubleSided(x, false) > 0.0f;

            } else if (robinCoeff) {
                return robinCoeff(x) > 0.0f;
            }

            return false;
        };
    }

    // members
    float absorption;
    std::function<T(const Vector<DIM>&)> source;
    std::function<T(const Vector<DIM>&)> dirichlet;
    std::function<T(const Vector<DIM>&)> neumann;
    std::function<T(const Vector<DIM>&)> robin;
    std::function<float(const Vector<DIM>&)> robinCoeff;
    std::function<T(const Vector<DIM>&, bool)> dirichletDoubleSided;
    std::function<T(const Vector<DIM>&, bool)> neumannDoubleSided;
    std::function<T(const Vector<DIM>&, bool)> robinDoubleSided;
    std::function<float(const Vector<DIM>&, bool)> robinCoeffDoubleSided;
    std::function<bool(const Vector<DIM>&)> hasNonZeroRobinCoeff;
};

} // zombie

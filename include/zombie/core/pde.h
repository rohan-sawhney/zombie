// This file defines an interface for Partial Differential Equations (PDEs),
// specifically Poisson and screened Poisson equations, with Dirichlet, Neumann,
// and Robin boundary conditions. As part of the problem setup, users of Zombie
// should populate the callback functions defined by the PDE interface.

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
    PDE();

    // members
    float absorption;
    std::function<T(const Vector<DIM>&)> source;
    std::function<T(const Vector<DIM>&, bool)> dirichlet;
    std::function<T(const Vector<DIM>&, bool)> neumann;
    std::function<T(const Vector<DIM>&, bool)> robin;
    std::function<float(const Vector<DIM>&, bool)> robinCoeff;
    std::function<bool(const Vector<DIM>&)> hasNonZeroRobinCoeff; // set automatically
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline PDE<T, DIM>::PDE():
absorption(0.0f),
source({}),
dirichlet({}),
neumann({}),
robin({}),
robinCoeff({}) {
    hasNonZeroRobinCoeff = [this](const Vector<DIM>& x) {
        if (robinCoeff) {
            return robinCoeff(x, true) > 0.0f ||
                   robinCoeff(x, false) > 0.0f;
        }

        return false;
    };
}

} // zombie

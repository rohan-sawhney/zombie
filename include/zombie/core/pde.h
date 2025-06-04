// This file defines an interface for Partial Differential Equations (PDEs),
// specifically Poisson and screened Poisson equations, with Dirichlet, Neumann,
// and Robin boundary conditions. As part of the problem setup, users of Zombie
// should populate the callback functions defined by the PDE interface.

#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <zombie/core/sampling.h>

namespace zombie {

template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <typename T, size_t DIM>
struct PDE {
    // constructor
    PDE();

    // members
    float absorptionCoeff; // must be positive or equal to zero
    bool areRobinConditionsPureNeumann; // set to false if Robin coefficients are non-zero anywhere
    bool areRobinCoeffsNonnegative; // set to false if Robin coefficients are negative anywhere

    // returns source term
    std::function<T(const Vector<DIM>&)> source;

    // returns Dirichlet boundary conditions
    std::function<T(const Vector<DIM>&, bool)> dirichlet;

    // returns Robin boundary conditions and coefficients
    std::function<T(const Vector<DIM>&, const Vector<DIM>&, bool)> robin; // dual purposes for Neumann conditions when Robin coeff is zero
    std::function<float(const Vector<DIM>&, const Vector<DIM>&, bool)> robinCoeff;

    // checks if the PDE has reflecting boundary conditions (Neumann or Robin) at the given point
    std::function<bool(const Vector<DIM>&)> hasReflectingBoundaryConditions;

    // check if the PDE has a non-zero robin coefficient value at the given point
    std::function<bool(const Vector<DIM>&)> hasNonZeroRobinCoeff; // set automatically

    // importance sampling factory
    typename ImportanceSampler<DIM>::SamplerFactoryFn importanceSampler;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T, size_t DIM>
inline PDE<T, DIM>::PDE():
absorptionCoeff(0.0f),
areRobinConditionsPureNeumann(true),
areRobinCoeffsNonnegative(true),
source({}),
dirichlet({}),
robin({}),
robinCoeff({}),
hasReflectingBoundaryConditions({}),
importanceSampler({})
{
    hasNonZeroRobinCoeff = [this](const Vector<DIM>& x) {
        if (this->robinCoeff) {
            Vector<DIM> n = Vector<DIM>::Zero();
            n(0) = 1.0f;

            return std::fabs(this->robinCoeff(x, n, true)) > 0.0f ||
                   std::fabs(this->robinCoeff(x, n, false)) > 0.0f;
        }

        return false;
    };
}

} // zombie

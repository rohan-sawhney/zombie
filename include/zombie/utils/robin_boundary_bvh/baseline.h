// This file extends the 'Baseline' structure from the FCPW library to support Robin
// boundary conditions. Users of Zombie need not interact with this file directly.

#pragma once

#include <zombie/utils/robin_boundary_bvh/geometry.h>

namespace zombie {

using namespace fcpw;

template<size_t DIM, typename PrimitiveType=Primitive<DIM>>
class RobinBaseline: public Baseline<DIM, PrimitiveType> {
public:
    // constructor
    RobinBaseline(std::vector<PrimitiveType *>& primitives_,
                  std::vector<SilhouettePrimitive<DIM> *>& silhouettes_);

    // updates robin coefficient for each triangle
    void updateRobinCoefficients(const std::vector<float>& minCoeffValues,
                                 const std::vector<float>& maxCoeffValues);

    // computes the squared Robin star radius
    int computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                 bool flipNormalOrientation,
                                 float silhouettePrecision) const;
};

template<size_t DIM, typename PrimitiveType>
std::unique_ptr<RobinBaseline<DIM, PrimitiveType>> createRobinBaseline(
                                    std::vector<PrimitiveType *>& primitives,
                                    std::vector<SilhouettePrimitive<DIM> *>& silhouettes);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template<size_t DIM, typename PrimitiveType>
inline RobinBaseline<DIM, PrimitiveType>::RobinBaseline(std::vector<PrimitiveType *>& primitives_,
                                                        std::vector<SilhouettePrimitive<DIM> *>& silhouettes_):
Baseline<DIM, PrimitiveType>(primitives_, silhouettes_)
{
    // do nothing
}

template<size_t DIM, typename PrimitiveType>
inline void RobinBaseline<DIM, PrimitiveType>::updateRobinCoefficients(const std::vector<float>& minCoeffValues,
                                                                       const std::vector<float>& maxCoeffValues)
{
    for (int p = 0; p < (int)Baseline<DIM, PrimitiveType>::primitives.size(); p++) {
        PrimitiveType *prim = Baseline<DIM, PrimitiveType>::primitives[p];

        prim->minRobinCoeff = minCoeffValues[prim->getIndex()];
        prim->maxRobinCoeff = maxCoeffValues[prim->getIndex()];
    }
}

template<size_t DIM, typename PrimitiveType>
inline int RobinBaseline<DIM, PrimitiveType>::computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                                                       bool flipNormalOrientation,
                                                                       float silhouettePrecision) const
{
    int nPrimitives = (int)Baseline<DIM, PrimitiveType>::primitives.size();
    for (int p = 0; p < nPrimitives; p++) {
        Baseline<DIM, PrimitiveType>::primitives[p]->computeSquaredStarRadius(s, flipNormalOrientation,
                                                                              silhouettePrecision);
    }

    return nPrimitives;
}

template<size_t DIM, typename PrimitiveType>
std::unique_ptr<RobinBaseline<DIM, PrimitiveType>> createRobinBaseline(
                                    std::vector<PrimitiveType *>& primitives,
                                    std::vector<SilhouettePrimitive<DIM> *>& silhouettes)
{
    if (primitives.size() > 0) {
        return std::unique_ptr<RobinBaseline<DIM, PrimitiveType>>(
                new RobinBaseline<DIM, PrimitiveType>(primitives, silhouettes));
    }

    return nullptr;
}

} // namespace zombie

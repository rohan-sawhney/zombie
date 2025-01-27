// This file extends the 'Baseline' structure from the FCPW library to support reflectance-based
// boundary conditions. Users of Zombie need not interact with this file directly.

#pragma once

#include <zombie/utils/reflectance_boundary_bvh/geometry.h>

namespace zombie {

using namespace fcpw;

template<size_t DIM, typename PrimitiveType=Primitive<DIM>>
class ReflectanceBaseline: public Baseline<DIM, PrimitiveType> {
public:
    // constructor
    ReflectanceBaseline(std::vector<PrimitiveType *>& primitives_,
                        std::vector<SilhouettePrimitive<DIM> *>& silhouettes_);

    // updates reflectance coefficient for each triangle
    void updateReflectanceCoefficients(const std::vector<float>& minCoeffValues,
                                       const std::vector<float>& maxCoeffValues);

    // computes the squared reflectance star radius
    int computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                 bool flipNormalOrientation,
                                 float silhouettePrecision) const;
};

template<size_t DIM, typename PrimitiveType>
std::unique_ptr<ReflectanceBaseline<DIM, PrimitiveType>> createReflectanceBaseline(
                                            std::vector<PrimitiveType *>& primitives,
                                            std::vector<SilhouettePrimitive<DIM> *>& silhouettes);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template<size_t DIM, typename PrimitiveType>
inline ReflectanceBaseline<DIM, PrimitiveType>::ReflectanceBaseline(std::vector<PrimitiveType *>& primitives_,
                                                                    std::vector<SilhouettePrimitive<DIM> *>& silhouettes_):
Baseline<DIM, PrimitiveType>(primitives_, silhouettes_)
{
    // do nothing
}

template<size_t DIM, typename PrimitiveType>
inline void ReflectanceBaseline<DIM, PrimitiveType>::updateReflectanceCoefficients(const std::vector<float>& minCoeffValues,
                                                                                   const std::vector<float>& maxCoeffValues)
{
    for (int p = 0; p < (int)Baseline<DIM, PrimitiveType>::primitives.size(); p++) {
        PrimitiveType *prim = Baseline<DIM, PrimitiveType>::primitives[p];

        prim->minReflectanceCoeff = minCoeffValues[prim->getIndex()];
        prim->maxReflectanceCoeff = maxCoeffValues[prim->getIndex()];
    }
}

template<size_t DIM, typename PrimitiveType>
inline int ReflectanceBaseline<DIM, PrimitiveType>::computeSquaredStarRadius(BoundingSphere<DIM>& s,
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
std::unique_ptr<ReflectanceBaseline<DIM, PrimitiveType>> createReflectanceBaseline(
                                            std::vector<PrimitiveType *>& primitives,
                                            std::vector<SilhouettePrimitive<DIM> *>& silhouettes)
{
    if (primitives.size() > 0) {
        return std::unique_ptr<ReflectanceBaseline<DIM, PrimitiveType>>(
                new ReflectanceBaseline<DIM, PrimitiveType>(primitives, silhouettes));
    }

    return nullptr;
}

} // namespace zombie

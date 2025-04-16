// This file implements nearest neighbor search using the nanoflann library.

#pragma once

#include <functional>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include "nanoflann.hpp"

namespace zombie {

template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;

// Helper class to implement interface for nanoflann
template <size_t DIM>
struct PointCloud {
    // member
    std::vector<Vector<DIM>> points;

    // nanoflann interface
    inline size_t kdtree_get_point_count() const { return points.size(); }
    inline float kdtree_get_pt(const size_t idx, size_t dim) const { return points[idx][dim]; }
    template <class BBOX>
    bool kdtree_get_bbox(BBOX& bb) const { return false; }
};

// Accelerated k nearest neighbor and radius search queries
template <size_t DIM>
class NearestNeighborFinder {
public:
    // constructor
    NearestNeighborFinder();

    // build acceleration structure
    void buildAccelerationStructure(const std::vector<Vector<DIM>>& points);

    // returns the indices of points in the input set
    size_t kNearest(const Vector<DIM>& queryPt, size_t k,
                    std::vector<size_t>& outIndices) const;

    // returns all neighbors within a ball of input radius
    size_t radiusSearch(const Vector<DIM>& queryPt, float radius,
                        std::vector<size_t>& outIndices) const;

protected:
    // members
    PointCloud<DIM> data;
    nanoflann::KDTreeSingleIndexAdaptorParams params;
    using KD_Tree_T = nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<float, PointCloud<DIM>>,
                                                          PointCloud<DIM>, DIM>;
    KD_Tree_T tree;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <size_t DIM>
inline NearestNeighborFinder<DIM>::NearestNeighborFinder():
params(10 /* max leaf */, nanoflann::KDTreeSingleIndexAdaptorFlags::SkipInitialBuildIndex),
tree(DIM, data, params)
{
    // do nothing
}

template <size_t DIM>
inline void NearestNeighborFinder<DIM>::buildAccelerationStructure(const std::vector<Vector<DIM>>& points)
{
    data.points = points;
    tree.buildIndex();
}

template <size_t DIM>
inline size_t NearestNeighborFinder<DIM>::kNearest(const Vector<DIM>& queryPt, size_t k,
                                                   std::vector<size_t>& outIndices) const
{
    if (k > data.points.size()) {
        std::cerr << "k is greater than number of points" << std::endl;
        exit(EXIT_FAILURE);
    }

    outIndices.resize(k);
    std::vector<float> outSquaredDists(k);
    return tree.knnSearch(&queryPt[0], k, &outIndices[0], &outSquaredDists[0]);
}

template <size_t DIM>
inline size_t NearestNeighborFinder<DIM>::radiusSearch(const Vector<DIM>& queryPt, float radius,
                                                       std::vector<size_t>& outIndices) const
{
    float squaredRadius = radius*radius; // nanoflann wants a SQUARED raidus
    std::vector<nanoflann::ResultItem<uint32_t, float>> resultItems;
    size_t nResultItems = tree.radiusSearch(&queryPt[0], squaredRadius, resultItems);

    outIndices.resize(nResultItems);
    for (size_t i = 0; i < nResultItems; i++) {
        outIndices[i] = resultItems[i].first;
    }

    return nResultItems;
}

} // zombie

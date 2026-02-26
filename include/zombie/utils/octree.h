#pragma once

#include "../core/geometric_queries.h"   
#include <fcpw/core/bounding_volumes.h>  
#include <vector>

namespace zombie
{
// NodeData is typically a pointer of data type.
template <typename NodeData, size_t DIM>
struct OctNode
{
    static constexpr int32_t kNumChildren = 1 << DIM;

    OctNode()
    {
        for (int i = 0; i < kNumChildren; ++i)
            children[i] = nullptr;
    }
    ~OctNode()
    {
        for (int i = 0; i < kNumChildren; ++i)
            delete children[i];
    }

    OctNode*              children[kNumChildren];
    std::vector<NodeData> data;
};

// Reference: pbrt-v2.
template <typename NodeData, size_t DIM>
class Octree
{
public:
    using BBox  = fcpw::BoundingBox<DIM>;
    using Point = Vector<DIM>;

    Octree(const BBox& b, int md = 16) :
        maxDepth(md), bound(b) {}

    void Add(const NodeData& dataItem, const BBox& dataBound)
    {
        addPrivate(&root, bound, dataItem, dataBound,
                   (dataBound.pMin - dataBound.pMax).squaredNorm());
    }

    template <typename LookupProc> void Lookup(const Point& p,
                                               LookupProc&  process)
    {
        if (!bound.contains(p)) return;
        this->lookupPrivate(&root, bound, p, process);
    }

    static inline BBox octreeChildBound(int child, const BBox& nodeBound, const Point& pMid)
    {
        BBox childBound;
        for (int i = 0; i < static_cast<int>(DIM); ++i)
        {
            int bit            = 1 << (DIM - 1 - i); // same ordering as child index construction
            childBound.pMin[i] = (child & bit) ? pMid[i] : nodeBound.pMin[i];
            childBound.pMax[i] = (child & bit) ? nodeBound.pMax[i] : pMid[i];
        }
        return childBound;
    }

private:
    void addPrivate(OctNode<NodeData, DIM>* node, const BBox& nodeBound, const NodeData& dataItem, const BBox& dataBound, float diag2, int depth = 0)
    {
        // Possibly add data item to current octree node
        if (depth == maxDepth ||
            (nodeBound.pMin - nodeBound.pMax).squaredNorm() < diag2)
        {
            node->data.push_back(dataItem);
            return;
        }

        // Otherwise add data item to octree children
        Point pMid = .5f * nodeBound.pMin + .5f * nodeBound.pMax;

        // Determine which children the item overlaps
        bool dimFlags[DIM][2];
        for (int i = 0; i < static_cast<int>(DIM); ++i)
        {
            dimFlags[i][0] = (dataBound.pMin[i] <= pMid[i]);
            dimFlags[i][1] = (dataBound.pMax[i] > pMid[i]);
        }

        bool over[OctNode<NodeData, DIM>::kNumChildren];
        for (int i = 0; i < OctNode<NodeData, DIM>::kNumChildren; ++i)
        {
            bool inside = true;
            for (int d = 0; d < static_cast<int>(DIM); ++d)
            {
                // extract the d dimension's bit (MSB = axis 0)
                int bit = (i >> (DIM - 1 - d)) & 1;
                inside &= dimFlags[d][bit];
            }
            over[i] = inside;
        }

        for (int child = 0; child < OctNode<NodeData, DIM>::kNumChildren; ++child)
        {
            if (!over[child]) continue;
            // Allocate octree node if needed and continue recursive traversal
            if (!node->children[child])
                node->children[child] = new OctNode<NodeData, DIM>;
            BBox childBound = octreeChildBound(child, nodeBound, pMid);
            addPrivate(node->children[child], childBound,
                       dataItem, dataBound, diag2, depth + 1);
        }
    }

    template <typename LookupProc> bool lookupPrivate(OctNode<NodeData, DIM>* node,
                                                      const BBox&             nodeBound,
                                                      const Point&            p,
                                                      LookupProc&             process)
    {
        for (uint32_t i = 0; i < node->data.size(); ++i)
            if (!process(node->data[i]))
                return false;

        // Determine which octree child node _p_ is inside
        Point pMid  = .5f * nodeBound.pMin + .5f * nodeBound.pMax;
        int   child = 0;
        for (int d = 0; d < static_cast<int>(DIM); ++d)
        {
            if (p[d] > pMid[d])
                child += (1 << (DIM - 1 - d));
        }

        if (!node->children[child])
            return true;
        BBox childBound = octreeChildBound(child, nodeBound, pMid);
        return lookupPrivate(node->children[child], childBound, p, process);
    }

private:
    int                    maxDepth;
    BBox                   bound;
    OctNode<NodeData, DIM> root;
};
} // namespace zombie

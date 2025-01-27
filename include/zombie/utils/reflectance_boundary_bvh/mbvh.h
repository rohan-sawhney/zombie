// This file extends the 'Mbvh' structure from the FCPW library to support reflectance-based
// boundary conditions. Users of Zombie need not interact with this file directly.

#pragma once

#include <zombie/utils/reflectance_boundary_bvh/bvh.h>

namespace zombie {

using namespace fcpw;

template<size_t DIM>
struct ReflectanceMbvhNode {
    // constructor
    ReflectanceMbvhNode(): boxMin(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(maxFloat)),
                           boxMax(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(minFloat)),
                           coneAxis(FloatP<FCPW_MBVH_BRANCHING_FACTOR>(0.0f)),
                           coneHalfAngle(M_PI), coneRadius(0.0f),
                           minReflectanceCoeff(maxFloat), maxReflectanceCoeff(minFloat),
                           child(maxInt) {}

    // members
    VectorP<FCPW_MBVH_BRANCHING_FACTOR, DIM> boxMin, boxMax;
    VectorP<FCPW_MBVH_BRANCHING_FACTOR, DIM> coneAxis;
    FloatP<FCPW_MBVH_BRANCHING_FACTOR> coneHalfAngle;
    FloatP<FCPW_MBVH_BRANCHING_FACTOR> coneRadius;
    FloatP<FCPW_MBVH_BRANCHING_FACTOR> minReflectanceCoeff;
    FloatP<FCPW_MBVH_BRANCHING_FACTOR> maxReflectanceCoeff;
    IntP<FCPW_MBVH_BRANCHING_FACTOR> child; // use sign to differentiate between inner and leaf nodes
};

template<size_t WIDTH, size_t DIM>
struct MbvhLeafNode {
    // constructor
    MbvhLeafNode(): maxReflectanceCoeff(minFloat), primitiveIndex(-1) {
        for (size_t i = 0; i < DIM; ++i) {
            positions[i] = VectorP<WIDTH, DIM>(maxFloat);
            normals[i] = VectorP<WIDTH, DIM>(0.0f);
            hasAdjacentFace[i] = MaskP<WIDTH>(false);
            ignoreAdjacentFace[i] = MaskP<WIDTH>(true);
        }
    }

    // members
    VectorP<WIDTH, DIM> positions[DIM];
    VectorP<WIDTH, DIM> normals[DIM];
    FloatP<WIDTH> maxReflectanceCoeff;
    IntP<WIDTH> primitiveIndex;
    MaskP<WIDTH> hasAdjacentFace[DIM];
    MaskP<WIDTH> ignoreAdjacentFace[DIM];
};

template<size_t WIDTH, size_t DIM,
         typename PrimitiveType,
         typename NodeType,
         typename NodeBound>
class ReflectanceMbvh: public Mbvh<WIDTH, DIM,
                                   PrimitiveType,
                                   SilhouettePrimitive<DIM>,
                                   NodeType,
                                   MbvhLeafNode<WIDTH, DIM>,
                                   MbvhSilhouetteLeafNode<WIDTH, DIM>> {
public:
    // constructor
    ReflectanceMbvh(std::vector<PrimitiveType *>& primitives_,
                    std::vector<SilhouettePrimitive<DIM> *>& silhouettes_);

    // refits the mbvh
    void refit();

    // updates reflectance coefficient for each triangle
    void updateReflectanceCoefficients(const std::vector<float>& minCoeffValues,
                                       const std::vector<float>& maxCoeffValues);

    // computes the squared reflectance star radius
    int computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                 bool flipNormalOrientation,
                                 float silhouettePrecision) const;

protected:
    // checks which nodes should be visited during traversal
    MaskP<FCPW_MBVH_BRANCHING_FACTOR> visitNodes(const enokiVector<DIM>& sc, float r2, int nodeIndex,
                                                 FloatP<FCPW_MBVH_BRANCHING_FACTOR>& r2MinBound,
                                                 FloatP<FCPW_MBVH_BRANCHING_FACTOR>& r2MaxBound,
                                                 MaskP<FCPW_MBVH_BRANCHING_FACTOR>& hasSilhouettes) const;
};

template<size_t DIM, typename PrimitiveType, typename MbvhNodeBound, typename BvhNodeBound>
std::unique_ptr<ReflectanceMbvh<FCPW_SIMD_WIDTH, DIM, PrimitiveType, ReflectanceMbvhNode<DIM>, MbvhNodeBound>> createVectorizedReflectanceBvh(
                                                        ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, BvhNodeBound> *reflectanceBvh,
                                                        std::vector<PrimitiveType *>& primitives,
                                                        std::vector<SilhouettePrimitive<DIM> *>& silhouettes,
                                                        bool printStats=true);

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template<size_t WIDTH, size_t DIM,
         typename PrimitiveType,
         typename NodeType,
         typename NodeBound>
inline ReflectanceMbvh<WIDTH, DIM, PrimitiveType, NodeType, NodeBound>::ReflectanceMbvh(std::vector<PrimitiveType *>& primitives_,
                                                                                        std::vector<SilhouettePrimitive<DIM> *>& silhouettes_):
Mbvh<WIDTH, DIM,
     PrimitiveType,
     SilhouettePrimitive<DIM>,
     NodeType,
     MbvhLeafNode<WIDTH, DIM>,
     MbvhSilhouetteLeafNode<WIDTH, DIM>>(primitives_, silhouettes_)
{
    using MbvhBase = Mbvh<WIDTH, DIM,
                          PrimitiveType,
                          SilhouettePrimitive<DIM>,
                          NodeType,
                          MbvhLeafNode<WIDTH, DIM>,
                          MbvhSilhouetteLeafNode<WIDTH, DIM>>;

    MbvhBase::primitiveTypeSupportsVectorizedQueries = std::is_same<PrimitiveType, ReflectanceLineSegment<typename PrimitiveType::Bound>>::value ||
                                                       std::is_same<PrimitiveType, ReflectanceTriangle<typename PrimitiveType::Bound>>::value;
}

template<size_t DIM>
inline void assignGeometricDataToNode(const ReflectanceBvhNode<DIM>& bvhNode, ReflectanceMbvhNode<DIM>& mbvhNode, int index)
{
    // assign bvh node's bounding cone and reflectance coefficients to mbvh node
    for (size_t j = 0; j < DIM; j++) {
        mbvhNode.coneAxis[j][index] = bvhNode.cone.axis[j];
    }

    mbvhNode.coneHalfAngle[index] = bvhNode.cone.halfAngle;
    mbvhNode.coneRadius[index] = bvhNode.cone.radius;
    mbvhNode.minReflectanceCoeff[index] = bvhNode.minReflectanceCoeff;
    mbvhNode.maxReflectanceCoeff[index] = bvhNode.maxReflectanceCoeff;
}

template<typename NodeType, typename LeafNodeType, typename PrimitiveBound>
inline void populateLeafNode(const NodeType& node,
                             const std::vector<ReflectanceLineSegment<PrimitiveBound> *>& primitives,
                             std::vector<LeafNodeType>& leafNodes, size_t WIDTH)
{
    int leafOffset = -node.child[0] - 1;
    int referenceOffset = node.child[2];
    int nReferences = node.child[3];

    // populate leaf node with reflectance line segments
    for (int p = 0; p < nReferences; p++) {
        int referenceIndex = referenceOffset + p;
        int leafIndex = leafOffset + p/WIDTH;
        int w = p%WIDTH;
        const ReflectanceLineSegment<PrimitiveBound> *lineSegment = primitives[referenceIndex];
        LeafNodeType& leafNode = leafNodes[leafIndex];

        leafNode.maxReflectanceCoeff[w] = lineSegment->maxReflectanceCoeff;
        leafNode.primitiveIndex[w] = lineSegment->getIndex();
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < 2; j++) {
                leafNode.positions[i][j][w] = lineSegment->soup->positions[lineSegment->indices[i]][j];
                leafNode.normals[i][j][w] = lineSegment->n[i][j];
            }

            leafNode.hasAdjacentFace[i][w] = lineSegment->hasAdjacentFace[i];
            leafNode.ignoreAdjacentFace[i][w] = lineSegment->ignoreAdjacentFace[i];
        }
    }
}

template<typename NodeType, typename LeafNodeType, typename PrimitiveBound>
inline void populateLeafNode(const NodeType& node,
                             const std::vector<ReflectanceTriangle<PrimitiveBound> *>& primitives,
                             std::vector<LeafNodeType>& leafNodes, size_t WIDTH)
{
    int leafOffset = -node.child[0] - 1;
    int referenceOffset = node.child[2];
    int nReferences = node.child[3];

    // populate leaf node with reflectance triangles
    for (int p = 0; p < nReferences; p++) {
        int referenceIndex = referenceOffset + p;
        int leafIndex = leafOffset + p/WIDTH;
        int w = p%WIDTH;
        const ReflectanceTriangle<PrimitiveBound> *triangle = primitives[referenceIndex];
        LeafNodeType& leafNode = leafNodes[leafIndex];

        leafNode.maxReflectanceCoeff[w] = triangle->maxReflectanceCoeff;
        leafNode.primitiveIndex[w] = triangle->getIndex();
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                leafNode.positions[i][j][w] = triangle->soup->positions[triangle->indices[i]][j];
                leafNode.normals[i][j][w] = triangle->n[i][j];
            }

            leafNode.hasAdjacentFace[i][w] = triangle->hasAdjacentFace[i];
            leafNode.ignoreAdjacentFace[i][w] = triangle->ignoreAdjacentFace[i];
        }
    }
}

template<size_t DIM>
inline void assignBoundingCone(const BoundingCone<DIM>& cone, ReflectanceMbvhNode<DIM>& node, int index)
{
    for (size_t i = 0; i < DIM; i++) {
        node.coneAxis[i][index] = cone.axis[i];
    }

    node.coneHalfAngle[index] = cone.halfAngle;
    node.coneRadius[index] = cone.radius;
}

template<size_t DIM>
inline void mergeBoundingCones(const BoundingCone<DIM>& coneA, const BoundingCone<DIM>& coneB,
                               const BoundingBox<DIM>& boxA, const BoundingBox<DIM>& boxB,
                               const BoundingBox<DIM>& mergedBox, ReflectanceMbvhNode<DIM>& node,
                               BoundingCone<DIM>& cone)
{
    cone = mergeBoundingCones<DIM>(coneA, coneB, boxA.centroid(), boxB.centroid(), mergedBox.centroid());
}

template<size_t WIDTH,
         size_t DIM,
         typename NodeType,
         typename PrimitiveType>
inline std::pair<BoundingBox<DIM>, BoundingCone<DIM>> refitRecursive(const std::vector<PrimitiveType *>& primitives,
                                                                     std::vector<NodeType>& flatTree, int nodeIndex)
{
    BoundingBox<DIM> box;
    BoundingCone<DIM> cone;
    cone.halfAngle = -M_PI;
    NodeType& node(flatTree[nodeIndex]);

    if (node.child[0] < 0) { // leaf
        // compute bounding box
        int referenceOffset = node.child[2];
        int nReferences = node.child[3];

        for (int p = 0; p < nReferences; p++) {
            int referenceIndex = referenceOffset + p;
            const PrimitiveType *prim = primitives[referenceIndex];

            box.expandToInclude(prim->boundingBox());
        }

        // compute bounding cone
        Vector<DIM> centroid = box.centroid();
        cone = computeBoundingConeForPrimitives<DIM, PrimitiveType>(
            primitives, centroid, nReferences, referenceOffset);

    } else { // not a leaf
        for (int w = 0; w < FCPW_MBVH_BRANCHING_FACTOR; w++) {
            if (node.child[w] != maxInt) {
                // refit child
                std::pair<BoundingBox<DIM>, BoundingCone<DIM>> childBoxCone =
                    refitRecursive<WIDTH, DIM, NodeType, PrimitiveType>(
                        primitives, flatTree, node.child[w]);

                // expand bounding box
                BoundingBox<DIM> currentBox = box;
                BoundingBox<DIM> childBox = childBoxCone.first;
                for (size_t i = 0; i < DIM; i++) {
                    node.boxMin[i][w] = childBox.pMin[i];
                    node.boxMax[i][w] = childBox.pMax[i];
                }
                box.expandToInclude(childBox);

                // expand bounding cone
                BoundingCone<DIM> childCone = childBoxCone.second;
                assignBoundingCone(childCone, node, w);
                mergeBoundingCones(cone, childCone, currentBox, childBox, box, node, cone);
            }
        }
    }

    return std::make_pair(box, cone);
}

template<size_t WIDTH, size_t DIM,
         typename PrimitiveType,
         typename NodeType,
         typename NodeBound>
inline void ReflectanceMbvh<WIDTH, DIM, PrimitiveType, NodeType, NodeBound>::refit()
{
    using MbvhBase = Mbvh<WIDTH, DIM,
                          PrimitiveType,
                          SilhouettePrimitive<DIM>,
                          NodeType,
                          MbvhLeafNode<WIDTH, DIM>,
                          MbvhSilhouetteLeafNode<WIDTH, DIM>>;

    // update leaf nodes
    MbvhBase::populateLeafNodes();

    // update flatTree
    if (MbvhBase::nNodes > 0) {
        refitRecursive<WIDTH, DIM, NodeType, PrimitiveType>(
            MbvhBase::primitives, MbvhBase::flatTree, 0);
    }
}

template<size_t WIDTH, size_t DIM, typename NodeType, typename PrimitiveType>
inline std::pair<float, float> updateReflectanceCoefficientsRecursive(const std::vector<PrimitiveType *>& primitives,
                                                                      std::vector<NodeType>& flatTree, int nodeIndex)
{
    NodeType& node(flatTree[nodeIndex]);
    std::pair<float, float> minMaxReflectanceCoeffs = std::make_pair(maxFloat, minFloat);

    if (node.child[0] < 0) {
        int referenceOffset = node.child[2];
        int nReferences = node.child[3];

        for (int p = 0; p < nReferences; p++) {
            int referenceIndex = referenceOffset + p;
            const PrimitiveType *prim = primitives[referenceIndex];

            minMaxReflectanceCoeffs.first = std::min(minMaxReflectanceCoeffs.first, prim->minReflectanceCoeff);
            minMaxReflectanceCoeffs.second = std::max(minMaxReflectanceCoeffs.second, prim->maxReflectanceCoeff);
        }

    } else {
        for (int w = 0; w < FCPW_MBVH_BRANCHING_FACTOR; w++) {
            if (node.child[w] != maxInt) {
                // compute min and max reflectance coefficients for child node
                std::pair<float, float> childMinMaxReflectanceCoeffs =
                    updateReflectanceCoefficientsRecursive<WIDTH, DIM, NodeType, PrimitiveType>(
                        primitives, flatTree, node.child[w]);

                // set reflectance coefficients for this node
                node.minReflectanceCoeff[w] = childMinMaxReflectanceCoeffs.first;
                node.maxReflectanceCoeff[w] = childMinMaxReflectanceCoeffs.second;
                minMaxReflectanceCoeffs.first = std::min(minMaxReflectanceCoeffs.first, node.minReflectanceCoeff[w]);
                minMaxReflectanceCoeffs.second = std::max(minMaxReflectanceCoeffs.second, node.maxReflectanceCoeff[w]);
            }
        }
    }

    return minMaxReflectanceCoeffs;
}

template<size_t WIDTH, size_t DIM,
         typename PrimitiveType,
         typename NodeType,
         typename NodeBound>
inline void ReflectanceMbvh<WIDTH, DIM, PrimitiveType, NodeType, NodeBound>::updateReflectanceCoefficients(const std::vector<float>& minCoeffValues,
                                                                                                           const std::vector<float>& maxCoeffValues)
{
    using MbvhBase = Mbvh<WIDTH, DIM,
                          PrimitiveType,
                          SilhouettePrimitive<DIM>,
                          NodeType,
                          MbvhLeafNode<WIDTH, DIM>,
                          MbvhSilhouetteLeafNode<WIDTH, DIM>>;

    // update reflectance coefficients for primitives
    for (int i = 0; i < MbvhBase::nNodes; i++) {
        NodeType& node = MbvhBase::flatTree[i];

        if (MbvhBase::isLeafNode(node)) {
            int leafOffset = -node.child[0] - 1;
            int referenceOffset = node.child[2];
            int nReferences = node.child[3];

            for (int p = 0; p < nReferences; p++) {
                int referenceIndex = referenceOffset + p;
                int leafIndex = leafOffset + p/WIDTH;
                int w = p%WIDTH;
                PrimitiveType *prim = MbvhBase::primitives[referenceIndex];
                MbvhLeafNode<WIDTH, DIM>& leafNode = MbvhBase::leafNodes[leafIndex];

                prim->minReflectanceCoeff = minCoeffValues[prim->getIndex()];
                prim->maxReflectanceCoeff = maxCoeffValues[prim->getIndex()];
                leafNode.maxReflectanceCoeff[w] = prim->maxReflectanceCoeff;
            }
        }
    }

    // update reflectance coefficients for vectorized nodes
    if (MbvhBase::nNodes > 0) {
        updateReflectanceCoefficientsRecursive<WIDTH, DIM, NodeType, PrimitiveType>(
            MbvhBase::primitives, MbvhBase::flatTree, 0);
    }
}

template<size_t WIDTH, size_t DIM,
         typename PrimitiveType,
         typename NodeType,
         typename NodeBound>
inline MaskP<FCPW_MBVH_BRANCHING_FACTOR> ReflectanceMbvh<WIDTH, DIM, PrimitiveType, NodeType, NodeBound>::visitNodes(
                                                                    const enokiVector<DIM>& sc, float r2, int nodeIndex,
                                                                    FloatP<FCPW_MBVH_BRANCHING_FACTOR>& r2MinBound,
                                                                    FloatP<FCPW_MBVH_BRANCHING_FACTOR>& r2MaxBound,
                                                                    MaskP<FCPW_MBVH_BRANCHING_FACTOR>& hasSilhouettes) const
{
    using MbvhBase = Mbvh<WIDTH, DIM,
                          PrimitiveType,
                          SilhouettePrimitive<DIM>,
                          NodeType,
                          MbvhLeafNode<WIDTH, DIM>,
                          MbvhSilhouetteLeafNode<WIDTH, DIM>>;
    const NodeType& node(MbvhBase::flatTree[nodeIndex]);
    hasSilhouettes = true;

    // perform box overlap test
    MaskP<FCPW_MBVH_BRANCHING_FACTOR> overlapBox = enoki::neq(node.child, maxInt) && 
        overlapWideBox<FCPW_MBVH_BRANCHING_FACTOR, DIM>(node.boxMin, node.boxMax, sc, r2,
                                                        r2MinBound, r2MaxBound);

    // early out for perfectly absorbing case
    MaskP<FCPW_MBVH_BRANCHING_FACTOR> isNotPerfectlyAbsorbing = node.minReflectanceCoeff < maxFloat - epsilon;
    MaskP<FCPW_MBVH_BRANCHING_FACTOR> overlapNotPerfectlyAbsorbingBox = overlapBox && isNotPerfectlyAbsorbing;
    if (enoki::any(overlapNotPerfectlyAbsorbingBox)) {
        // perform cone overlap test
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> maximalAngles[2];
        hasSilhouettes = overlapNotPerfectlyAbsorbingBox;
        overlapWideCone<FCPW_MBVH_BRANCHING_FACTOR, DIM>(node.coneAxis, node.coneHalfAngle, node.coneRadius,
                                                         sc, node.boxMin, node.boxMax, r2MinBound,
                                                         maximalAngles[0], maximalAngles[1], hasSilhouettes);
        enoki::masked(r2MaxBound, hasSilhouettes) = maxFloat;

        // update mask for perfectly reflecting case
        MaskP<FCPW_MBVH_BRANCHING_FACTOR> isNotPerfectlyReflecting = node.maxReflectanceCoeff > epsilon;
        MaskP<FCPW_MBVH_BRANCHING_FACTOR> overlapsPerfectlyReflectingBox = overlapBox && ~isNotPerfectlyReflecting;
        if (enoki::any(overlapsPerfectlyReflectingBox)) {
            enoki::masked(overlapBox, overlapsPerfectlyReflectingBox) = hasSilhouettes;
        }

        // update bounds for reflectance case
        MaskP<FCPW_MBVH_BRANCHING_FACTOR> overlapReflectanceBox = overlapNotPerfectlyAbsorbingBox && isNotPerfectlyReflecting;
        if (enoki::any(overlapReflectanceBox)) {
            MaskP<FCPW_MBVH_BRANCHING_FACTOR> overlapReflectanceBoxNotCone = overlapReflectanceBox && ~hasSilhouettes;
            FloatP<FCPW_MBVH_BRANCHING_FACTOR> rMin = enoki::sqrt(r2MinBound);
            FloatP<FCPW_MBVH_BRANCHING_FACTOR> rMax = enoki::sqrt(r2MaxBound);
            FloatP<FCPW_MBVH_BRANCHING_FACTOR> minAbsCosTheta = enoki::min(enoki::abs(enoki::cos(maximalAngles[0])),
                                                                           enoki::abs(enoki::cos(maximalAngles[1])));
            FloatP<FCPW_MBVH_BRANCHING_FACTOR> maxAbsCosTheta = 1.0f; // assume maxCosTheta = 1.0f for simplicity
            enoki::masked(r2MinBound, overlapReflectanceBoxNotCone) = NodeBound::computeMinSquaredStarRadiusBound(
                rMin, rMax, node.minReflectanceCoeff, node.maxReflectanceCoeff, minAbsCosTheta, maxAbsCosTheta);
            enoki::masked(r2MaxBound, overlapReflectanceBoxNotCone) = NodeBound::computeMaxSquaredStarRadiusBound(
                rMin, rMax, node.minReflectanceCoeff, node.maxReflectanceCoeff, minAbsCosTheta, maxAbsCosTheta);
        }
    }

    return overlapBox;
}

template<size_t WIDTH>
inline void enqueueNodes(const IntP<WIDTH>& child, const FloatP<WIDTH>& tMin, const FloatP<WIDTH>& tMax,
                         const MaskP<WIDTH>& hasSilhouettes, const MaskP<WIDTH>& mask, float minDist,
                         float& tMaxMin, int& stackPtr, TraversalStack *subtree)
{
    // enqueue nodes
    int closestIndex = -1;
    for (int w = 0; w < WIDTH; w++) {
        if (mask[w]) {
            stackPtr++;
            subtree[stackPtr].node = child[w];
            subtree[stackPtr].distance = tMin[w]*(hasSilhouettes[w] ? 1.0f : -1.0f);
            tMaxMin = std::min(tMaxMin, tMax[w]);

            if (tMin[w] < minDist) {
                closestIndex = stackPtr;
                minDist = tMin[w];
            }
        }
    }

    // put closest node first
    if (closestIndex != -1) {
        std::swap(subtree[stackPtr], subtree[closestIndex]);
    }
}

template<>
inline void enqueueNodes<4>(const IntP<4>& child, const FloatP<4>& tMin, const FloatP<4>& tMax,
                            const MaskP<4>& hasSilhouettes, const MaskP<4>& mask, float minDist,
                            float& tMaxMin, int& stackPtr, TraversalStack *subtree)
{
    // sort nodes
    int order[4] = {0, 1, 2, 3};
    sortOrder4(tMin, order[0], order[1], order[2], order[3]);

    // enqueue overlapping nodes in sorted order
    for (int w = 0; w < 4; w++) {
        int W = order[w];

        if (mask[W]) {
            stackPtr++;
            subtree[stackPtr].node = child[W];
            subtree[stackPtr].distance = tMin[W]*(hasSilhouettes[W] ? 1.0f : -1.0f);
            tMaxMin = std::min(tMaxMin, tMax[W]);
        }
    }
}

template<size_t WIDTH, size_t DIM,
         typename PrimitiveType,
         typename NodeType,
         typename NodeBound>
inline int ReflectanceMbvh<WIDTH, DIM, PrimitiveType, NodeType, NodeBound>::computeSquaredStarRadius(BoundingSphere<DIM>& s,
                                                                                                     bool flipNormalOrientation,
                                                                                                     float silhouettePrecision) const
{
    using MbvhBase = Mbvh<WIDTH, DIM,
                          PrimitiveType,
                          SilhouettePrimitive<DIM>,
                          NodeType,
                          MbvhLeafNode<WIDTH, DIM>,
                          MbvhSilhouetteLeafNode<WIDTH, DIM>>;

    TraversalStack subtree[FCPW_MBVH_MAX_DEPTH];
    FloatP<FCPW_MBVH_BRANCHING_FACTOR> d2Min, d2Max;
    MaskP<FCPW_MBVH_BRANCHING_FACTOR> hasSilhouettes;
    enokiVector<DIM> sc = enoki::gather<enokiVector<DIM>>(s.c.data(), MbvhBase::range);
    int nodesVisited = 0;

    // push root node
    subtree[0].node = 0;
    subtree[0].distance = 0.0f;
    int stackPtr = 0;

    while (stackPtr >= 0) {
        // pop off the next node to work on
        int nodeIndex = subtree[stackPtr].node;
        float currentDist = subtree[stackPtr].distance;
        stackPtr--;

        // if this node is further than the current radius estimate, continue
        if (std::fabs(currentDist) > s.r2) continue;
        const NodeType& node(MbvhBase::flatTree[nodeIndex]);

        if (MbvhBase::isLeafNode(node)) {
            if (MbvhBase::primitiveTypeSupportsVectorizedQueries) {
                int leafOffset = -node.child[0] - 1;
                int nLeafs = node.child[1];
                int nReferences = node.child[3];
                int startReference = 0;
                nodesVisited++;

                for (int l = 0; l < nLeafs; l++) {
                    // perform vectorized primitive query
                    int leafIndex = leafOffset + l;
                    const MbvhLeafNode<WIDTH, DIM>& leafNode = MbvhBase::leafNodes[leafIndex];
                    FloatP<WIDTH> d2 = ReflectanceWidePrimitive<WIDTH, DIM, typename PrimitiveType::Bound>::computeSquaredStarRadiusWidePrimitive(
                                                                                        leafNode.positions, leafNode.normals, leafNode.maxReflectanceCoeff,
                                                                                        leafNode.hasAdjacentFace, leafNode.ignoreAdjacentFace, sc, s.r2,
                                                                                        flipNormalOrientation, silhouettePrecision, currentDist >= 0.0f);

                    // update squared radius
                    int W = std::min((int)WIDTH, nReferences - startReference);
                    for (int w = 0; w < W; w++) {
                        s.r2 = std::min(s.r2, d2[w]);
                    }

                    startReference += WIDTH;
                }

            } else {
                // primitive type does not support vectorized star radius query,
                // perform query to each primitive one by one
                int referenceOffset = node.child[2];
                int nReferences = node.child[3];

                for (int p = 0; p < nReferences; p++) {
                    int referenceIndex = referenceOffset + p;
                    const PrimitiveType *prim = MbvhBase::primitives[referenceIndex];
                    nodesVisited++;

                    // assume we are working only with reflectance primitives
                    prim->computeSquaredStarRadius(s, flipNormalOrientation, silhouettePrecision, currentDist >= 0.0f);
                }
            }

        } else {
            // determine which nodes to visit
            MaskP<FCPW_MBVH_BRANCHING_FACTOR> mask = visitNodes(sc, s.r2, nodeIndex, d2Min, d2Max, hasSilhouettes);

            // enqueue overlapping boxes in sorted order
            nodesVisited++;
            if (enoki::any(mask)) {
                enqueueNodes<FCPW_MBVH_BRANCHING_FACTOR>(node.child, d2Min, d2Max, hasSilhouettes,
                                                         mask, s.r2, s.r2, stackPtr, subtree);
            }
        }
    }

    return nodesVisited;
}

template<size_t DIM, typename PrimitiveType, typename MbvhNodeBound, typename BvhNodeBound>
std::unique_ptr<ReflectanceMbvh<FCPW_SIMD_WIDTH, DIM, PrimitiveType, ReflectanceMbvhNode<DIM>, MbvhNodeBound>> createVectorizedReflectanceBvh(
                                                        ReflectanceBvh<DIM, ReflectanceBvhNode<DIM>, PrimitiveType, BvhNodeBound> *reflectanceBvh,
                                                        std::vector<PrimitiveType *>& primitives,
                                                        std::vector<SilhouettePrimitive<DIM> *>& silhouettes,
                                                        bool printStats)
{
    if (primitives.size() > 0) {
        using namespace std::chrono;
        high_resolution_clock::time_point t1 = high_resolution_clock::now();

        std::unique_ptr<ReflectanceMbvh<FCPW_SIMD_WIDTH, DIM, PrimitiveType, ReflectanceMbvhNode<DIM>, MbvhNodeBound>> mbvh(
            new ReflectanceMbvh<FCPW_SIMD_WIDTH, DIM, PrimitiveType, ReflectanceMbvhNode<DIM>, MbvhNodeBound>(primitives, silhouettes));
        mbvh->template initialize<ReflectanceBvhNode<DIM>>(reflectanceBvh);

        if (printStats) {
            high_resolution_clock::time_point t2 = high_resolution_clock::now();
            duration<double> timeSpan = duration_cast<duration<double>>(t2 - t1);
            std::cout << FCPW_MBVH_BRANCHING_FACTOR << "-BVH construction time: " << timeSpan.count() << " seconds" << std::endl;
            mbvh->printStats();
        }

        return mbvh;
    }

    return nullptr;
}

} // namespace zombie

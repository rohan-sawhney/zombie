// This file implements reflectance bounds for Robin boundary conditions.
// Users of Zombie need not interact with this file directly.

#pragma once

#include <fcpw/fcpw.h>

namespace zombie {

using namespace fcpw;

struct RobinLineSegmentBound {
    // computes the squared star radius bound
    static void computeSquaredStarRadiusBound(BoundingSphere<2>& s, float maxRobinCoeff,
                                              const Vector2& viewDirClosest, float closestPtDist,
                                              const Vector2& viewDirFarthest, float farthestPtDist,
                                              const Vector2& planePt, const Vector2& planeNormal) {
        Vector2 planeClosestPt;
        float h = findClosestPointPlane<2>(planePt, planeNormal, s.c, planeClosestPt);
        float cosUpperBound = std::fabs((viewDirClosest/closestPtDist).dot(planeNormal));
        float cosLowerBound = std::fabs((viewDirFarthest/farthestPtDist).dot(planeNormal));
        float cosLine = std::sqrt(h*maxRobinCoeff)/SQRT_2;
        float cosLineSegment = std::clamp(cosLine, cosLowerBound, cosUpperBound);
        float cosLineSegment2 = cosLineSegment*cosLineSegment;
        float lineSegmentRadius = (h/cosLineSegment)*std::exp(cosLineSegment2/(h*maxRobinCoeff));
        float lineSegmentRadius2 = lineSegmentRadius*lineSegmentRadius;
        s.r2 = std::min(s.r2, lineSegmentRadius2);
    }

#ifdef FCPW_USE_ENOKI
    // computes the squared star radius bound
    template <size_t WIDTH>
    static void computeSquaredStarRadiusBound(const enokiVector2& sc, FloatP<WIDTH>& r2, const FloatP<WIDTH>& maxRobinCoeff,
                                              const MaskP<WIDTH> activeRobin, const Vector2P<WIDTH>& viewDirClosest,
                                              const FloatP<WIDTH>& closestPtDist, const FloatP<WIDTH>& closestPtDist2,
                                              const Vector2P<WIDTH>& viewDirFarthest, const FloatP<WIDTH>& farthestPtDist,
                                              const Vector2P<WIDTH>& planePt, const Vector2P<WIDTH>& planeNormal);
#endif
};

struct RobinTriangleBound {
    // computes the squared star radius bound
    static void computeSquaredStarRadiusBound(BoundingSphere<3>& s, float maxRobinCoeff,
                                              const Vector3& viewDirClosest, float closestPtDist,
                                              const Vector3& viewDirFarthest, float farthestPtDist,
                                              const Vector3& planePt, const Vector3& planeNormal) {
        Vector3 planeClosestPt;
        float h = findClosestPointPlane<3>(planePt, planeNormal, s.c, planeClosestPt);
        float cosUpperBound = std::fabs((viewDirClosest/closestPtDist).dot(planeNormal));
        float cosLowerBound = std::fabs((viewDirFarthest/farthestPtDist).dot(planeNormal));
        float maxCosForBound = std::sqrt(h*maxRobinCoeff);
        if (maxCosForBound >= cosLowerBound) {
            float cosPlane = maxCosForBound/SQRT_3;
            float cosTriangle = std::clamp(cosPlane, cosLowerBound, cosUpperBound);
            float cosTriangle2 = cosTriangle*cosTriangle;
            float triangleRadius = h*h*maxRobinCoeff/(cosTriangle*(h*maxRobinCoeff - cosTriangle2));
            float triangleRadius2 = triangleRadius*triangleRadius;
            s.r2 = std::min(s.r2, triangleRadius2);
        }
    }

#ifdef FCPW_USE_ENOKI
    // computes the squared star radius bound
    template <size_t WIDTH>
    static void computeSquaredStarRadiusBound(const enokiVector3& sc, FloatP<WIDTH>& r2, const FloatP<WIDTH>& maxRobinCoeff,
                                              const MaskP<WIDTH> activeRobin, const Vector3P<WIDTH>& viewDirClosest,
                                              const FloatP<WIDTH>& closestPtDist, const FloatP<WIDTH>& closestPtDist2,
                                              const Vector3P<WIDTH>& viewDirFarthest, const FloatP<WIDTH>& farthestPtDist,
                                              const Vector3P<WIDTH>& planePt, const Vector3P<WIDTH>& planeNormal);
#endif
};

template <size_t DIM>
struct RobinBvhNodeBound {
    // computes the minimum squared star radius bound
    static float computeMinSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minRobinCoeff, float maxRobinCoeff,
                                                  float minCosTheta, float maxCosTheta) {
        std::cerr << "RobinBvhNodeBound::computeMinSquaredStarRadiusBound(): DIM: " << DIM << " not supported" << std::endl;
        exit(EXIT_FAILURE);

        return 0.0f;
    }

    // computes the maximum squared star radius bound
    static float computeMaxSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minRobinCoeff, float maxRobinCoeff,
                                                  float minCosTheta, float maxCosTheta) {
        std::cerr << "RobinBvhNodeBound::computeMaxSquaredStarRadiusBound(): DIM: " << DIM << " not supported" << std::endl;
        exit(EXIT_FAILURE);

        return 0.0f;
    }
};

template <>
struct RobinBvhNodeBound<2> {
    // computes the minimum squared star radius bound
    static float computeMinSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minRobinCoeff, float maxRobinCoeff,
                                                  float minCosTheta, float maxCosTheta) {
        float rBound = rMin*std::exp(minCosTheta/(maxRobinCoeff*rMax));
        return rBound*rBound;
    }

    // computes the maximum squared star radius bound
    static float computeMaxSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minRobinCoeff, float maxRobinCoeff,
                                                  float minCosTheta, float maxCosTheta) {
        float rBound = rMax*std::exp(maxCosTheta/(minRobinCoeff*rMin));
        return rBound*rBound;
    }
};

template <>
struct RobinBvhNodeBound<3> {
    // computes the minimum squared star radius bound
    static float computeMinSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minRobinCoeff, float maxRobinCoeff,
                                                  float minCosTheta, float maxCosTheta) {
        if (rMax < minCosTheta/maxRobinCoeff) {
            return maxFloat;
        }

        float rBound = rMin/(1.0f - (minCosTheta/(maxRobinCoeff*rMax)));
        return rBound*rBound;
    }

    // computes the maximum squared star radius bound
    static float computeMaxSquaredStarRadiusBound(float rMin, float rMax,
                                                  float minRobinCoeff, float maxRobinCoeff,
                                                  float minCosTheta, float maxCosTheta) {
        if (rMin < maxCosTheta/minRobinCoeff) {
            return maxFloat;
        }

        float rBound = rMax/(1.0f - (maxCosTheta/(minRobinCoeff*rMin)));
        return rBound*rBound;
    }
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Wide operations

#ifdef FCPW_USE_ENOKI

template <size_t WIDTH>
inline void RobinLineSegmentBound::computeSquaredStarRadiusBound(const enokiVector2& sc, FloatP<WIDTH>& r2, const FloatP<WIDTH>& maxRobinCoeff,
                                                                 const MaskP<WIDTH> activeRobin, const Vector2P<WIDTH>& viewDirClosest,
                                                                 const FloatP<WIDTH>& closestPtDist, const FloatP<WIDTH>& closestPtDist2,
                                                                 const Vector2P<WIDTH>& viewDirFarthest, const FloatP<WIDTH>& farthestPtDist,
                                                                 const Vector2P<WIDTH>& planePt, const Vector2P<WIDTH>& planeNormal) {
    Vector2P<WIDTH> planeClosestPt;
    FloatP<WIDTH> h = findClosestPointWidePlane<WIDTH, 2>(planePt, planeNormal, sc, planeClosestPt);
    FloatP<WIDTH> cosUpperBound = enoki::abs(enoki::dot(viewDirClosest*enoki::rcp(closestPtDist), planeNormal));
    FloatP<WIDTH> cosLowerBound = enoki::abs(enoki::dot(viewDirFarthest*enoki::rcp(farthestPtDist), planeNormal));
    FloatP<WIDTH> hMaxRobinCoeff = h*maxRobinCoeff;
    FloatP<WIDTH> cosLine = enoki::sqrt(hMaxRobinCoeff)/SQRT_2;
    FloatP<WIDTH> cosLineSegment = enoki::clamp(cosLine, cosLowerBound, cosUpperBound);
    FloatP<WIDTH> cosLineSegment2 = cosLineSegment*cosLineSegment;
    FloatP<WIDTH> lineSegmentRadius = (h*enoki::rcp(cosLineSegment))*enoki::exp(cosLineSegment2*enoki::rcp(hMaxRobinCoeff));
    FloatP<WIDTH> lineSegmentRadius2 = lineSegmentRadius*lineSegmentRadius;
    enoki::masked(r2, activeRobin && r2 > closestPtDist2) = enoki::min(r2, lineSegmentRadius2);
}

template <size_t WIDTH>
inline void RobinTriangleBound::computeSquaredStarRadiusBound(const enokiVector3& sc, FloatP<WIDTH>& r2, const FloatP<WIDTH>& maxRobinCoeff,
                                                              const MaskP<WIDTH> activeRobin, const Vector3P<WIDTH>& viewDirClosest,
                                                              const FloatP<WIDTH>& closestPtDist, const FloatP<WIDTH>& closestPtDist2,
                                                              const Vector3P<WIDTH>& viewDirFarthest, const FloatP<WIDTH>& farthestPtDist,
                                                              const Vector3P<WIDTH>& planePt, const Vector3P<WIDTH>& planeNormal) {
    Vector3P<WIDTH> planeClosestPt;
    FloatP<WIDTH> h = findClosestPointWidePlane<WIDTH, 3>(planePt, planeNormal, sc, planeClosestPt);
    FloatP<WIDTH> cosUpperBound = enoki::abs(enoki::dot(viewDirClosest*enoki::rcp(closestPtDist), planeNormal));
    FloatP<WIDTH> cosLowerBound = enoki::abs(enoki::dot(viewDirFarthest*enoki::rcp(farthestPtDist), planeNormal));
    FloatP<WIDTH> hMaxRobinCoeff = h*maxRobinCoeff;
    FloatP<WIDTH> maxCosForBound = enoki::sqrt(hMaxRobinCoeff);
    FloatP<WIDTH> cosPlane = maxCosForBound/SQRT_3;
    FloatP<WIDTH> cosTriangle = enoki::clamp(cosPlane, cosLowerBound, cosUpperBound);
    FloatP<WIDTH> cosTriangle2 = cosTriangle*cosTriangle;
    FloatP<WIDTH> triangleRadius = h*hMaxRobinCoeff*enoki::rcp(cosTriangle*(hMaxRobinCoeff - cosTriangle2));
    FloatP<WIDTH> triangleRadius2 = triangleRadius*triangleRadius;
    enoki::masked(r2, activeRobin && maxCosForBound >= cosLowerBound && r2 > closestPtDist2) = enoki::min(r2, triangleRadius2);
}

template <size_t DIM>
struct RobinMbvhNodeBound {
    // computes the minimum squared star radius bound
    static FloatP<FCPW_MBVH_BRANCHING_FACTOR> computeMinSquaredStarRadiusBound(const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMin,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMax,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minCosTheta,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxCosTheta) {
        std::cerr << "RobinMbvhNodeBound::computeMinSquaredStarRadiusBound(): DIM: " << DIM << " not supported" << std::endl;
        exit(EXIT_FAILURE);

        return 0.0f;
    }

    // computes the maximum squared star radius bound
    static FloatP<FCPW_MBVH_BRANCHING_FACTOR> computeMaxSquaredStarRadiusBound(const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMin,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMax,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minCosTheta,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxCosTheta) {
        std::cerr << "RobinMbvhNodeBound::computeMaxSquaredStarRadiusBound(): DIM: " << DIM << " not supported" << std::endl;
        exit(EXIT_FAILURE);

        return 0.0f;
    }
};

template <>
struct RobinMbvhNodeBound<2> {
    // computes the minimum squared star radius bound
    static FloatP<FCPW_MBVH_BRANCHING_FACTOR> computeMinSquaredStarRadiusBound(const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMin,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMax,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minCosTheta,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxCosTheta) {
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> rBound = rMin*enoki::exp(minCosTheta*enoki::rcp(maxRobinCoeff*rMax));
        return rBound*rBound;
    }

    // computes the maximum squared star radius bound
    static FloatP<FCPW_MBVH_BRANCHING_FACTOR> computeMaxSquaredStarRadiusBound(const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMin,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMax,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minCosTheta,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxCosTheta) {
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> rBound = rMax*enoki::exp(maxCosTheta*enoki::rcp(minRobinCoeff*rMin));
        return rBound*rBound;
    }
};

template <>
struct RobinMbvhNodeBound<3> {
    // computes the minimum squared star radius bound
    static FloatP<FCPW_MBVH_BRANCHING_FACTOR> computeMinSquaredStarRadiusBound(const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMin,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMax,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minCosTheta,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxCosTheta) {
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> cosThetaOverRobinCoeff = minCosTheta*enoki::rcp(maxRobinCoeff);
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> rBound = rMin*enoki::rcp(1.0f - cosThetaOverRobinCoeff*enoki::rcp(rMax));
        enoki::masked(rBound, rMax < cosThetaOverRobinCoeff) = maxFloat;

        return rBound*rBound;
    }

    // computes the maximum squared star radius bound
    static FloatP<FCPW_MBVH_BRANCHING_FACTOR> computeMaxSquaredStarRadiusBound(const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMin,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& rMax,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxRobinCoeff,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& minCosTheta,
                                                                               const FloatP<FCPW_MBVH_BRANCHING_FACTOR>& maxCosTheta) {
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> cosThetaOverRobinCoeff = maxCosTheta*enoki::rcp(minRobinCoeff);
        FloatP<FCPW_MBVH_BRANCHING_FACTOR> rBound = rMax*enoki::rcp(1.0f - cosThetaOverRobinCoeff*enoki::rcp(rMin));
        enoki::masked(rBound, rMin < cosThetaOverRobinCoeff) = maxFloat;

        return rBound*rBound;
    }
};

#endif

} // zombie

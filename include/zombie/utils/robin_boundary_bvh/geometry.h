// This file extends the 'LineSegment' and 'Triangle' primitive structures from
// the FCPW library to support Robin boundary conditions. Users of Zombie need
// not interact with this file directly.

#pragma once

#include <fcpw/fcpw.h>

#define SQRT_2 1.4142135623730950488016887242097f
#define SQRT_3 1.7320508075688772935274463415059f

namespace zombie {

using namespace fcpw;

template<typename PrimitiveBound>
class RobinLineSegment: public LineSegment {
public:
    // constructor
    RobinLineSegment();

    // computes the squared Robin sphere radius
    void computeSquaredStarRadius(BoundingSphere<2>& s,
                                  bool flipNormalOrientation,
                                  float silhouettePrecision,
                                  bool performSilhouetteTests=true) const;

    // members
    Vector2 n[2];
    float minRobinCoeff;
    float maxRobinCoeff;
    bool hasAdjacentFace[2];
    bool ignoreAdjacentFace[2];
    typedef PrimitiveBound Bound;
};

template<typename PrimitiveBound>
class RobinTriangle: public Triangle {
public:
    // constructor
    RobinTriangle();

    // computes the squared Robin sphere radius
    void computeSquaredStarRadius(BoundingSphere<3>& s,
                                  bool flipNormalOrientation,
                                  float silhouettePrecision,
                                  bool performSilhouetteTests=true) const;

    // members
    Vector3 n[3];
    float minRobinCoeff;
    float maxRobinCoeff;
    bool hasAdjacentFace[3];
    bool ignoreAdjacentFace[3];
    typedef PrimitiveBound Bound;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template<size_t DIM>
inline float findClosestPointPlane(const Vector<DIM>& p, const Vector<DIM>& n,
                                   const Vector<DIM>& x, Vector<DIM>& pt)
{
    float t = n.dot(p - x)/n.dot(n);
    pt = x - t*n;

    return std::fabs(t);
}

inline float findFarthestPointLineSegment(const Vector2& pa, const Vector2& pb,
                                          const Vector2& x, Vector2& pt)
{
    float da = (x - pa).squaredNorm();
    float db = (x - pb).squaredNorm();

    if (da > db) {
        pt = pa;
        return std::sqrt(da);
    }

    pt = pb;
    return std::sqrt(db);
}

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
    template<size_t WIDTH>
    static void computeSquaredStarRadiusBound(const enokiVector2& sc, FloatP<WIDTH>& r2, const FloatP<WIDTH>& maxRobinCoeff,
                                              const MaskP<WIDTH> activeRobin, const Vector2P<WIDTH>& viewDirClosest,
                                              const FloatP<WIDTH>& closestPtDist, const FloatP<WIDTH>& closestPtDist2,
                                              const Vector2P<WIDTH>& viewDirFarthest, const FloatP<WIDTH>& farthestPtDist,
                                              const Vector2P<WIDTH>& planePt, const Vector2P<WIDTH>& planeNormal);
#endif
};

template<typename PrimitiveBound>
inline RobinLineSegment<PrimitiveBound>::RobinLineSegment():
LineSegment(),
n{Vector2::Zero(), Vector2::Zero()},
minRobinCoeff(minFloat),
maxRobinCoeff(maxFloat)
{
    for (size_t i = 0; i < 2; ++i) {
        hasAdjacentFace[i] = false;
        ignoreAdjacentFace[i] = true;
    }
}

template<typename PrimitiveBound>
inline void RobinLineSegment<PrimitiveBound>::computeSquaredStarRadius(BoundingSphere<2>& s,
                                                                       bool flipNormalOrientation,
                                                                       float silhouettePrecision,
                                                                       bool performSilhouetteTests) const
{
    const Vector2& pa = soup->positions[indices[0]];
    const Vector2& pb = soup->positions[indices[1]];

    // find closest point on line segment
    Vector2 closestPt;
    float tLineSegment;
    float closestPtDist = findClosestPointLineSegment<2>(pa, pb, s.c, closestPt, tLineSegment);
    float closestPtDist2 = closestPtDist*closestPtDist;
    if (closestPtDist2 > s.r2) return;

    if (maxRobinCoeff < maxFloat - epsilon) {
        // perform silhouette tests for Neuamnn and Robin cases
        Vector2 n0 = normal(true);
        Vector2 viewDirClosest = s.c - closestPt;

        if (performSilhouetteTests) {
            for (int j = 0; j < 2; j++) {
                if (!ignoreAdjacentFace[j]) {
                    const Vector2& pj = soup->positions[indices[j]];
                    const Vector2& nj = n[j];

                    bool isSilhouette = !hasAdjacentFace[j];
                    if (!isSilhouette) {
                        if (j == 0) {
                            isSilhouette = isSilhouetteVertex(n0, nj, viewDirClosest, closestPtDist,
                                                              flipNormalOrientation, silhouettePrecision);

                        } else {
                            isSilhouette = isSilhouetteVertex(nj, n0, viewDirClosest, closestPtDist,
                                                              flipNormalOrientation, silhouettePrecision);
                        }
                    }

                    if (isSilhouette) {
                        float silhouettePtDist2 = (s.c - pj).squaredNorm();
                        s.r2 = std::min(s.r2, silhouettePtDist2);
                    }
                }
            }
        }

        if (maxRobinCoeff > epsilon) {
            // [Robin Case]: shrink radius to ensure bounded reflectance
            if (closestPtDist < epsilon) {
                s.r2 = closestPtDist;

            } else if (s.r2 > closestPtDist2) {
                Vector2 farthestPt;
                float farthestPtDist = findFarthestPointLineSegment(pa, pb, s.c, farthestPt);
                Vector2 viewDirFarthest = s.c - farthestPt;
                PrimitiveBound::computeSquaredStarRadiusBound(s, maxRobinCoeff, viewDirClosest, closestPtDist,
                                                              viewDirFarthest, farthestPtDist, pa, n0);
            }

        } else {
            // [Neumann case]: dist to closest visibility silhouette
        }

    } else {
        // [Dirichlet case]: dist to closest point on line segment
        s.r2 = std::min(s.r2, closestPtDist2);
    }
}

inline float findFarthestPointTriangle(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                                       const Vector3& x, Vector3& pt)
{
    float da = (x - pa).squaredNorm();
    float db = (x - pb).squaredNorm();
    float dc = (x - pc).squaredNorm();

    if (da >= db && da >= dc) {
        pt = pa;
        return std::sqrt(da);

    } else if (db >= da && db >= dc) {
        pt = pb;
        return std::sqrt(db);
    }

    pt = pc;
    return std::sqrt(dc);
}

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
    template<size_t WIDTH>
    static void computeSquaredStarRadiusBound(const enokiVector3& sc, FloatP<WIDTH>& r2, const FloatP<WIDTH>& maxRobinCoeff,
                                              const MaskP<WIDTH> activeRobin, const Vector3P<WIDTH>& viewDirClosest,
                                              const FloatP<WIDTH>& closestPtDist, const FloatP<WIDTH>& closestPtDist2,
                                              const Vector3P<WIDTH>& viewDirFarthest, const FloatP<WIDTH>& farthestPtDist,
                                              const Vector3P<WIDTH>& planePt, const Vector3P<WIDTH>& planeNormal);
#endif
};

template<typename PrimitiveBound>
inline RobinTriangle<PrimitiveBound>::RobinTriangle():
Triangle(),
n{Vector3::Zero(), Vector3::Zero(), Vector3::Zero()},
minRobinCoeff(minFloat),
maxRobinCoeff(maxFloat)
{
    for (size_t i = 0; i < 3; ++i) {
        hasAdjacentFace[i] = false;
        ignoreAdjacentFace[i] = true;
    }
}

template<typename PrimitiveBound>
inline void RobinTriangle<PrimitiveBound>::computeSquaredStarRadius(BoundingSphere<3>& s,
                                                                    bool flipNormalOrientation,
                                                                    float silhouettePrecision,
                                                                    bool performSilhouetteTests) const
{
    const Vector3& pa = soup->positions[indices[0]];
    const Vector3& pb = soup->positions[indices[1]];
    const Vector3& pc = soup->positions[indices[2]];

    // find closest point on triangle
    Vector3 closestPt;
    Vector2 tTriangle;
    float closestPtDist = findClosestPointTriangle(pa, pb, pc, s.c, closestPt, tTriangle);
    float closestPtDist2 = closestPtDist*closestPtDist;
    if (closestPtDist2 > s.r2) return;

    if (maxRobinCoeff < maxFloat - epsilon) {
        // perform silhouette tests for Neuamnn and Robin cases
        Vector3 n0 = normal(true);
        Vector3 viewDirClosest = s.c - closestPt;

        if (performSilhouetteTests) {
            for (int j = 0; j < 3; j++) {
                if (!ignoreAdjacentFace[j]) {
                    const Vector3& pj = soup->positions[indices[j]];
                    const Vector3& pk = soup->positions[indices[(j + 1)%3]];
                    const Vector3& nj = n[j];

                    bool isSilhouette = !hasAdjacentFace[j];
                    if (!isSilhouette) {
                        isSilhouette = isSilhouetteEdge(pj, pk, nj, n0, viewDirClosest, closestPtDist,
                                                        flipNormalOrientation, silhouettePrecision);
                    }

                    if (isSilhouette) {
                        Vector3 pt;
                        float t;
                        float silhouettePtDist = findClosestPointLineSegment<3>(pj, pk, s.c, pt, t);
                        s.r2 = std::min(s.r2, silhouettePtDist*silhouettePtDist);
                    }
                }
            }
        }

        if (maxRobinCoeff > epsilon) {
            // [Robin Case]: shrink radius to ensure bounded reflectance
            if (closestPtDist < epsilon) {
                s.r2 = closestPtDist;

            } else if (s.r2 > closestPtDist2) {
                Vector3 farthestPt;
                float farthestPtDist = findFarthestPointTriangle(pa, pb, pc, s.c, farthestPt);
                Vector3 viewDirFarthest = s.c - farthestPt;
                PrimitiveBound::computeSquaredStarRadiusBound(s, maxRobinCoeff, viewDirClosest, closestPtDist,
                                                              viewDirFarthest, farthestPtDist, pa, n0);
            }

        } else {
            // [Neumann case]: dist to closest visibility silhouette
        }

    } else {
        // [Dirichlet case]: dist to closest point on triangle
        s.r2 = std::min(s.r2, closestPtDist2);
    }
}

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Wide operations

#ifdef FCPW_USE_ENOKI

template<size_t WIDTH, size_t DIM>
inline FloatP<WIDTH> findClosestPointWidePlane(const VectorP<WIDTH, DIM>& p, const VectorP<WIDTH, DIM>& n,
                                               const VectorP<WIDTH, DIM>& x, VectorP<WIDTH, DIM>& pt)
{
    FloatP<WIDTH> t = enoki::dot(n, p - x)*enoki::rcp(enoki::dot(n, n));
    pt = x - t*n;

    return enoki::abs(t);
}

template<size_t WIDTH>
inline FloatP<WIDTH> findFarthestPointWideLineSegment(const Vector2P<WIDTH>& pa, const Vector2P<WIDTH>& pb,
                                                      const Vector2P<WIDTH>& x, Vector2P<WIDTH>& pt)
{
    FloatP<WIDTH> da = enoki::squared_norm(x - pa);
    FloatP<WIDTH> db = enoki::squared_norm(x - pb);

    pt = pb;
    FloatP<WIDTH> d = enoki::sqrt(db);
    MaskP<WIDTH> mask = da > db;
    enoki::masked(pt, mask) = pa;
    enoki::masked(d, mask) = enoki::sqrt(da);

    return d;
}

template<size_t WIDTH>
inline FloatP<WIDTH> findFarthestPointWideTriangle(const Vector3P<WIDTH>& pa, const Vector3P<WIDTH>& pb,
                                                   const Vector3P<WIDTH>& pc, const Vector3P<WIDTH>& x,
                                                   Vector3P<WIDTH>& pt)
{
    FloatP<WIDTH> da = enoki::squared_norm(x - pa);
    FloatP<WIDTH> db = enoki::squared_norm(x - pb);
    FloatP<WIDTH> dc = enoki::squared_norm(x - pc);

    pt = pc;
    FloatP<WIDTH> d = enoki::sqrt(dc);

    MaskP<WIDTH> mask1 = da >= db && da >= dc;
    enoki::masked(pt, mask1) = pa;
    enoki::masked(d, mask1) = enoki::sqrt(da);

    MaskP<WIDTH> mask2 = db >= da && db >= dc;
    enoki::masked(pt, mask2) = pb;
    enoki::masked(d, mask2) = enoki::sqrt(db);

    return d;
}

template<size_t WIDTH>
inline Vector2P<WIDTH> computeNormalWideLineSegment(const Vector2P<WIDTH>& pa,
                                                    const Vector2P<WIDTH>& pb)
{
    Vector2P<WIDTH> v = pb - pa;
    Vector2P<WIDTH> n;
    n[0] = v[1];
    n[1] = -v[0];

    return enoki::normalize(n);
}

template<size_t WIDTH>
inline Vector3P<WIDTH> computeNormalWideTriangle(const Vector3P<WIDTH>& pa,
                                                 const Vector3P<WIDTH>& pb,
                                                 const Vector3P<WIDTH>& pc)
{
    Vector3P<WIDTH> v1 = pb - pa;
    Vector3P<WIDTH> v2 = pc - pa;

    return enoki::normalize(enoki::cross(v1, v2));
}

template<size_t WIDTH, size_t DIM, typename PrimitiveBound>
struct RobinWidePrimitive {
    // computes the squared Robin sphere radius
    static FloatP<WIDTH> computeSquaredStarRadiusWidePrimitive(
        const VectorP<WIDTH, DIM> *p, const VectorP<WIDTH, DIM> *n, const FloatP<WIDTH>& maxRobinCoeff,
        const MaskP<WIDTH> *hasAdjacentFace, const MaskP<WIDTH> *ignoreAdjacentFace,
        const enokiVector<DIM>& sc, float sr2, bool flipNormalOrientation,
        float silhouettePrecision, bool performSilhouetteTests=true) {
        std::cerr << "computeSquaredStarRadiusWidePrimitive(): WIDTH: " << WIDTH << " DIM: " << DIM << " not supported" << std::endl;
        exit(EXIT_FAILURE);

        return 0.0f;
    }
};

template<size_t WIDTH>
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

template<size_t WIDTH, typename PrimitiveBound>
struct RobinWidePrimitive<WIDTH, 2, PrimitiveBound> {
    // computes the squared Robin sphere radius
    static FloatP<WIDTH> computeSquaredStarRadiusWidePrimitive(
        const Vector2P<WIDTH> *p, const Vector2P<WIDTH> *n, const FloatP<WIDTH>& maxRobinCoeff,
        const MaskP<WIDTH> *hasAdjacentFace, const MaskP<WIDTH> *ignoreAdjacentFace,
        const enokiVector2& sc, float sr2, bool flipNormalOrientation,
        float silhouettePrecision, bool performSilhouetteTests=true) {
        const Vector2P<WIDTH>& pa = p[0];
        const Vector2P<WIDTH>& pb = p[1];
        FloatP<WIDTH> r2 = sr2;

        // find closest point on wide line segment
        Vector2P<WIDTH> closestPt;
        FloatP<WIDTH> tLineSegment;
        FloatP<WIDTH> closestPtDist = findClosestPointWideLineSegment<WIDTH, 2>(pa, pb, sc, closestPt, tLineSegment);
        FloatP<WIDTH> closestPtDist2 = closestPtDist*closestPtDist;
        MaskP<WIDTH> active = closestPtDist2 <= r2;

        if (enoki::any(active)) {
            // [Dirichlet case]: dist to closest point on line segment
            MaskP<WIDTH> isNotDirichlet = maxRobinCoeff < maxFloat - epsilon;
            enoki::masked(r2, active && ~isNotDirichlet) = enoki::min(r2, closestPtDist2);

            MaskP<WIDTH> activeNotDirichlet = active && isNotDirichlet;
            if (enoki::any(activeNotDirichlet)) {
                // [Neumann case]: dist to closest visibility silhouette
                Vector2P<WIDTH> n0 = computeNormalWideLineSegment<WIDTH>(pa, pb);
                Vector2P<WIDTH> viewDirClosest = sc - closestPt;

                if (performSilhouetteTests) {
                    for (int j = 0; j < 2; j++) {
                        MaskP<WIDTH> performTest = activeNotDirichlet && ~ignoreAdjacentFace[j];
                        if (enoki::any(performTest)) {
                            const Vector2P<WIDTH>& pj = p[j];
                            const Vector2P<WIDTH>& nj = n[j];

                            MaskP<WIDTH> isSilhouette = ~hasAdjacentFace[j];
                            if (j == 0) {
                                enoki::masked(isSilhouette, hasAdjacentFace[j]) = isWideSilhouetteVertex<WIDTH>(
                                    n0, nj, viewDirClosest, closestPtDist, flipNormalOrientation, silhouettePrecision);

                            } else {
                                enoki::masked(isSilhouette, hasAdjacentFace[j]) = isWideSilhouetteVertex<WIDTH>(
                                    nj, n0, viewDirClosest, closestPtDist, flipNormalOrientation, silhouettePrecision);
                            }

                            MaskP<WIDTH> activeSilhouette = performTest && isSilhouette;
                            if (enoki::any(activeSilhouette)) {
                                FloatP<WIDTH> silhouettePtDist2 = enoki::squared_norm(sc - pj);
                                enoki::masked(r2, activeSilhouette) = enoki::min(r2, silhouettePtDist2);
                            }
                        }
                    }
                }

                // [Robin Case]: shrink radius to ensure bounded reflectance
                MaskP<WIDTH> isNotNeumann = maxRobinCoeff > epsilon;
                MaskP<WIDTH> activeRobin = activeNotDirichlet && isNotNeumann;
                MaskP<WIDTH> activeOnRobinBoundary = activeRobin && closestPtDist < epsilon;
                if (enoki::any(activeOnRobinBoundary)) {
                    enoki::masked(r2, activeOnRobinBoundary) = closestPtDist;
                }

                if (enoki::any(activeRobin && r2 > closestPtDist2)) {
                    Vector2P<WIDTH> farthestPt;
                    FloatP<WIDTH> farthestPtDist = findFarthestPointWideLineSegment<WIDTH>(pa, pb, sc, farthestPt);
                    Vector2P<WIDTH> viewDirFarthest = sc - farthestPt;
                    PrimitiveBound::computeSquaredStarRadiusBound(sc, r2, maxRobinCoeff, activeRobin,
                                                                  viewDirClosest, closestPtDist, closestPtDist2,
                                                                  viewDirFarthest, farthestPtDist, pa, n0);
                }
            }
        }

        return r2;
    }
};

template<size_t WIDTH>
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

template<size_t WIDTH, typename PrimitiveBound>
struct RobinWidePrimitive<WIDTH, 3, PrimitiveBound> {
    // computes the squared Robin sphere radius
    static FloatP<WIDTH> computeSquaredStarRadiusWidePrimitive(
        const Vector3P<WIDTH> *p, const Vector3P<WIDTH> *n, const FloatP<WIDTH>& maxRobinCoeff,
        const MaskP<WIDTH> *hasAdjacentFace, const MaskP<WIDTH> *ignoreAdjacentFace,
        const enokiVector3& sc, float sr2, bool flipNormalOrientation,
        float silhouettePrecision, bool performSilhouetteTests=true) {
        const Vector3P<WIDTH>& pa = p[0];
        const Vector3P<WIDTH>& pb = p[1];
        const Vector3P<WIDTH>& pc = p[2];
        FloatP<WIDTH> r2 = sr2;

        // find closest point on triangle
        Vector3P<WIDTH> closestPt;
        Vector2P<WIDTH> tTriangle;
        FloatP<WIDTH> closestPtDist = findClosestPointWideTriangle<WIDTH>(pa, pb, pc, sc, closestPt, tTriangle);
        FloatP<WIDTH> closestPtDist2 = closestPtDist*closestPtDist;
        MaskP<WIDTH> active = closestPtDist2 <= r2;

        if (enoki::any(active)) {
            // [Dirichlet case]: dist to closest point on triangle
            MaskP<WIDTH> isNotDirichlet = maxRobinCoeff < maxFloat - epsilon;
            enoki::masked(r2, active && ~isNotDirichlet) = enoki::min(r2, closestPtDist2);

            MaskP<WIDTH> activeNotDirichlet = active && isNotDirichlet;
            if (enoki::any(activeNotDirichlet)) {
                // [Neumann case]: dist to closest visibility silhouette
                Vector3P<WIDTH> n0 = computeNormalWideTriangle<WIDTH>(pa, pb, pc);
                Vector3P<WIDTH> viewDirClosest = sc - closestPt;

                if (performSilhouetteTests) {
                    for (int j = 0; j < 3; j++) {
                        MaskP<WIDTH> performTest = activeNotDirichlet && ~ignoreAdjacentFace[j];
                        if (enoki::any(performTest)) {
                            const Vector3P<WIDTH>& pj = p[j];
                            const Vector3P<WIDTH>& pk = p[(j + 1)%3];
                            const Vector3P<WIDTH>& nj = n[j];

                            MaskP<WIDTH> isSilhouette = ~hasAdjacentFace[j];
                            enoki::masked(isSilhouette, hasAdjacentFace[j]) = isWideSilhouetteEdge<WIDTH>(
                                pj, pk, nj, n0, viewDirClosest, closestPtDist, flipNormalOrientation, silhouettePrecision);

                            MaskP<WIDTH> activeSilhouette = performTest && isSilhouette;
                            if (enoki::any(activeSilhouette)) {
                                Vector3P<WIDTH> pt;
                                FloatP<WIDTH> t;
                                FloatP<WIDTH> silhouettePtDist = findClosestPointWideLineSegment<WIDTH, 3>(pj, pk, sc, pt, t);
                                FloatP<WIDTH> silhouettePtDist2 = silhouettePtDist*silhouettePtDist;
                                enoki::masked(r2, activeSilhouette) = enoki::min(r2, silhouettePtDist2);
                            }
                        }
                    }
                }

                // [Robin Case]: shrink radius to ensure bounded reflectance
                MaskP<WIDTH> isNotNeumann = maxRobinCoeff > epsilon;
                MaskP<WIDTH> activeRobin = activeNotDirichlet && isNotNeumann;
                MaskP<WIDTH> activeOnRobinBoundary = activeRobin && closestPtDist < epsilon;
                if (enoki::any(activeOnRobinBoundary)) {
                    enoki::masked(r2, activeOnRobinBoundary) = closestPtDist;
                }

                if (enoki::any(activeRobin && r2 > closestPtDist2)) {
                    Vector3P<WIDTH> farthestPt;
                    FloatP<WIDTH> farthestPtDist = findFarthestPointWideTriangle<WIDTH>(pa, pb, pc, sc, farthestPt);
                    Vector3P<WIDTH> viewDirFarthest = sc - farthestPt;
                    PrimitiveBound::computeSquaredStarRadiusBound(sc, r2, maxRobinCoeff, activeRobin,
                                                                  viewDirClosest, closestPtDist, closestPtDist2,
                                                                  viewDirFarthest, farthestPtDist, pa, n0);
                }
            }
        }

        return r2;
    }
};

#endif // FCPW_USE_ENOKI

} // namespace zombie

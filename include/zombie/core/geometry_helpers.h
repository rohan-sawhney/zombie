// This file implements helper functions for 2D line segments and 3D triangles.

#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace zombie {

template <size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

Vector2 computeLineSegmentNormal(const Vector2& pa, const Vector2& pb, bool normalize)
{
    Vector2 s = pb - pa;
    Vector2 n(s[1], -s[0]);

    return normalize ? n.normalized() : n;
}

float computeLineSegmentSurfaceArea(const Vector2& pa, const Vector2& pb)
{
    return computeLineSegmentNormal(pa, pb, false).norm();
}

float computeClosestPointOnLineSegment(const Vector2& pa, const Vector2& pb,
                                       const Vector2& x, Vector2& pt)
{
    Vector2 u = pb - pa;
    Vector2 v = x - pa;

    float c1 = u.dot(v);
    if (c1 <= 0.0f) {
        pt = pa;
        return (x - pa).norm();
    }

    float c2 = u.dot(u);
    if (c2 <= c1) {
        pt = pb;
        return (x - pb).norm();
    }

    float t = c1/c2;
    pt = pa + u*t;
    return (x - pt).norm();
}

float computeFarthestPointOnLineSegment(const Vector2& pa, const Vector2& pb,
                                        const Vector2& x, Vector2& pt) {
    float da = (x - pa).squaredNorm();
    float db = (x - pb).squaredNorm();

    if (da > db) {
        pt = pa;
        return std::sqrt(da);
    }

    pt = pb;
    return std::sqrt(db);
}

Vector2 samplePointOnLineSegment(const Vector2& pa, const Vector2& pb, float *u,
                                 Vector2& n, float& pdf) {
    Vector2 s = pb - pa;
    Vector2 pt = pa + u[0]*s;
    n = Vector2(s[1], -s[0]);
    float norm = n.norm();
    n /= norm;
    pdf = 1.0f/norm;

    return pt;
}

Vector3 computeTriangleNormal(const Vector3& pa, const Vector3& pb, const Vector3& pc, bool normalize) {
    Vector3 n = (pb - pa).cross(pc - pa);
    return normalize ? n.normalized() : n;
}

float computeTriangleSurfaceArea(const Vector3& pa, const Vector3& pb, const Vector3& pc) {
    return 0.5f*computeTriangleNormal(pa, pb, pc, false).norm();
}

float computeTriangleAngle(const Vector3& pa, const Vector3& pb, const Vector3& pc) {
    Vector3 u = (pb - pa).normalized();
    Vector3 v = (pc - pa).normalized();

    return std::acos(std::max(-1.0f, std::min(1.0f, u.dot(v))));
}

float computeClosestPointOnTriangle(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                                    const Vector3& x, Vector3& pt) {
    // check if x in vertex region outside pa
    Vector3 ab = pb - pa;
    Vector3 ac = pc - pa;
    Vector3 ax = x - pa;
    float d1 = ab.dot(ax);
    float d2 = ac.dot(ax);
    if (d1 <= 0.0f && d2 <= 0.0f) {
        // barycentric coordinates (1, 0, 0)
        pt = pa;
        return (x - pa).norm();
    }

    // check if x in vertex region outside pb
    Vector3 bx = x - pb;
    float d3 = ab.dot(bx);
    float d4 = ac.dot(bx);
    if (d3 >= 0.0f && d4 <= d3) {
        // barycentric coordinates (0, 1, 0)
        pt = pb;
        return (x - pb).norm();
    }

    // check if x in vertex region outside pc
    Vector3 cx = x - pc;
    float d5 = ab.dot(cx);
    float d6 = ac.dot(cx);
    if (d6 >= 0.0f && d5 <= d6) {
        // barycentric coordinates (0, 0, 1)
        pt = pc;
        return (x - pc).norm();
    }

    // check if x in edge region of ab, if so return projection of x onto ab
    float vc = d1*d4 - d3*d2;
    if (vc <= 0.0f && d1 >= 0.0f && d3 <= 0.0f) {
        // barycentric coordinates (1 - v, v, 0)
        float v = d1/(d1 - d3);
        pt = pa + ab*v;
        return (x - pt).norm();
    }

    // check if x in edge region of ac, if so return projection of x onto ac
    float vb = d5*d2 - d1*d6;
    if (vb <= 0.0f && d2 >= 0.0f && d6 <= 0.0f) {
        // barycentric coordinates (1 - w, 0, w)
        float w = d2/(d2 - d6);
        pt = pa + ac*w;
        return (x - pt).norm();
    }

    // check if x in edge region of bc, if so return projection of x onto bc
    float va = d3*d6 - d5*d4;
    if (va <= 0.0f && (d4 - d3) >= 0.0f && (d5 - d6) >= 0.0f) {
        // barycentric coordinates (0, 1 - w, w)
        float w = (d4 - d3)/((d4 - d3) + (d5 - d6));
        pt = pb + (pc - pb)*w;
        return (x - pt).norm();
    }

    // x inside face region. Compute pt through its barycentric coordinates (u, v, w)
    float denom = 1.0f/(va + vb + vc);
    float v = vb*denom;
    float w = vc*denom;
    pt = pa + ab*v + ac*w; //= u*a + v*b + w*c
    return (x - pt).norm();
}

float computeFarthestPointOnTriangle(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                                     const Vector3& x, Vector3& pt) {
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

Vector3 samplePointOnTriangle(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                              float *u, Vector3& n, float& pdf) {
    float u1 = std::sqrt(u[0]);
    float u2 = u[1];
    float a = 1.0f - u1;
    float b = u2*u1;
    float c = 1.0f - a - b;
    Vector3 pt = pa*a + pb*b + pc*c;
    n = (pb - pa).cross(pc - pa);
    float norm = n.norm();
    n /= norm;
    pdf = 2.0f/norm;

    return pt;
}

} // namespace zombie

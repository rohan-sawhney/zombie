// This file provides various utiiity functions, used internally by the algorithms
// implemented in Zombie, for generating random samples.

#pragma once

#define _USE_MATH_DEFINES
#define NOMINMAX
#include <algorithm>
#include <math.h>
#include <iostream>
#include <limits>
#include <memory>
#include <chrono>
#include <random>
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <vector>
#include "pcg32.h"

namespace zombie {

template<size_t DIM>
using Vector = Eigen::Matrix<float, DIM, 1>;
using Vector2 = Vector<2>;
using Vector3 = Vector<3>;

template <size_t DIM>
inline Vector<DIM> sampleUnitSphereUniform(float *u)
{
    std::cerr << "sampleUnitSphereUniform not implemented for DIM: " << DIM << std::endl;
    return Vector<DIM>::Zero();
}

template <>
inline Vector2 sampleUnitSphereUniform<2>(float *u)
{
    float phi = 2.0f*M_PI*u[0];

    return Vector2(std::cos(phi), std::sin(phi));
}

template <>
inline Vector3 sampleUnitSphereUniform<3>(float *u)
{
    float z = 1.0f - 2.0f*u[0];
    float r = std::sqrt(std::max(0.0f, 1.0f - z*z));
    float phi = 2.0f*M_PI*u[1];

    return Vector3(r*std::cos(phi), r*std::sin(phi), z);
}

template <size_t DIM>
inline float pdfSampleSphereUniform(float r)
{
    std::cerr << "pdfSampleSphereUniform not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float pdfSampleSphereUniform<2>(float r)
{
    return 1.0f/(2.0f*M_PI*r);
}

template <>
inline float pdfSampleSphereUniform<3>(float r)
{
    return 1.0f/(4.0f*M_PI*r*r);
}

template <size_t DIM>
inline Vector<DIM> sampleUnitBallUniform(float *u)
{
    std::cerr << "sampleUnitBallUniform not implemented for DIM: " << DIM << std::endl;
    return Vector<DIM>::Zero();
}

template <>
inline Vector2 sampleUnitBallUniform<2>(float *u)
{
    float r = std::sqrt(u[1]);
    return r*sampleUnitSphereUniform<2>(u);
}

template <>
inline Vector3 sampleUnitBallUniform<3>(float *u)
{
    float r = std::cbrt(u[2]);
    return r*sampleUnitSphereUniform<3>(u);
}

template <size_t DIM>
inline float pdfSampleBallUniform(float r)
{
    std::cerr << "pdfSampleBallUniform not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float pdfSampleBallUniform<2>(float r)
{
    return 1.0f/(M_PI*r*r);
}

template <>
inline float pdfSampleBallUniform<3>(float r)
{
    return 3.0f/(4.0f*M_PI*r*r*r);
}

template <size_t DIM>
inline Vector<DIM> sampleUnitHemisphereCosine(float *u)
{
    std::cerr << "sampleUnitHemisphereCosine not implemented for DIM: " << DIM << std::endl;
    return Vector<DIM>::Zero();
}

template <>
inline Vector2 sampleUnitHemisphereCosine<2>(float *u)
{
    float u1 = 2.0f*u[0] - 1.0f;
    float z = std::sqrt(std::max(0.0f, 1.0f - u1*u1));

    return Vector2(u1, z);
}

inline Vector2 sampleUnitDiskConcentric(float *u)
{
    // map uniform random numbers to [-1,1]^2
    float u1 = 2.0f*u[0] - 1.0f;
    float u2 = 2.0f*u[1] - 1.0f;

    // handle degeneracy at the origin
    if (u1 == 0 && u2 == 0) {
        return Vector2::Zero();
    }

    // apply concentric mapping to point
    float theta, r;
    if (std::fabs(u1) > std::fabs(u2)) {
        r = u1;
        theta = 0.25f*M_PI*(u2/u1);

    } else {
        r = u2;
        theta = 0.5f*M_PI*(1.0f - 0.5f*(u1/u2));
    }

    return r*Vector2(std::cos(theta), std::sin(theta));
}

template <>
inline Vector3 sampleUnitHemisphereCosine<3>(float *u)
{
    Vector2 d = sampleUnitDiskConcentric(u);
    float z = std::sqrt(std::max(0.0f, 1.0f - d.squaredNorm()));

    return Vector3(d(0), d(1), z);
}

template <size_t DIM>
inline float pdfSampleUnitHemisphereCosine(float cosTheta)
{
    std::cerr << "pdfSampleUnitHemisphereCosine not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float pdfSampleUnitHemisphereCosine<2>(float cosTheta)
{
    return cosTheta/2.0f;
}

template <>
inline float pdfSampleUnitHemisphereCosine<3>(float cosTheta)
{
    return cosTheta/M_PI;
}

template <size_t DIM>
inline void transformCoordinates(const Vector<DIM>& n, Vector<DIM>& d)
{
    std::cerr << "transformCoordinates not implemented for DIM: " << DIM << std::endl;
}

template <>
inline void transformCoordinates<2>(const Vector2& n, Vector2& d)
{
    // compute orthonormal basis
    Vector2 s(n[1], -n[0]);

    // transform
    d = d(0)*s + d(1)*n;
}

template <>
inline void transformCoordinates<3>(const Vector3& n, Vector3& d)
{
    // compute orthonormal basis; source: https://graphics.pixar.com/library/OrthonormalB/paper.pdf
    float sign = std::copysignf(1.0f, n[2]);
    const float a = -1.0f/(sign + n[2]);
    const float b = n[0]*n[1]*a;
    Vector3 b1(1.0f + sign*n[0]*n[0]*a, sign*b, -sign*n[0]);
    Vector3 b2(b, sign + n[1]*n[1]*a, -n[1]);

    // transform
    d = d(0)*b1 + d(1)*b2 + d(2)*n;
}

template <size_t DIM>
inline Vector<DIM> lineSegmentNormal(const Vector<DIM>& pa, const Vector<DIM>& pb, bool normalize)
{
    std::cerr << "lineSegmentNormal not implemented for DIM: " << DIM << std::endl;
    return Vector<DIM>::Zero();
}

template <>
inline Vector2 lineSegmentNormal<2>(const Vector2& pa, const Vector2& pb, bool normalize)
{
    Vector2 s = pb - pa;
    Vector2 n(s[1], -s[0]);

    return normalize ? n.normalized() : n;
}

template <size_t DIM>
inline float lineSegmentSurfaceArea(const Vector<DIM>& pa, const Vector<DIM>& pb)
{
    std::cerr << "lineSegmentSurfaceArea not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float lineSegmentSurfaceArea<2>(const Vector2& pa, const Vector2& pb)
{
    return lineSegmentNormal<2>(pa, pb, false).norm();
}

template <size_t DIM>
inline float sampleLineSegmentUniformly(const Vector<DIM>& pa, const Vector<DIM>& pb,
                                        float *u, Vector<DIM>& pt, Vector<DIM>& n)
{
    std::cerr << "sampleLineSegmentUniformly not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float sampleLineSegmentUniformly<2>(const Vector2& pa, const Vector2& pb,
                                           float *u, Vector2& pt, Vector2& n)
{
    Vector2 s = pb - pa;
    pt = pa + u[0]*s;
    n = Vector2(s[1], -s[0]);
    float norm = n.norm();
    n /= norm;

    return norm;
}

template <size_t DIM>
inline Vector<DIM> triangleNormal(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc, bool normalize)
{
    std::cerr << "triangleNormal not implemented for DIM: " << DIM << std::endl;
    return Vector<DIM>::Zero();
}

template <>
inline Vector3 triangleNormal<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc, bool normalize)
{
    Vector3 n = (pb - pa).cross(pc - pa);

    return normalize ? n.normalized() : n;
}

template <size_t DIM>
inline float triangleSurfaceArea(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc)
{
    std::cerr << "triangleSurfaceArea not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float triangleSurfaceArea<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc)
{
    return 0.5f*triangleNormal<3>(pa, pb, pc, false).norm();
}

template <size_t DIM>
inline float triangleAngle(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc)
{
    std::cerr << "triangleAngle not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float triangleAngle<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc)
{
    Vector3 u = (pb - pa).normalized();
    Vector3 v = (pc - pa).normalized();

    return std::acos(std::max(-1.0f, std::min(1.0f, u.dot(v))));
}

template <size_t DIM>
inline float sampleTriangleUniformly(const Vector<DIM>& pa, const Vector<DIM>& pb, const Vector<DIM>& pc,
                                     float *u, Vector<DIM>& pt, Vector<DIM>& n)
{
    std::cerr << "sampleTriangleUniformly not implemented for DIM: " << DIM << std::endl;
    return 0.0f;
}

template <>
inline float sampleTriangleUniformly<3>(const Vector3& pa, const Vector3& pb, const Vector3& pc,
                                        float *u, Vector3& pt, Vector3& n)
{
    float u1 = std::sqrt(u[0]);
    float u2 = u[1];
    float a = 1.0f - u1;
    float b = u2*u1;
    float c = 1.0f - a - b;
    pt = pa*a + pb*b + pc*c;
    n = (pb - pa).cross(pc - pa);
    float norm = n.norm();
    n /= norm;

    return 0.5f*norm;
}

// source: https://pbr-book.org/3ed-2018/Monte_Carlo_Integration/Sampling_Random_Variables
class CDFTable {
public:
    // builds a CDF table from the input list of non-negative weights
    float build(const std::vector<float>& weights) {
        // initialize table
        int nWeights = (int)weights.size();
        if (nWeights > 0) {
            table.resize(nWeights + 1);
            table[0] = 0.0f;

            for (int i = 1; i < nWeights + 1; i++) {
                table[i] = table[i - 1] + weights[i - 1];
            }

            float total = table[nWeights];
            if (total == 0.0f) {
                for (int i = 1; i < nWeights + 1; i++) {
                    table[i] = float(i)/float(nWeights);
                }

            } else {
                for (int i = 1; i < nWeights + 1; i++) {
                    table[i] /= total;
                }
            }

            return total;
        }

        return 0.0f;
    }

    // generate sample from table using one uniform sample in the range [0, 1)
    int sample(float u) const {
        int size = (int)table.size();
        int first = 0;
        int len = size;

        while (len > 0) {
            int half = len >> 1;
            int middle = first + half;

            // bisect range based on table value at the middle index
            if (table[middle] <= u) {
                first = middle + 1;
                len -= half + 1;

            } else {
                len = half;
            }
        }

        return std::clamp(first - 1, 0, size - 2);
    }

protected:
    // member
    std::vector<float> table;
};

// source: https://github.com/NVIDIAGameWorks/Falcor/blob/master/Source/Falcor/Utils/Sampling/AliasTable.cpp
class AliasTable {
public:
    // This builds an alias table via the O(N) algorithm from Vose 1991, "A linear algorithm for generating random
    // numbers with a given distribution," IEEE Transactions on Software Engineering 17(9), 972-975.
    //
    // Basic idea:  creating each alias table entry combines one overweighted sample and one underweighted sample
    // into one alias table entry plus a residual sample (the overweighted sample minus some of its weight).
    //
    // By first separating all inputs into 2 temporary buffer (one overweighted set, with weights above the
    // average; one underweighted set, with weights below average), we can simply walk through the lists once,
    // merging the first elements in each temporary buffer. The residual sample is interted into either the
    // overweighted or underweighted set, depending on its residual weight.
    //
    // The main complexity is dealing with corner cases, thanks to numerical precision issues, where you don't
    // have 2 valid entries to combine. By definition, in these corner cases, all remaining unhandled samples
    // actually have the average weight (within numerical precision limits)
    float build(std::vector<float> weights) {
        // use >= since we reserve 0xFFFFFFFFu as an invalid flag marker during construction
        if ((uint32_t)weights.size() >= std::numeric_limits<uint32_t>::max()) {
            std::cerr << "too many entries for alias table" << std::endl;
            return EXIT_FAILURE;
        }

        // our working set / intermediate buffers (underweight & overweight); initialize to "invalid"
        uint32_t nWeights = (uint32_t)weights.size();
        std::vector<uint32_t> lowIdx(nWeights, 0xFFFFFFFFu);
        std::vector<uint32_t> highIdx(nWeights, 0xFFFFFFFFu);

        // sum element weights
        float totalWeight = 0.0f;
        for (uint32_t i = 0; i < nWeights; i++) {
            totalWeight += weights[i];
        }

        // find the average weight
        float avgWeight = float(totalWeight/double(nWeights));

        // initialize working set. Inset inputs into our lists of above-average or below-average weight elements
        int lowCount = 0;
        int highCount = 0;
        for (uint32_t i = 0; i < nWeights; i++) {
            if (weights[i] < avgWeight) lowIdx[lowCount++] = i;
            else highIdx[highCount++] = i;
        }

        // create alias table entries by merging above- and below-average samples
        table.clear();
        table.resize(nWeights);
        for (uint32_t i = 0; i < nWeights; i++) {
            // usual case: we have an above-average and below-average sample we can combine into one alias table entry
            if ((lowIdx[i] != 0xFFFFFFFFu) && (highIdx[i] != 0xFFFFFFFFu)) {
                // create an alias table tuple:
                table[i] = {weights[lowIdx[i]]/avgWeight, highIdx[i], lowIdx[i]};

                // we've removed some weight from element highIdx[i]; update it's weight, then re-enter it
                // on the end of either the above-average or below-average lists
                float updatedWeight = (weights[lowIdx[i]] + weights[highIdx[i]]) - avgWeight;
                weights[highIdx[i]] = updatedWeight;
                if (updatedWeight < avgWeight) lowIdx[lowCount++] = highIdx[i];
                else highIdx[highCount++] = highIdx[i];
            }

            // the next two cases can only occur towards the end of table creation, because either:
            //    (a) all the remaining possible alias table entries have weight *exactly* equal to avgWeight,
            //        which means these alias table entries only have one input item that is selected
            //        with 100% probability
            //    (b) all the remaining alias table entires have *almost* avgWeight, but due to (compounding)
            //        precision issues throughout the process, they don't have *quite* that value. In this case
            //        treating these entries as having exactly avgWeight (as in case (a)) is the only right
            //        thing to do mathematically (other than re-generating the alias table using higher precision
            //        or trying to reduce catasrophic numerical cancellation in the "updatedWeight" computation above).
            else if (highIdx[i] != 0xFFFFFFFFu) {
                table[i] = {1.0f, highIdx[i], highIdx[i]};

            } else if (lowIdx[i] != 0xFFFFFFFFu) {
                table[i] = {1.0f, lowIdx[i], lowIdx[i]};
            }

            // if there is neither a highIdx[i] or lowIdx[i] for some array element(s). By construction,
            // this cannot occur (without some logic bug above)
            else {
                std::cerr << "alias table construction incorrect" << std::endl;
                return EXIT_FAILURE;
            }
        }

        return totalWeight;
    }

    // generate sample from table using two uniform samples in the range [0, 1)
    uint32_t sample(float u1, float u2) const {
        uint32_t tableSize = (uint32_t)table.size();
        uint32_t index = std::min(tableSize - 1, (uint32_t)(u1*tableSize));
        const AliasTableItem& item = table[index];

        return u2 >= item.threshold ? item.indexA : item.indexB;
    }

protected:
    // member
    struct AliasTableItem {
        float threshold; // if rand() < threshold, pick indexB (else pick indexA)
        uint32_t indexA; // the "redirect" index, if uniform sampling would overweight indexB
        uint32_t indexB; // the original / permutation index, sampled uniformly in [0...mCount-1]
    };
    std::vector<AliasTableItem> table;
};

/////////////////////////////////////////////////////////////////////////////////////////////////

// source: https://pbr-book.org/3ed-2018/Sampling_and_Reconstruction/Stratified_Sampling#LatinHypercube
// NOTE: sample quality reduces with increasing dimension
template <size_t DIM>
inline void generateStratifiedSamples(std::vector<float>& samples, int nSamples, pcg32& sampler)
{
    const float epsilon = std::numeric_limits<float>::epsilon();
    const float oneMinusEpsilon = 1.0f - epsilon;
    float invNSamples = 1.0f/nSamples;
    samples.resize(DIM*nSamples);

    // generate LHS samples along diagonal
    for (int i = 0; i < nSamples; ++i) {
        for (int j = 0; j < DIM; ++j) {
            float sj = (i + sampler.nextFloat())*invNSamples;
            samples[DIM*i + j] = std::min(sj, oneMinusEpsilon);
        }
    }

    // generate LHS samples in each dimension
    for (int i = 0; i < DIM; ++i) {
        for (int j = 0; j < nSamples; ++j) {
            int other = j + sampler.nextUInt(nSamples - j);
            std::swap(samples[DIM*j + i], samples[DIM*other + i]);
        }
    }
}

template <size_t DIM>
inline Vector<DIM> sampleUnitSphereUniform(pcg32& sampler)
{
    std::cerr << "sampleUnitSphereUniform not implemented for DIM: " << DIM << std::endl;
    return Vector<DIM>::Zero();
}

template <>
inline Vector2 sampleUnitSphereUniform<2>(pcg32& sampler)
{
    float u[1] = { sampler.nextFloat() };
    return sampleUnitSphereUniform<2>(u);
}

template <>
inline Vector3 sampleUnitSphereUniform<3>(pcg32& sampler)
{
    float u[2] = { sampler.nextFloat(), sampler.nextFloat() };
    return sampleUnitSphereUniform<3>(u);
}

} // zombie

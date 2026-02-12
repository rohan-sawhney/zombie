/*
	pbrt source code Copyright(c) 1998-2012 Matt Pharr and Greg Humphreys.

	This file is part of pbrt.

	Redistribution and use in source and binary forms, with or without
	modification, are permitted provided that the following conditions are
	met:

	- Redistributions of source code must retain the above copyright
	  notice, this list of conditions and the following disclaimer.

	- Redistributions in binary form must reproduce the above copyright
	  notice, this list of conditions and the following disclaimer in the
	  documentation and/or other materials provided with the distribution.

	THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS
	IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED
	TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
	PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
	HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
	SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
	LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
	DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
	THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
	(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
	OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

 */

#pragma once

#include <iostream>
#include <cmath>

#include "pde.h"

namespace sh
{
// Global Macros
template <typename T>
T* StackAlloc(std::size_t count)
{
    return static_cast<T*>(alloca(count * sizeof(T)));
}

// Spherical Harmonics Declarations
constexpr inline int SHTerms(int lmax)
{
    return (lmax + 1) * (lmax + 1);
}

constexpr inline int SHIndex(int l, int m)
{
    return l * l + l + m;
}

// Spherical Harmonics Local Definitions
static void legendrep(float x, int lmax, float* out)
{
#define P(l, m) out[SHIndex(l, m)]
    // Compute $m=0$ Legendre values using recurrence
    P(0, 0) = 1.f;
    P(1, 0) = x;
    for (int l = 2; l <= lmax; ++l)
    {
        P(l, 0) = ((2 * l - 1) * x * P(l - 1, 0) - (l - 1) * P(l - 2, 0)) / l;
    }

    // Compute $m=l$ edge using Legendre recurrence
    float neg   = -1.f;
    float dfact = 1.f;
    float xroot = std::sqrt(std::max(0.f, 1.f - x * x));
    float xpow  = xroot;
    for (int l = 1; l <= lmax; ++l)
    {
        P(l, l) = neg * dfact * xpow;

        neg *= -1.f;        // neg = (-1)^l
        dfact *= 2 * l + 1; // dfact = (2*l-1)!!
        xpow *= xroot;      // xpow = powf(1.f - x*x, float(l) * 0.5f);
    }

    // Compute $m=l-1$ edge using Legendre recurrence
    for (int l = 2; l <= lmax; ++l)
    {
        P(l, l - 1) = x * (2 * l - 1) * P(l - 1, l - 1);
    }

    // Compute $m=1, \ldots, l-2$ values using Legendre recurrence
    for (int l = 3; l <= lmax; ++l)
        for (int m = 1; m <= l - 2; ++m)
        {
            P(l, m) = ((2 * (l - 1) + 1) * x * P(l - 1, m) -
                       (l - 1 + m) * P(l - 2, m)) /
                (l - m);
        }
#if 0
		// wrap up with the negative m ones now
		// P(l,-m)(x) = -1^m (l-m)!/(l+m)! P(l,m)(x)
		for (int l = 1; l <= lmax; ++l) {
			float fa = 1.f, fb = fact(2 * l);
			// fa = fact(l+m), fb = fact(l-m)
			for (int m = -l; m < 0; ++m) {
				float neg = ((-m) & 0x1) ? -1.f : 1.f;
				P(l, m) = neg * fa / fb * P(l, -m);
				fb /= l - m;
				fa *= (l + m + 1) > 1 ? (l + m + 1) : 1.;
			}
		}
#endif
#undef P
}

static inline float fact(float v);
static inline float divfact(int a, int b);
static inline float K(int l, int m)
{
    return sqrt((2.f * l + 1.f) * 1 / (4 * M_PI) * divfact(l, m));
}

static inline float divfact(int a, int b)
{
    if (b == 0) return 1.f;
    float fa = a, fb = fabsf(b);
    float v = 1.f;
    for (float x = fa - fb + 1.f; x <= fa + fb; x += 1.f) v *= x;
    return 1.f / v;
}

// n!! = 1 if n==0 or 1, otherwise n * (n-2)!!
static float dfact(float v)
{
    if (v <= 1.f) return 1.f;
    return v * dfact(v - 2.f);
}

static inline float fact(float v)
{
    if (v <= 1.f) return 1.f;
    return v * fact(v - 1.f);
}

static void sinCosIndexed(float s, float c, int n, float* sout, float* cout)
{
    float si = 0, ci = 1;
    for (int i = 0; i < n; ++i)
    {
        // Compute $\sin{}i\phi$ and $\cos{}i\phi$ using recurrence
        *sout++     = si;
        *cout++     = ci;
        float oldsi = si;
        si          = si * c + ci * s;
        ci          = ci * c - oldsi * s;
    }
}

static inline float lambda(float l)
{
    return sqrt((4.f * M_PI) / (2.f * l + 1.));
}

// Spherical Harmonics Definitions
static inline void SHEvaluate(const zombie::Vector<3>& w, int lmax, float* out)
{
    if (lmax > 28)
    {
        std::cerr << "SHEvaluate() runs out of numerical precision for lmax > 28. "
                     "If you need more bands, try recompiling using doubles."
                  << std::endl;
        exit(1);
    }

    // Compute Legendre polynomial values for $\cos\theta$
    legendrep(w.z(), lmax, out);

    // Compute $K_l^m$ coefficients
    float* Klm = StackAlloc<float>(SHTerms(lmax));
    for (int l = 0; l <= lmax; ++l)
        for (int m = -l; m <= l; ++m) Klm[SHIndex(l, m)] = K(l, m);

    // Compute $\sin\phi$ and $\cos\phi$ values
    float *sins = StackAlloc<float>(lmax + 1), *coss = StackAlloc<float>(lmax + 1);
    float  xyLen = std::sqrt(std::max(0.0f, 1.f - w.z() * w.z()));
    if (xyLen == 0.f)
    {
        for (int i = 0; i <= lmax; ++i) sins[i] = 0.f;
        for (int i = 0; i <= lmax; ++i) coss[i] = 1.f;
    }
    else
        sinCosIndexed(w.y() / xyLen, w.x() / xyLen, lmax + 1, sins, coss);

    // Apply SH definitions to compute final $(l,m)$ values
    static const float sqrt2 = std::sqrt(2.f);
    for (int l = 0; l <= lmax; ++l)
    {
        for (int m = -l; m < 0; ++m)
        {
            out[SHIndex(l, m)] =
                sqrt2 * Klm[SHIndex(l, m)] * out[SHIndex(l, -m)] * sins[-m];
        }
        out[SHIndex(l, 0)] *= Klm[SHIndex(l, 0)];
        for (int m = 1; m <= l; ++m)
        {
            out[SHIndex(l, m)] *= sqrt2 * Klm[SHIndex(l, m)] * coss[m];
        }
    }
}
#undef ALLOCA
} // namespace sh

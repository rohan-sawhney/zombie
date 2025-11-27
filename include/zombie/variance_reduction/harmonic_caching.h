// This file implements the Harmonic Caching algorithm for reducing variance
// of the walk-on-spheres and walk-on-stars estimators at a set of user-selected
// evaluation points via sample caching and reuse.
//
// Resources:
// - Harmonic Caching for Walk on Spheres [2025]
//
// Note that this only contains basic (and mostly useful) implementation in the
// harmonic caching paper, and advanced features such as unbiased caching is not supported.

#pragma once

#include <shared_mutex>
#include <algorithm>
#include <cmath>

#include <zombie/point_estimation/walk_on_stars.h>
#include <zombie/core/sh.h>
#include <zombie/utils/octree.h>

namespace zombie
{
namespace bessel
{
/// \brief
///    Modified spherical bessel function of the first kind. Used for harmonic
/// series expansion in 3D.
static inline float ModifiedSphericalBesselFirst(float l, float x)
{
    /// https://mathworld.wolfram.com/ModifiedSphericalBesselFunctionoftheFirstKind.html.
    return std::sqrt(M_PI / (2 * x)) * std::cyl_bessel_if(l + 0.5, x);
}
} // namespace bessel

namespace hc
{
template <typename T, size_t DIM>
struct HarmonicCoefficients;

template <typename T, size_t DIM>
struct HarmonicCacheRecord
{
public:
    /// Harmonic coefficients.
    HarmonicCoefficients<T, DIM> coeffs;

    /// Cache record center.
    Vector<DIM> p;

    /// Closest distance to the boundary. This is the radius used for coefficients estimation.
    float rd{0};

public:
    HarmonicCacheRecord(int32_t runtimeOrders) :
        coeffs{runtimeOrders}
    {
    }
};

// Fourier coefficients.
template <typename T>
struct HarmonicCoefficients<T, 2>
{
    HarmonicCoefficients(int32_t runtimeOrders)
    {
        this->a0 = T{0};
        this->an.resize(runtimeOrders, T{0});
        this->bn.resize(runtimeOrders, T{0});
    }

    /// \brief
    ///    Divide MC estimates of a0, an, bn by numValidSamples. This is
    /// important if you have valid walk within maximum length < totalMCWalks for
    /// this codebase.
    void performMCDivisionForNumValidSamples(int32_t numValidSamples)
    {
        if (numValidSamples == 0)
            return;

        this->a0 /= static_cast<float>(numValidSamples);
        for (int i = 0; i < this->an.size(); ++i)
        {
            this->an[i] /= static_cast<float>(numValidSamples);
            this->bn[i] /= static_cast<float>(numValidSamples);
        }
    }

    /// \brief
    ///    Get the maximum truncation order.
    /// an.size() == 0, then 0 order. an.size() == 5, then order 0...5
    int32_t GetFourierOrders() const noexcept
    {
        return this->an.size();
    }

    /// \brief
    ///    Return solution at r=0.
    T returnSolutionAtOrigin() const noexcept
    {
        /// For screened equations, is this true?
        return this->a0;
    }

public:
    T              a0;
    std::vector<T> an; /// starting from a1.So an[0] is actually a1.
    std::vector<T> bn; /// as above.
};

template <typename T>
struct HarmonicCoefficients<T, 3>
{
    HarmonicCoefficients(int32_t runtimeOrders) :
        numOrders{runtimeOrders}
    {
        this->alm.resize(SHTerms(runtimeOrders), T{0});
    }

    /// \brief
    ///    Divide MC estimates of a0, an, bn by numValidSamples. This is
    /// important if you have valid walk within maximum length < totalMCWalks for
    /// this codebase.
    void performMCDivisionForNumValidSamples(int32_t numValidSamples)
    {
        if (numValidSamples == 0)
            return;
        for (int i = 0; i < SHTerms(numOrders); ++i)
        {
            this->alm[i] /= static_cast<float>(numValidSamples);
        }
    }

    /// \brief
    ///    This is SH order.
    int32_t GetFourierOrders() const noexcept
    {
        return this->numOrders;
    }

    /// \brief
    ///    Return solution at r=0. Note that in 3D, u(0)=a0/(2*sqrt(pi)) instead for both screened and non-screened equations.
    T returnSolutionAtOrigin() const noexcept
    {
        return this->alm[0] / (2.0f * std::sqrt(M_PI));
    }

public:
    /// Maximum order for reconstruction.
    int32_t numOrders{10};

    /// Storage of SH coefficients.
    std::vector<T> alm;
};

template <typename T, size_t DIM>
struct HCReconstructionProcess
{
    HCReconstructionProcess(const Vector<DIM>& evaluationPt,
                            const PDE<T, DIM>& pde,
                            float              minWeight,
                            bool               PDEIgnoreSourceTerm,
                            int32_t            nWalksForSourceTermReconstruction,
                            pcg32&             sampler) :
        L{0},
        weight{0.0},
        pt{evaluationPt},
        pde{pde},
        minWeight{minWeight},
        m_bPDEIgnoreSourceTerm{PDEIgnoreSourceTerm},
        m_nWalksForSourceTermReconstruction{nWalksForSourceTermReconstruction},
        sampler{sampler}
    {
    }

    bool operator()(const HarmonicCacheRecord<T, DIM>* record)
    {
        double splatR = (this->pt - record->p).norm();
        /// Check if splat radius is within maxRadius of the cached disk.
        /// This is maximum reconstruction radius.
        if ( //(isFinalGatheringPass && splatR < coeff->radiusForCoeffs) ||
            (splatR < record->rd * HarmonicCaching<T, DIM>::kRho))
        {
            /// Reconstruct boundary part.
            auto [Li, wi] = HarmonicCaching<T, DIM>::reconstructBoundaryContribution(this->pt, pde.absorptionCoeff, *record);
            this->L += Li * wi;

            /// Weight only count once!
            this->weight += wi;

            if (!this->m_bPDEIgnoreSourceTerm)
            {
                auto [Li, wi] = HarmonicCaching<T, DIM>::reconstructSourceContribution(
                    this->pt,
                    *record,
                    pde,
                    this->m_nWalksForSourceTermReconstruction,
                    this->sampler);

                /// wi is essentially the same as boundary part.
                this->L += Li * wi;
            }
        }

        /// Always return true! False will terminate searching process.
        return true;
    }

    /// \brief
    ///    Return if successfully found Green's ball such that total weight > minWeight.
    /// If minWeight == 0, at least 1 valid Green's ball found.
    bool Successful() const noexcept
    {
        /// not equal.
        return this->weight > this->minWeight;
    }

    T getSolution() const noexcept
    {
        return this->L / this->weight;
    }

    /// Evaluation point.
    Vector<DIM> pt;

    /// Config for lookup.
    const PDE<T, DIM>& pde;

    float   minWeight{0};
    bool    m_bPDEIgnoreSourceTerm{false};
    int32_t m_nWalksForSourceTermReconstruction{0};
    pcg32&  sampler;

    /// Lookup results.
    T     L{0};
    float weight{0};
};

template <typename T, size_t DIM>
class HarmonicCaching
{
public:
    HarmonicCaching(const WalkOnStars<T, DIM>& wos,
                    Vector<DIM>                pMin,
                    Vector<DIM>                pMax,
                    float                      minWeightForRecordLookup,
                    int32_t                    nWalksNearBoundary,
                    int32_t                    M,
                    float                      lambda) :
        m_walkOnStars{wos},
        m_MinimumWeightForRecordLookup{minWeightForRecordLookup},
        m_nWalksNearBoundary{nWalksNearBoundary},
        m_nWalksForSourceTermReconstruction{M},
        m_lambda{lambda}
    {
        fcpw::BoundingBox<DIM> domainBBox{};
        domainBBox.expandToInclude(pMin);
        domainBBox.expandToInclude(pMax);

        /// Set depth to 16.
        this->m_pOctree = std::make_unique<Octree<HarmonicCacheRecord<T, DIM>*, DIM>>(domainBBox, 16);
    }

    // clang-format off
    HarmonicCaching(HarmonicCaching&&) = delete;
    HarmonicCaching& operator= (HarmonicCaching&&) = delete;
    HarmonicCaching(const HarmonicCaching&) = delete;
    HarmonicCaching& operator= (const HarmonicCaching&) = delete;
    // clang-format on

    ~HarmonicCaching()
    {
        /// Release octree data node.
        for (const auto& coeffPtr : this->m_CachedRecords)
        {
            delete coeffPtr;
        }
    }

    /// \brief
    ///    Standard harmonic caching algorithm with refinement pass on.
    void solve(const PDE<T, DIM>&                     pde,
               const WalkSettings&                    walkSettings,
               std::vector<SamplePoint<T, DIM>>&      samplePts,
               std::vector<SampleStatistics<T, DIM>>& statistics,
               bool                                   runSingleThreaded = false,
               std::function<void(int, int)>          reportProgress    = {});

    friend class HCReconstructionProcess<T, DIM>;

protected:
    /// \brief
    ///    Populate irradiance cache and return solution at samplePt.
    T lazyCacheUpdate(const PDE<T, DIM>&   pde,
                      const WalkSettings&  walkSettings,
                      SamplePoint<T, DIM>& samplePt,
                      bool                 updateSolution);

    /// \brief
    ///    Compute coefficients at a single point using Booth's expansion.
    ///
    /// \param nWalks indicates number of Monte Carlo samples for coefficients computation.
    /// \return Coefficient and first source contribution (not include boundary term, call this yourself).
    [[nodiscard]] std::pair<HarmonicCacheRecord<T, DIM>*, T>
    computeCoefficientsAt(const PDE<T, DIM>&   pde,
                          const WalkSettings&  walkSettings,
                          SamplePoint<T, DIM>& samplePt) const;

    /// \brief
    ///    Reconstruct solution (boundary part only) at x using harmonic series expansion.
    /// Return unweighted solution and weight.
    static std::pair<T, float>
    reconstructBoundaryContribution(Vector<DIM>&                       x,
                                    float                              screening,
                                    const HarmonicCacheRecord<T, DIM>& cacheRecord);

    /// \brief
    ///    Reconstruct solution (source part only) at x using off-centered Green's function.
    /// Return unweighted solution and weight.
    static std::pair<T, float>
    reconstructSourceContribution(Vector<DIM>&                       x,
                                  const HarmonicCacheRecord<T, DIM>& cacheRecord,
                                  const PDE<T, DIM>&                 pde,
                                  int32_t                            M,
                                  pcg32&                             sampler);

    /// \brief
    ///    ratio is splat ratio. ratio == 1 means splat to largest disk and we
    /// want ratio == 1, weight == 0.
    constexpr static inline float weightingKernel(float ratio, float clampRatio = 0.9)
    {
        if (ratio >= clampRatio)
            return 0;
        /// More aggressive smooth falloff to 0 at clampRatio instead of clamp to 0.
        float d = 1 - ratio / clampRatio;
        return 3 * d * d - 2 * d * d * d;
    };

private:
    /// Underlying estimator.
    const WalkOnStars<T, DIM>& m_walkOnStars;

    /// Fourier orders.
    /// numFixedFourierOrders == 1 indicates a0, a1. 2 terms in total.
    int L{10};

    std::vector<HarmonicCacheRecord<T, DIM>*> m_CachedRecords;

    std::shared_mutex m_CachedRecordsMutex;

    /// Octree.
    std::unique_ptr<Octree<HarmonicCacheRecord<T, DIM>*, DIM>> m_pOctree;

    /// Hyperparameters.
    float   m_MinimumWeightForRecordLookup{1};       /// w_min.
    int32_t m_nWalksForSourceTermReconstruction{64}; /// M.
    float   m_lambda{0.5};                           /// lambda.

    static constexpr float kHarmonicSeriesEpsilon{1e-6f}; /// avoid radial function divided by 0.
    static constexpr float kRho{0.9};                     /// Maximum reconstruction radius ratio.

    /// nWalks near Dirichlet/Neumann/Robin boundaries.
    int32_t m_nWalksNearBoundary{128};
};

template <typename T, size_t DIM>
void HarmonicCaching<T, DIM>::solve(const PDE<T, DIM>&                     pde,
                                    const WalkSettings&                    walkSettings,
                                    std::vector<SamplePoint<T, DIM>>&      samplePts,
                                    std::vector<SampleStatistics<T, DIM>>& statistics,
                                    bool                                   runSingleThreaded,
                                    std::function<void(int, int)>          reportProgress)
{
    size_t nPoints = samplePts.size();
    if (nPoints != statistics.size())
    {
        statistics.clear();
        statistics.resize(nPoints);
    }

    std::random_device rd;
    std::mt19937       g(rd());

    std::vector<size_t> tileIndexMapping(nPoints, 0);
    for (int i = 0; i < tileIndexMapping.size(); ++i)
    {
        tileIndexMapping[i] = i;
    }
    std::shuffle(tileIndexMapping.begin(), tileIndexMapping.end(), g);

    {
        /// Population pass.
        auto run = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i)
            {
                size_t index = tileIndexMapping[i];
                this->lazyCacheUpdate(pde, walkSettings, samplePts[index], false);
            }

            if (reportProgress)
            {
                int tbb_thread_id = tbb::this_task_arena::current_thread_index();
                reportProgress(range.end() - range.begin(), tbb_thread_id);
            }
        };

        tbb::blocked_range<size_t> range(0, nPoints);
        tbb::parallel_for(range, run);
    }

    {
        /// Reconstruction Pass.
        auto run = [&](const tbb::blocked_range<size_t>& range) {
            for (size_t i = range.begin(); i < range.end(); ++i)
            {
                size_t index = tileIndexMapping[i];
                T      u     = this->lazyCacheUpdate(pde, walkSettings, samplePts[index], true);

                statistics[index].addSolutionEstimate(u);
            }
        };

        tbb::blocked_range<size_t> range(0, nPoints);
        tbb::parallel_for(range, run);
    }
}

template <typename T, size_t DIM>
inline T HarmonicCaching<T, DIM>::lazyCacheUpdate(const PDE<T, DIM>&   pde,
                                                  const WalkSettings&  walkSettings,
                                                  SamplePoint<T, DIM>& samplePt,
                                                  bool                 finalReconstructionPass)
{
    /// This is PopulateIrradianceCache().
    /// First check if we are near Dirichlet/Neumann boundary, if so,
    /// don't populate any cache record, simply run WoSt and return.
    /// For simplicity,
    /// When rendering pass == true:
    ///      If on absorbing boundary, near absorbing boundary, near reflecting boundary:
    ///           Run WoSt -> Return Color3.
    ///      Else:
    ///           Search for cache/Populate cache.
    /// When rendering pass == false:
    ///      If on absorbing boundary, near absorbing boundary, near reflecting boundary:
    ///           Return Color3{0}
    ///      Else:
    ///           Search for cache/Populate cache.
    if (samplePt.type == SampleType::OnAbsorbingBoundary ||
        samplePt.type == SampleType::OnReflectingBoundary ||
        samplePt.distToReflectingBoundary <= walkSettings.epsilonShellForReflectingBoundary ||
        samplePt.distToAbsorbingBoundary <= walkSettings.epsilonShellForAbsorbingBoundary)
    {
        if (finalReconstructionPass)
        {
            /// Need to copy to a new sample point to avoid stats add up.
            SamplePoint<T, DIM>      newSamplePt = samplePt;
            SampleStatistics<T, DIM> sampleStats;
            this->m_walkOnStars.estimateSolution(pde, walkSettings, this->m_nWalksNearBoundary, newSamplePt, sampleStats);

            T solution = sampleStats.getEstimatedSolution();
            return solution;
        }
        else
        {
            /// Should never be used!
            return T{1000000};
        }
    }


    float boundaryDist = std::min(samplePt.distToAbsorbingBoundary,
                                  samplePt.distToReflectingBoundary);
    float r            = RADIUS_SHRINK_PERCENTAGE * boundaryDist;

    bool  needPopulateNewCacheRecord{false};
    T     totalSolution{0};
    float totalWeights{0};
    {
        /// Acquire a reader lock.
        std::shared_lock<std::shared_mutex> lock(this->m_CachedRecordsMutex);

        /// If this is a rendering pass, then ignoreSourceContribution inherits globally.
        /// If this is a cache population pass, then we always don't do source reconstruction since
        /// we don't care about solution yet.
        bool ignoreSourceContribution = finalReconstructionPass ? walkSettings.ignoreSourceContribution : true;

        HCReconstructionProcess<T, DIM> proc{samplePt.pt,
                                             pde,
                                             this->m_MinimumWeightForRecordLookup,
                                             ignoreSourceContribution,
                                             this->m_nWalksForSourceTermReconstruction,
                                             samplePt.rng};
        this->m_pOctree->Lookup(samplePt.pt, proc);
        if (proc.Successful())
        {
            needPopulateNewCacheRecord = false;
            return proc.getSolution();
        }
        else
        {
            /// Account for lookup as well.
            totalSolution = proc.L;
            totalWeights  = proc.weight;

            needPopulateNewCacheRecord = true;
        }
    }

    if (needPopulateNewCacheRecord)
    {
        /// Estimate new coefficients.
        const auto& [record, firstSourceContribution] = this->computeCoefficientsAt(pde, walkSettings, samplePt);

        /// Only 1 thread can update the quadtree and cache.
        {
            /// Acquire a writer lock.
            std::unique_lock<std::shared_mutex> lock(this->m_CachedRecordsMutex);
            /// Update quadtree.
            if constexpr (DIM == 2)
            {
                /// WARNING: We record the largest ball radius.
                fcpw::BoundingBox<DIM> bbox{Vector2(samplePt.pt.x() - r, samplePt.pt.y() - r)};
                bbox.expandToInclude(Vector2(samplePt.pt.x() + r, samplePt.pt.y() + r));

                this->m_pOctree->Add(record, bbox);
            }
            else
            {
                fcpw::BoundingBox<DIM> bbox{Vector3(samplePt.pt.x() - r, samplePt.pt.y() - r, samplePt.pt.z() - r)};
                bbox.expandToInclude(Vector3(samplePt.pt.x() + r, samplePt.pt.y() + r, samplePt.pt.z() + r));

                this->m_pOctree->Add(record, bbox);
            }
            /// Update cached coefficients ptrs.
            this->m_CachedRecords.push_back(record);
        }

        //cacheStorageBytes += coefficients.GetMemoryUsageBytes();

        /// a0 is the boundary part.
        T boundarySolution = record->coeffs.returnSolutionAtOrigin();
        T sourceSolution   = firstSourceContribution;

        /// This totalWeights+=1 comes from we populate a new cache record, and also
        /// eval solution at the cache record center. Then for a smoothstep function
        /// the weight at 0 is 1. So we add the weight 1 to total weights.
        totalSolution += boundarySolution + sourceSolution; /// wi == 1.
        totalWeights += 1;
        /// Account for source term.
        return totalSolution / totalWeights;
    }
}

template <typename T, size_t DIM>
inline std::pair<HarmonicCacheRecord<T, DIM>*, T>
HarmonicCaching<T, DIM>::computeCoefficientsAt(const PDE<T, DIM>&   pde,
                                               const WalkSettings&  walkSettings,
                                               SamplePoint<T, DIM>& samplePt) const
{
    assert(samplePt.type == SampleType::InDomain &&
           samplePt.distToAbsorbingBoundary > walkSettings.epsilonShellForAbsorbingBoundary &&
           samplePt.distToReflectingBoundary > walkSettings.epsilonShellForReflectingBoundary);

    // use the distance to the boundary as the first sphere radius for all walks;
    // shrink the radius slightly for numerical robustness---using a conservative
    // distance does not impact correctness
    float boundaryDist         = std::min(samplePt.distToAbsorbingBoundary,
                                          samplePt.distToReflectingBoundary);
    samplePt.firstSphereRadius = RADIUS_SHRINK_PERCENTAGE * boundaryDist;

    auto NumSamplesOfR = [this](float r) -> int {
        /// assume rmax = 1.0
        if constexpr (DIM == 2)
            return std::max<int>(32.0, this->m_lambda * 160000 * r);
        else
        {
            return std::clamp<int>(this->m_lambda * 160000 * r * r, 512.0f, 10000.0f);
        }
    };

    size_t nWalks = NumSamplesOfR(samplePt.firstSphereRadius);

    /// Generate nWalks stratified samples in DIM-1 dimension. For 2D, we need 1D samples.
    std::vector<float> stratifiedSamples; /// Flatten storage of stratified samples.
    generateStratifiedSamples<DIM - 1>(stratifiedSamples, nWalks, samplePt.rng);
    assert(stratifiedSamples.size() == nWalks * (DIM - 1));

    /// Generate nWalks stratified samples for source term.
    std::vector<float> stratifiedSamplesSource; /// Flatten storage of stratified samples.
    if (!walkSettings.ignoreSourceContribution)
    {
        generateStratifiedSamples<DIM - 1>(stratifiedSamplesSource, nWalks, samplePt.rng);
        assert(stratifiedSamplesSource.size() == nWalks * (DIM - 1));
    }

    /// Allocate a new HarmonicCacheRecord.
    HarmonicCacheRecord<T, DIM>*  record = new HarmonicCacheRecord<T, DIM>{this->L};
    HarmonicCoefficients<T, DIM>& coefficients{record->coeffs};
    float                         estimateFirstSourceContribution{0};
    int32_t                       nValidWalks{0};

    /// Note: for ALLOCA, don't put this into the loop since the memory will never release until
    /// at the end of this function.
    float* Ylm{nullptr};
    if constexpr (DIM == 3)
    {
        Ylm = sh::StackAlloc<float>(sh::SHTerms(coefficients.GetFourierOrders()));
    }

    std::queue<WalkState<T, DIM>> stateQueue;
    for (int w = 0; w < nWalks; w++)
    {
        /// No need to reseed here.
        /// Reset walk state and reset current query pt.
        WalkState<T, DIM> state(samplePt.pt, Vector<DIM>::Zero(), Vector<DIM>::Zero(),
                                0.0f, 1.0f, 0, false);

        std::unique_ptr<GreensFnBall<DIM>> greensFn;
        // initialize the greens function
        if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0)
        {
            greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);
        }
        else
        {
            greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
        }

        /// Initialize Green's ball.
        greensFn->updateBall(state.currentPt, samplePt.firstSphereRadius);

        /// We include source term contribution and use it only to compute the solution
        /// but not include in the coefficient!
        // compute the source contribution inside the ball
        T     firstSourceContribution{0};
        float sourcePdf{0};
        if (!walkSettings.ignoreSourceContribution)
        {
            /// Importance sampling Green's function.
            /// This is used for sampling the direction.
            float* u = &stratifiedSamplesSource[(DIM - 1) * w];

            /// First sample the direction.
            Vector<DIM> sourceDirection = SphereSampler<DIM>::sampleUnitSphereUniform(u);
            /// Then importance sampling the radius.
            /// pdf in solid angle.
            float       _r{0};
            Vector<DIM> sourcePt = greensFn->sampleVolume(sourceDirection, samplePt.rng, _r, sourcePdf);

            float greensFnNorm = greensFn->norm();
            /// 1-sample estimator for source term. importance sampling the green's function
            /// gives G*f/(G/normG)=normG*f.
            T sourceContribution = greensFnNorm * pde.source(sourcePt);

            /// Don't include first source contribution to coefficients!
            //state.totalSourceContribution += state.throughput * sourceContribution;

            firstSourceContribution = sourceContribution;
            estimateFirstSourceContribution += firstSourceContribution;
        }

        /// Sample a point uniformly on the sphere; update the current position
        /// of the walk, its throughput.
        float* u = &stratifiedSamples[(DIM - 1) * w];
        //unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
        //samplePt.sampler.seed(seed);
        //float urand = StratifiedSample1D(nWalks, w, samplePt.sampler, true);
        ////float  urand = samplePt.sampler.nextFloat();
        //float* u     = &urand;
        float       theta{0};
        Vector<DIM> boundaryDirection;
        if constexpr (DIM == 2)
        {
            boundaryDirection = SphereSampler<DIM>::sampleUnitSphereUniform(u, theta);
        }
        else
        {
            boundaryDirection = SphereSampler<DIM>::sampleUnitSphereUniform(u);
        }
        /// Solid angle pdf, that's why we use r=1 as input.
        float boundaryPdf = SphereSampler<DIM>::pdfSampleSphereUniform(1.0f);

        /// Update Green's ball.
        Vector<DIM> ySurf = greensFn->c + greensFn->R * boundaryDirection;

        /// Update walk state.
        state.prevDistance  = greensFn->R;
        state.prevDirection = (ySurf - state.currentPt) / greensFn->R;
        state.currentPt     = ySurf;

        /// Note that we should *NOT* update throughput of the walk since we want to know the solution at
        /// greensFn->ySurf instead of the ball center.
        //state.throughput *= greensFn->poissonKernel() / boundaryPdf;

        // add the state to the queue
        stateQueue.emplace(state);
        int  splitsPerformed   = -1;
        T    totalContribution = T(0.0f);
        bool success           = false;

        while (!stateQueue.empty())
        {
            state = stateQueue.front();
            stateQueue.pop();
            splitsPerformed++;

            // initialize the greens function
            if (splitsPerformed > 0)
            {
                if (pde.absorptionCoeff > 0.0f && walkSettings.stepsBeforeApplyingTikhonov == 0)
                {
                    greensFn = std::make_unique<YukawaGreensFnBall<DIM>>(pde.absorptionCoeff);
                }
                else
                {
                    greensFn = std::make_unique<HarmonicGreensFnBall<DIM>>();
                }
            }

            // compute the distance to the absorbing boundary
            float distToAbsorbingBoundary = this->m_walkOnStars.getGeometricQueries().computeDistToAbsorbingBoundary(state.currentPt, false);

            // perform the walk with the dequeued state
            WalkCompletionCode code = this->m_walkOnStars.walk(pde, walkSettings, distToAbsorbingBoundary, 0.0f, false, greensFn, samplePt.rng, state, stateQueue);

            if (code == WalkCompletionCode::ReachedAbsorbingBoundary ||
                code == WalkCompletionCode::TerminatedWithRussianRoulette ||
                code == WalkCompletionCode::ExceededMaxWalkLength)
            {
                // compute the walk contribution
                T terminalContribution = this->m_walkOnStars.getTerminalContribution(code, pde, walkSettings, state);
                totalContribution += state.throughput * terminalContribution +
                    state.totalReflectingBoundaryContribution +
                    state.totalSourceContribution;

                // record the walk length
                //statistics.addWalkLength(state.walkLength);
                success = true;
            }
        }

        if (success)
        {
            if constexpr (DIM == 2)
            {
                for (int i = 0; i < coefficients.GetFourierOrders(); ++i)
                {
                    float radialPart{0};
                    assert(samplePt.firstSphereRadius == RADIUS_SHRINK_PERCENTAGE * boundaryDist);

                    // Avoid division by very small Bessel values
                    //if (besselVal < 1e-10) besselVal = 1e-10;

                    T anScalar = totalContribution * std::cos((i + 1) * theta) * 2;
                    T bnScalar = totalContribution * std::sin((i + 1) * theta) * 2;
                    coefficients.an[i] += T{anScalar};
                    coefficients.bn[i] += T{bnScalar};
                }
                T a0Scalar = totalContribution /* / nWalks*/;
                coefficients.a0 += T{a0Scalar};

                /// Record a valid walk.
                ++nValidWalks;
            }
            else
            {
                /// Compute SH coefficients for direction \omega.
                sh::SHEvaluate(boundaryDirection, coefficients.GetFourierOrders(), Ylm);

                /// Compute WoS SH coefficients.
                for (int l = 0; l <= coefficients.GetFourierOrders(); ++l)
                {
                    /// TODO: Use new form of series expansion in the SIGGRAPH Asia paper (which still requires the old form for a0).
                    /// Screening == 0: 1/R^n
                    /// Screening >0: 1/i_l(R*sqrt(alpha))
                    float invRadialPart = pde.absorptionCoeff > 0 ?
                        1.0 / bessel::ModifiedSphericalBesselFirst(l, samplePt.firstSphereRadius * std::sqrt(pde.absorptionCoeff)) :
                        1.0 / std::pow(samplePt.firstSphereRadius, l);

                    /// Clamp infinity large coefficients to zero. Skip this
                    /// order reconstruction.
                    if (std::isinf(invRadialPart))
                    {
                        std::cerr << "inf 1/r..." << std::endl;
                        break;
                    }
                    for (int m = -l; m <= l; ++m)
                    {
                        T almScalar = totalContribution * Ylm[sh::SHIndex(l, m)] *
                            invRadialPart /
                            (boundaryPdf); /// Divided by nValid walk finally.
                        coefficients.alm[sh::SHIndex(l, m)] += T{almScalar};

                        if (std::isnan(coefficients.alm[sh::SHIndex(l, m)]))
                        {
                            std::cerr << "nan coefficients" << std::endl;
                        }
                    }
                }

                /// Record a valid walk.
                ++nValidWalks;
            }
        }
    }

    if constexpr (DIM == 2)
    {
        if (pde.absorptionCoeff > 0)
        {
            /// Note that 2D Screened Poisson Equation requires u(0)=a0 in our convention. So we multiply by Poisson kernel.
            coefficients.a0 /= std::cyl_bessel_if(0, samplePt.firstSphereRadius * std::sqrt(pde.absorptionCoeff)); /// This is Poisson kernel.
        }
    }
    else
    {
        if (pde.absorptionCoeff > 0)
        {
            /// 3D Screened Poisson Equation requires u(0)=1/(2*sqrt(pi))*a0 in our convention.
            /// No-op since we perform this computation above.
        }
    }

    coefficients.performMCDivisionForNumValidSamples(nValidWalks);
    /// MC Division by valid samples for source term; assuming all source term walks are valid.
    estimateFirstSourceContribution /= static_cast<float>(nWalks);
    record->rd = samplePt.firstSphereRadius;
    record->p  = samplePt.pt;

    return std::make_pair(record, estimateFirstSourceContribution);
}

template <typename T, size_t DIM>
std::pair<T, float>
HarmonicCaching<T, DIM>::reconstructBoundaryContribution(Vector<DIM>&                       x,
                                                         float                              screening,
                                                         const HarmonicCacheRecord<T, DIM>& cacheRecord)
{
    /// x is the evaluation point.
    /// Compute splatting distance.
    float splatR = (x - cacheRecord.p).norm();

    if (splatR < HarmonicCaching<T, DIM>::kHarmonicSeriesEpsilon)
    {
        /// weighting at 0 == 1.
        static_assert(HarmonicCaching<T, DIM>::weightingKernel(0.0f) == 1);
        return std::make_pair(cacheRecord.coeffs.returnSolutionAtOrigin(), HarmonicCaching<T, DIM>::weightingKernel(0.0f));
    }

    /// Reconstructed u using series expansion.
    T reconstruction{0};
    if constexpr (DIM == 2)
    {
        Vector<DIM> dir   = x - cacheRecord.p;
        float       theta = std::atan2(dir.y(), dir.x());

        reconstruction = cacheRecord.coeffs.a0;
        if (screening > 0)
        {
            reconstruction *= std::cyl_bessel_if(0, splatR * std::sqrt(screening)); /// Note that for zeroth term we put I_0(R*sqrt(c)) into coefficient such that u0=a0 for 2D.
        }

        for (int i = 0; i < cacheRecord.coeffs.GetFourierOrders(); ++i)
        {
            const double& R = cacheRecord.rd;

            double radialPart{0};
            if (screening > 0)
            {
                radialPart = std::cyl_bessel_if((i + 1), splatR * std::sqrt(screening)) / std::cyl_bessel_if((i + 1), R * std::sqrt(screening));
            }
            else
            {
                radialPart = std::pow(splatR / R, i + 1);
            }

            reconstruction += radialPart *
                (cacheRecord.coeffs.an[i] * std::cos((i + 1) * theta) + cacheRecord.coeffs.bn[i] * std::sin((i + 1) * theta));
        }
    }
    else
    {
        static_assert(DIM == 3, "Not supported dimensionality for series expansion.");

        Vector<DIM> dir = (x - cacheRecord.p).normalized();

        /// Compute SH coefficients for direction \omega.
        float* ylm = sh::StackAlloc<float>(sh::SHTerms(cacheRecord.coeffs.GetFourierOrders()));

        sh::SHEvaluate(dir, cacheRecord.coeffs.GetFourierOrders(), ylm);

        /// Compute WoS SH coefficients.
        for (int l = 0; l <= cacheRecord.coeffs.GetFourierOrders(); ++l)
        {
            /// Screening == 0: r^n
            /// Screening >0: i_l(r*sqrt(alpha))
            float radialPart = screening > 0 ? bessel::ModifiedSphericalBesselFirst(l, splatR * std::sqrt(screening)) : std::pow(splatR, l);

            /// TODO: Check if (radialPart < 1e-10).

            for (int m = -l; m <= l; ++m)
            {
                /// SH reconstruction.
                reconstruction += cacheRecord.coeffs.alm[sh::SHIndex(l, m)] * ylm[sh::SHIndex(l, m)] *
                    radialPart;
            }
        }
    }

    /// To make things clear, we use the splatRadius/largestRadius as an input to the weighting function,
    /// in the smoothStep function, it falls off to 0, e.g. when splatRadius >= 0.9 largestRadius.
    float weight = HarmonicCaching<T, DIM>::weightingKernel(splatR / cacheRecord.rd);
    return std::make_pair(reconstruction, weight);
}

template <typename T, size_t DIM>
std::pair<T, float>
HarmonicCaching<T, DIM>::reconstructSourceContribution(Vector<DIM>&                       x,
                                                       const HarmonicCacheRecord<T, DIM>& cacheRecord,
                                                       const PDE<T, DIM>&                 pde,
                                                       int32_t                            M,
                                                       pcg32&                             sampler)
{
    /// Note: do not use greensFn.sampleVolume(), which is the centered green's function
    /// sampling. Here we need importance sampling off-centered Green's function, which
    /// is tricky.
    T estimateFirstSourceContribution{0};

    /// Generate nWalks stratified samples for source term.
    /// Note that we need 2D samples for a disk.
    std::vector<float> stratifiedSamplesSource; /// Flatten storage of stratified samples.
    generateStratifiedSamples<DIM>(stratifiedSamplesSource, M, sampler);
    assert(stratifiedSamplesSource.size() == M * DIM);

    for (int w = 0; w < M; ++w)
    {
        /// Initialize the green's ball.
        HarmonicGreensFnBall<DIM> greensFn{};
        /// Update ball radius to be *radiusForCoeffs*.
        greensFn.updateBall(cacheRecord.p, cacheRecord.rd);

        float  firstSourceContribution{0};
        float* u = &stratifiedSamplesSource[DIM * w];

        /// Simply uniform sample a disk.
        Vector<DIM> sourcePt = SphereSampler<DIM>::sampleUnitBallUniform(u);
        /// Scale to the disk.
        sourcePt *= cacheRecord.rd;
        /// Move to the green's ball.
        sourcePt = cacheRecord.p + sourcePt;

        /// This is area measure. Integration in area measure.
        float pdf = SphereSampler<DIM>::pdfSampleBallUniform(cacheRecord.rd);
        /// Green function is symmetric.
        float green = greensFn.evaluate(sourcePt, x);

        T sourceContribution    = green * pde.source(sourcePt) / pdf;
        firstSourceContribution = sourceContribution;
        estimateFirstSourceContribution += firstSourceContribution;
    }

    /// MC Division by valid samples for source term; assuming all source term walks are valid.
    estimateFirstSourceContribution /= static_cast<float>(M);
    /// Compute extrapolation radius.
    float splatR = (x - cacheRecord.p).norm();

    /// To make things clear, we use the splatRadius/largestRadius as an input to the weighting function,
    /// in the smoothStep function, it falls off to 0, e.g. when splatRadius >= 0.9 largestRadius.
    float weight = HarmonicCaching<T, DIM>::weightingKernel(splatR / cacheRecord.rd);
    return std::make_pair(estimateFirstSourceContribution, weight);
}
} // namespace hc
} // namespace zombie
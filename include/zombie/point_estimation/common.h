// This file provides various convenience structs and enums to specify settings
// and extract outputs from the algorithms implemented in Zombie. Specifically, in
// addition to the PDE and GeometricQueries interfaces, the user should populate, via
// their constructors, the WalkSettings, SamplePoint, and SampleEstimationData structs
// for the walk-on-spheres and walk-on-stars algorithms. For each SamplePoint where
// the PDE is to be solved, the SampleStatistics struct can be used to query the
// estimated solution, gradient, and other statistics.

#pragma once

#include <zombie/core/pde.h>
#include <zombie/core/geometric_queries.h>
#include <zombie/core/distributions.h>

#define RADIUS_SHRINK_PERCENTAGE 0.99f

namespace zombie {

struct WalkSettings {
    // constructors
    WalkSettings(float epsilonShellForAbsorbingBoundary_,
                 float epsilonShellForReflectingBoundary_,
                 int maxWalkLength_, bool solveDoubleSided_):
                 epsilonShellForAbsorbingBoundary(epsilonShellForAbsorbingBoundary_),
                 epsilonShellForReflectingBoundary(epsilonShellForReflectingBoundary_),
                 silhouettePrecision(1e-3f),
                 russianRouletteThreshold(0.0f),
                 maxWalkLength(maxWalkLength_),
                 stepsBeforeApplyingTikhonov(maxWalkLength_),
                 stepsBeforeUsingMaximalSpheres(maxWalkLength_),
                 solveDoubleSided(solveDoubleSided_),
                 useGradientControlVariates(true),
                 useGradientAntitheticVariates(true),
                 useCosineSamplingForDerivatives(false),
                 ignoreAbsorbingBoundaryContribution(false),
                 ignoreReflectingBoundaryContribution(false),
                 ignoreSourceContribution(false),
                 printLogs(false) {}
    WalkSettings(float epsilonShellForAbsorbingBoundary_,
                 float epsilonShellForReflectingBoundary_,
                 float silhouettePrecision_, float russianRouletteThreshold_,
                 int maxWalkLength_, int stepsBeforeApplyingTikhonov_,
                 int stepsBeforeUsingMaximalSpheres_, bool solveDoubleSided_,
                 bool useGradientControlVariates_, bool useGradientAntitheticVariates_,
                 bool useCosineSamplingForDerivatives_, bool ignoreAbsorbingBoundaryContribution_,
                 bool ignoreReflectingBoundaryContribution_, bool ignoreSourceContribution_,
                 bool printLogs_):
                 epsilonShellForAbsorbingBoundary(epsilonShellForAbsorbingBoundary_),
                 epsilonShellForReflectingBoundary(epsilonShellForReflectingBoundary_),
                 silhouettePrecision(silhouettePrecision_),
                 russianRouletteThreshold(russianRouletteThreshold_),
                 maxWalkLength(maxWalkLength_),
                 stepsBeforeApplyingTikhonov(stepsBeforeApplyingTikhonov_),
                 stepsBeforeUsingMaximalSpheres(stepsBeforeUsingMaximalSpheres_),
                 solveDoubleSided(solveDoubleSided_),
                 useGradientControlVariates(useGradientControlVariates_),
                 useGradientAntitheticVariates(useGradientAntitheticVariates_),
                 useCosineSamplingForDerivatives(useCosineSamplingForDerivatives_),
                 ignoreAbsorbingBoundaryContribution(ignoreAbsorbingBoundaryContribution_),
                 ignoreReflectingBoundaryContribution(ignoreReflectingBoundaryContribution_),
                 ignoreSourceContribution(ignoreSourceContribution_),
                 printLogs(printLogs_) {}

    // members
    float epsilonShellForAbsorbingBoundary;
    float epsilonShellForReflectingBoundary;
    float silhouettePrecision;
    float russianRouletteThreshold;
    int maxWalkLength;
    int stepsBeforeApplyingTikhonov;
    int stepsBeforeUsingMaximalSpheres;
    bool solveDoubleSided; // NOTE: this flag should be set to true if domain is open
    bool useGradientControlVariates;
    bool useGradientAntitheticVariates;
    bool useCosineSamplingForDerivatives;
    bool ignoreAbsorbingBoundaryContribution;
    bool ignoreReflectingBoundaryContribution;
    bool ignoreSourceContribution;
    bool printLogs;
};

template <typename T, size_t DIM>
struct WalkState {
    // constructor
    WalkState(const Vector<DIM>& currentPt_, const Vector<DIM>& currentNormal_,
              const Vector<DIM>& prevDirection_, float prevDistance_, float throughput_,
              bool onReflectingBoundary_, int walkLength_):
              greensFn(nullptr),
              currentPt(currentPt_),
              currentNormal(currentNormal_),
              prevDirection(prevDirection_),
              prevDistance(prevDistance_),
              throughput(throughput_),
              onReflectingBoundary(onReflectingBoundary_),
              totalReflectingBoundaryContribution(0.0f),
              totalSourceContribution(0.0f),
              walkLength(walkLength_) {}

    // members
    std::unique_ptr<GreensFnBall<DIM>> greensFn;
    Vector<DIM> currentPt;
    Vector<DIM> currentNormal;
    Vector<DIM> prevDirection;
    float prevDistance;
    float throughput;
    bool onReflectingBoundary;
    T totalReflectingBoundaryContribution;
    T totalSourceContribution;
    int walkLength;
};

enum class WalkCompletionCode {
    ReachedAbsorbingBoundary,
    TerminatedWithRussianRoulette,
    ExceededMaxWalkLength,
    EscapedDomain
};

// NOTE: For data with multiple channels (e.g., 2D or 3D positions, rgb etc.), use
// Eigen::Array (in place of Eigen::VectorXf) as it supports component wise operations
template <typename T, size_t DIM>
class SampleStatistics {
public:
    // constructor
    SampleStatistics() {
        reset();
    }

    // resets statistics
    void reset() {
        solutionMean = T(0.0f);
        solutionM2 = T(0.0f);
        for (int i = 0; i < DIM; i++) {
            gradientMean[i] = T(0.0f);
            gradientM2[i] = T(0.0f);
        }
        totalFirstSourceContribution = T(0.0f);
        totalDerivativeContribution = T(0.0f);
        nSolutionEstimates = 0;
        nGradientEstimates = 0;
        totalWalkLength = 0;
    }

    // adds solution estimate to running sum
    void addSolutionEstimate(const T& estimate) {
        nSolutionEstimates += 1;
        update(estimate, solutionMean, solutionM2, nSolutionEstimates);
    }

    // adds gradient estimate to running sum
    void addGradientEstimate(const T *boundaryEstimate, const T *sourceEstimate) {
        nGradientEstimates += 1;
        for (int i = 0; i < DIM; i++) {
            update(boundaryEstimate[i] + sourceEstimate[i], gradientMean[i],
                   gradientM2[i], nGradientEstimates);
        }
    }

    // adds gradient estimate to running sum
    void addGradientEstimate(const T *estimate) {
        nGradientEstimates += 1;
        for (int i = 0; i < DIM; i++) {
            update(estimate[i], gradientMean[i], gradientM2[i], nGradientEstimates);
        }
    }

    // adds source contribution for the first step to running sum
    void addFirstSourceContribution(const T& contribution) {
        totalFirstSourceContribution += contribution;
    }

    // adds derivative contribution to running sum
    void addDerivativeContribution(const T& contribution) {
        totalDerivativeContribution += contribution;
    }

    // adds walk length to running sum
    void addWalkLength(int length) {
        totalWalkLength += length;
    }

    // returns estimated solution
    T getEstimatedSolution() const {
        return solutionMean;
    }

    // returns variance of estimated solution
    T getEstimatedSolutionVariance() const {
        int N = std::max(1, nSolutionEstimates - 1);
        return solutionM2/N;
    }

    // returns estimated gradient
    const T* getEstimatedGradient() const {
        return gradientMean;
    }

    // returns variance of estimated gradient
    std::vector<T> getEstimatedGradientVariance() const {
        int N = std::max(1, nGradientEstimates - 1);
        std::vector<T> variance(DIM);

        for (int i = 0; i < DIM; i++) {
            variance[i] = gradientM2[i]/N;
        }

        return variance;
    }

    // returns mean source contribution for the first step
    T getMeanFirstSourceContribution() const {
        int N = std::max(1, nSolutionEstimates);
        return totalFirstSourceContribution/N;
    }

    // returns estimated derivative
    T getEstimatedDerivative() const {
        int N = std::max(1, nSolutionEstimates);
        return totalDerivativeContribution/N;
    }

    // returns number of solution estimates
    int getSolutionEstimateCount() const {
        return nSolutionEstimates;
    }

    // returns number of gradient estimates
    int getGradientEstimateCount() const {
        return nGradientEstimates;
    }

    // returns mean walk length
    float getMeanWalkLength() const {
        int N = std::max(1, nSolutionEstimates);
        return (float)totalWalkLength/N;
    }

protected:
    // updates statistics
    void update(const T& estimate, T& mean, T& M2, int N) {
        T delta = estimate - mean;
        mean += delta/N;
        T delta2 = estimate - mean;
        M2 += delta*delta2;
    }

    // members
    T solutionMean, solutionM2;
    T gradientMean[DIM], gradientM2[DIM];
    T totalFirstSourceContribution;
    T totalDerivativeContribution;
    int nSolutionEstimates, nGradientEstimates;
    int totalWalkLength;
};

enum class SampleType {
    InDomain, // applies to both interior and exterior sample points for closed domains
    OnAbsorbingBoundary,
    OnReflectingBoundary
};

template <typename T, size_t DIM>
struct SamplePoint {
    // constructor
    SamplePoint(const Vector<DIM>& pt_, const Vector<DIM>& normal_,
                SampleType type_, float pdf_, float distToAbsorbingBoundary_,
                float distToReflectingBoundary_):
                pt(pt_), normal(normal_), type(type_), pdf(pdf_),
                distToAbsorbingBoundary(distToAbsorbingBoundary_),
                distToReflectingBoundary(distToReflectingBoundary_),
                firstSphereRadius(0.0f), estimateBoundaryNormalAligned(false) {
        reset();
    }

    // resets solution data
    void reset() {
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        sampler = pcg32(seed);
        statistics = nullptr;
        solution = T(0.0f);
        normalDerivative = T(0.0f);
        source = T(0.0f);
        robin = T(0.0f);
        robinCoeff = 0.0f;
    }

    // members
    pcg32 sampler;
    Vector<DIM> pt;
    Vector<DIM> normal;
    SampleType type;
    float pdf;
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;
    float firstSphereRadius; // populated by WoSt
    bool estimateBoundaryNormalAligned;
    std::shared_ptr<SampleStatistics<T, DIM>> statistics; // populated by WoSt
    T solution, normalDerivative, source, robin; // not populated by WoSt, but available for downstream use (e.g. BVC)
    float robinCoeff; // not populated by WoSt, but available for downstream use (e.g. BVC)
};

enum class EstimationQuantity {
    Solution,
    SolutionAndGradient,
    None
};

template <size_t DIM>
struct SampleEstimationData {
    // constructors
    SampleEstimationData(): nWalks(0), estimationQuantity(EstimationQuantity::None),
                            directionForDerivative(Vector<DIM>::Zero()) {
        directionForDerivative(0) = 1.0f;
    }
    SampleEstimationData(int nWalks_, EstimationQuantity estimationQuantity_,
                         Vector<DIM> directionForDerivative_=Vector<DIM>::Zero()):
                         nWalks(nWalks_), estimationQuantity(estimationQuantity_),
                         directionForDerivative(directionForDerivative_) {}

    // members
    int nWalks;
    EstimationQuantity estimationQuantity;
    Vector<DIM> directionForDerivative; // needed only for computing direction derivatives
};

} // zombie

// This file provides various convenience structs and enums to specify settings
// and extract outputs from the algorithms implemented in Zombie. Specifically, in
// addition to the PDE and GeometricQueries interfaces, the user should populate, via
// their constructors, the WalkSettings and SamplePoint structs for the walk-on-spheres
// and walk-on-stars algorithms. For each SamplePoint where the PDE is to be solved,
// the SampleStatistics struct can be used to query the estimated solution, gradient,
// and other statistics.

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
                 splittingThreshold(std::numeric_limits<float>::max()),
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
                 float splittingThreshold_, int maxWalkLength_,
                 int stepsBeforeApplyingTikhonov_, int stepsBeforeUsingMaximalSpheres_,
                 bool solveDoubleSided_, bool useGradientControlVariates_,
                 bool useGradientAntitheticVariates_, bool useCosineSamplingForDerivatives_,
                 bool ignoreAbsorbingBoundaryContribution_, bool ignoreReflectingBoundaryContribution_,
                 bool ignoreSourceContribution_, bool printLogs_):
                 epsilonShellForAbsorbingBoundary(epsilonShellForAbsorbingBoundary_),
                 epsilonShellForReflectingBoundary(epsilonShellForReflectingBoundary_),
                 silhouettePrecision(silhouettePrecision_),
                 russianRouletteThreshold(russianRouletteThreshold_),
                 splittingThreshold(splittingThreshold_),
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
    float splittingThreshold;
    int maxWalkLength;
    int stepsBeforeApplyingTikhonov;
    int stepsBeforeUsingMaximalSpheres;
    bool solveDoubleSided; // NOTE: this flag should be set to true if the domain is not watertight
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
    // constructors
    WalkState(): greensFn(nullptr),
                 totalReflectingBoundaryContribution(0.0f),
                 totalSourceContribution(0.0f),
                 currentPt(Vector<DIM>::Zero()),
                 currentNormal(Vector<DIM>::Zero()),
                 prevDirection(Vector<DIM>::Zero()),
                 prevDistance(0.0f), throughput(1.0f),
                 walkLength(0), onReflectingBoundary(false) {}
    WalkState(const Vector<DIM>& currentPt_, const Vector<DIM>& currentNormal_,
              const Vector<DIM>& prevDirection_, float prevDistance_,
              float throughput_, int walkLength_, bool onReflectingBoundary_):
              greensFn(nullptr),
              totalReflectingBoundaryContribution(0.0f),
              totalSourceContribution(0.0f),
              currentPt(currentPt_),
              currentNormal(currentNormal_),
              prevDirection(prevDirection_),
              prevDistance(prevDistance_),
              throughput(throughput_),
              walkLength(walkLength_),
              onReflectingBoundary(onReflectingBoundary_) {}

    // members
    std::shared_ptr<GreensFnBall<DIM>> greensFn;
    T totalReflectingBoundaryContribution;
    T totalSourceContribution;
    Vector<DIM> currentPt;
    Vector<DIM> currentNormal;
    Vector<DIM> prevDirection;
    float prevDistance;
    float throughput;
    int walkLength;
    bool onReflectingBoundary;
};

enum class WalkCompletionCode {
    ReachedAbsorbingBoundary,
    TerminatedWithRussianRoulette,
    ExceededMaxWalkLength,
    EscapedDomain
};

// NOTE: For data with multiple channels (e.g., 2D or 3D positions, rgb etc.),
// use Eigen::Array<float, CHANNELS, 1> (in place of Eigen::Matrix<float, CHANNELS, 1>)
// as it supports component wise operations
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
        totalSplits = 0;
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

    // adds walk length to running sum
	void addSplits(int nSplits) {
		totalSplits += nSplits;
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

    // returns estimated gradient for specified channel
    T getEstimatedGradient(int channel) const {
        return gradientMean[channel];
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

    // returns mean splits performed per walk
	float getMeanSplits() const {
        int N = std::max(1, nSolutionEstimates);
		return (float)totalSplits/N;
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
    int totalWalkLength, totalSplits;
};

enum class SampleType {
    InDomain, // applies to both interior and exterior sample points for watertight domains
    OnAbsorbingBoundary,
    OnReflectingBoundary
};

enum class EstimationQuantity {
    Solution,
    SolutionAndGradient,
    None
};

template <typename T, size_t DIM>
struct SamplePoint {
    // constructor
    SamplePoint(const Vector<DIM>& pt_, const Vector<DIM>& normal_,
                SampleType type_, EstimationQuantity estimationQuantity_,
                float pdf_, float distToAbsorbingBoundary_, float distToReflectingBoundary_):
                pt(pt_), normal(normal_), type(type_),
                estimationQuantity(estimationQuantity_), pdf(pdf_),
                distToAbsorbingBoundary(distToAbsorbingBoundary_),
                distToReflectingBoundary(distToReflectingBoundary_),
                estimateBoundaryNormalAligned(false) {
        directionForDerivative = Vector<DIM>::Zero();
        directionForDerivative(0) = 1.0f;
        reset();
    }

    // resets solution data
    void reset() {
        auto now = std::chrono::high_resolution_clock::now();
        uint64_t seed = std::chrono::duration_cast<std::chrono::nanoseconds>(now.time_since_epoch()).count();
        sampler = pcg32(seed);
        statistics.reset();
        firstSphereRadius = 0.0f;
        robinCoeff = 0.0f;
        solution = T(0.0f);
        normalDerivative = T(0.0f);
        contribution = T(0.0f);
    }

    // members
    pcg32 sampler;
    SampleStatistics<T, DIM> statistics;         // populated by WoSt
    Vector<DIM> pt;
    Vector<DIM> normal;
    Vector<DIM> directionForDerivative;          // needed only for computing directional derivatives
    SampleType type;
    EstimationQuantity estimationQuantity;
    float pdf;
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;
    float firstSphereRadius;                     // populated by WoSt
    float robinCoeff;                            // not populated by WoSt, but available for downstream use (e.g. BVC, RWS)
    T solution, normalDerivative, contribution;  // not populated by WoSt, but available for downstream use (e.g. BVC, RWS)
    bool estimateBoundaryNormalAligned;
};

} // zombie

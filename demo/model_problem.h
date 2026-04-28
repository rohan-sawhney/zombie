// This file defines a ModelProblem class, which is used to describe a scalar- or
// vector-valued Poisson or screened Poisson PDE on a 2D domain via a boundary mesh,
// associated boundary conditions, source term, and robin and absorption coefficients.
//
// The boundary mesh is read from an OBJ file, while the input PDE data is read
// from images for the purposes of this demo. NOTE: Users may analogously define
// a ModelProblem class for 3D domains and/or vector-valued PDEs, as all functionality
// in Zombie is templated on the dimension and value type of the PDE.

#pragma once

#include <zombie/zombie.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

template <typename T>
class ModelProblem {
public:
    static constexpr size_t CHANNELS = ValueTraits<T>::channels;
    static constexpr size_t IMAGE_CHANNELS = ValueTraits<T>::imageChannels;

    // constructor
    ModelProblem(const json& config, std::string directoryPath);

    // getters
    bool solveDoubleSided() { return mSolveDoubleSided; }
    bool solveExterior() { return mSolveExterior; }
    const std::vector<Vector2i>& getBoundaryIndices() { return mIndices; }
    const std::vector<Vector2i>& getAbsorbingBoundaryIndices() { return mAbsorbingBoundaryIndices; }
    const std::vector<Vector2i>& getReflectingBoundaryIndices() { return mReflectingBoundaryIndices; }

    // getters for input domain
    const std::vector<Vector2>& getBoundaryPositions() { return mPositions; }
    const std::vector<Vector2>& getAbsorbingBoundaryPositions() { return mAbsorbingBoundaryPositions; }
    const std::vector<Vector2>& getReflectingBoundaryPositions() { return mReflectingBoundaryPositions; }
    const std::pair<Vector2, Vector2>& getBoundingBox() { return mBoundingBox; }
    const zombie::PDE<T, 2>& getPDE() { return mPde; }
    const zombie::GeometricQueries<2>& getGeometricQueries() { return mQueries; }

    // getters for inverted domain (needed for solving exterior problems)
    const zombie::KelvinTransform<T, 2>& getKelvinTransform() { return mKelvinTransform; }
    const std::vector<Vector2>& getInvertedAbsorbingBoundaryPositions() { return mInvertedAbsorbingBoundaryPositions; }
    const std::vector<Vector2>& getInvertedReflectingBoundaryPositions() { return mInvertedReflectingBoundaryPositions; }
    const std::pair<Vector2, Vector2>& getInvertedBoundingBox() { return mInvertedBoundingBox; }
    const zombie::PDE<T, 2>& getPDEInvertedDomain() { return mPdeInvertedDomain; }
    const zombie::GeometricQueries<2>& getGeometricQueriesInvertedDomain() { return mQueriesInvertedDomain; }

protected:
    // loads a boundary mesh from an OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // converts image values to PDE values
    T getPDEValue(const Image<IMAGE_CHANNELS>& image, Vector2 uv) const;

    // sets up the PDE
    void setupPDE();

    // partitions the boundary mesh into absorbing and reflecting parts
    void partitionBoundaryMesh();

    // populates geometric queries for the absorbing and reflecting boundary
    void populateGeometricQueries(const std::vector<Vector2>& absorbingBoundaryPositions,
                                  const std::vector<Vector2>& reflectingBoundaryPositions,
                                  const std::pair<Vector2, Vector2>& boundingBox,
                                  const std::vector<float>& minRobinCoeffValues,
                                  const std::vector<float>& maxRobinCoeffValues,
                                  bool areRobinConditionsPureNeumann,
                                  std::unique_ptr<zombie::SdfGrid<2>>& sdfGridForAbsorbingBoundary,
                                  zombie::FcpwDirichletBoundaryHandler<2>& absorbingBoundaryHandler,
                                  zombie::FcpwNeumannBoundaryHandler<2>& reflectingNeumannBoundaryHandler,
                                  zombie::FcpwRobinBoundaryHandler<2>& reflectingRobinBoundaryHandler,
                                  zombie::GeometricQueries<2>& queries);

    // applies a Kelvin transform to convert an exterior problem into an
    // equivalent interior problem with a modified PDE on the inverted domain
    void invertExteriorProblem();

    // members
    Image<IMAGE_CHANNELS> mSourceValue;
    Image<IMAGE_CHANNELS> mAbsorbingBoundaryValue;
    Image<IMAGE_CHANNELS> mReflectingBoundaryValue;
    Image<IMAGE_CHANNELS> mAbsorbingBoundaryNormalAlignedValue;
    Image<IMAGE_CHANNELS> mReflectingBoundaryNormalAlignedValue;
    Image<1> mIsReflectingBoundary;
    bool mSolveDoubleSided;
    bool mSolveExterior;
    bool mDomainIsWatertight;
    bool mUseSdfForAbsorbingBoundary;
    int mSdfGridResolution;
    float mRobinCoeff;
    float mAbsorptionCoeff;

    std::vector<Vector2i> mIndices;
    std::vector<Vector2i> mAbsorbingBoundaryIndices;
    std::vector<Vector2i> mReflectingBoundaryIndices;

    std::vector<Vector2> mPositions;
    std::vector<Vector2> mAbsorbingBoundaryPositions;
    std::vector<Vector2> mReflectingBoundaryPositions;
    std::pair<Vector2, Vector2> mBoundingBox;
    zombie::PDE<T, 2> mPde;
    std::vector<float> mMinRobinCoeffValues;
    std::vector<float> mMaxRobinCoeffValues;
    std::unique_ptr<zombie::SdfGrid<2>> mSdfGridForAbsorbingBoundary;
    zombie::FcpwDirichletBoundaryHandler<2> mAbsorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<2> mReflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<2> mReflectingRobinBoundaryHandler;
    zombie::GeometricQueries<2> mQueries;

    zombie::KelvinTransform<T, 2> mKelvinTransform;
    std::vector<Vector2> mInvertedAbsorbingBoundaryPositions;
    std::vector<Vector2> mInvertedReflectingBoundaryPositions;
    std::pair<Vector2, Vector2> mInvertedBoundingBox;
    zombie::PDE<T, 2> mPdeInvertedDomain;
    std::vector<float> mMinRobinCoeffValuesInvertedDomain;
    std::vector<float> mMaxRobinCoeffValuesInvertedDomain;
    std::unique_ptr<zombie::SdfGrid<2>> mSdfGridForInvertedAbsorbingBoundary;
    zombie::FcpwDirichletBoundaryHandler<2> mInvertedAbsorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<2> mInvertedReflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<2> mInvertedReflectingRobinBoundaryHandler;
    zombie::GeometricQueries<2> mQueriesInvertedDomain;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename T>
ModelProblem<T>::ModelProblem(const json& config, std::string directoryPath):
mKelvinTransform(Vector2(0.0f, 0.125f)), // ensure origin lies inside default domain for demo, a requirement for exterior problems
mSdfGridForAbsorbingBoundary(nullptr),
mSdfGridForInvertedAbsorbingBoundary(nullptr)
{
    // load config settings
    auto getFilePath = [config, directoryPath](const std::string& fileName) -> std::string {
        return directoryPath + getRequired<std::string>(config, fileName);
    };

    std::string geometryFile = getFilePath("geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    mSolveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    mSourceValue = Image<IMAGE_CHANNELS>(getFilePath("sourceValue"));
    mAbsorbingBoundaryValue = Image<IMAGE_CHANNELS>(getFilePath("absorbingBoundaryValue"));
    mReflectingBoundaryValue = Image<IMAGE_CHANNELS>(getFilePath("reflectingBoundaryValue"));
    if (mSolveDoubleSided) {
        mAbsorbingBoundaryNormalAlignedValue = Image<IMAGE_CHANNELS>(getFilePath("absorbingBoundaryNormalAlignedValue"));
        mReflectingBoundaryNormalAlignedValue = Image<IMAGE_CHANNELS>(getFilePath("reflectingBoundaryNormalAlignedValue"));
        if (mAbsorbingBoundaryNormalAlignedValue.h != mAbsorbingBoundaryValue.h ||
            mAbsorbingBoundaryNormalAlignedValue.w != mAbsorbingBoundaryValue.w ||
            mReflectingBoundaryNormalAlignedValue.h != mReflectingBoundaryValue.h ||
            mReflectingBoundaryNormalAlignedValue.w != mReflectingBoundaryValue.w) {
            std::cerr << "Error: normal-aligned value buffers must be the same shape as the value buffers" << std::endl;
            exit(EXIT_FAILURE);
        }
    }
    mIsReflectingBoundary = Image<1>(directoryPath + getRequired<std::string>(config, "isReflectingBoundary"));
    mSolveExterior = getOptional<bool>(config, "solveExterior", false);
    mDomainIsWatertight = getOptional<bool>(config, "domainIsWatertight", false);
    mUseSdfForAbsorbingBoundary = getOptional<bool>(config, "useSdfForAbsorbingBoundary", false);
    mSdfGridResolution = getOptional<int>(config, "sdfGridResolution", 128);
    mRobinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    mAbsorptionCoeff = mSolveExterior ? 0.0f : getOptional<float>(config, "absorptionCoeff", 0.0f); // kelvin transform requires absorption coefficient to be 0

    // load a boundary mesh from an OBJ file
    loadOBJ(geometryFile, normalize, flipOrientation);

    // setup the PDE
    setupPDE();

    // partition the boundary mesh into absorbing and reflecting boundary elements
    partitionBoundaryMesh();

    // specify the minimum and maximum Robin coefficient values for each reflecting boundary element:
    // we use a constant value for all elements in this demo, but Zombie supports variable coefficients
    mMinRobinCoeffValues.resize(mReflectingBoundaryIndices.size(), std::fabs(mRobinCoeff));
    mMaxRobinCoeffValues.resize(mReflectingBoundaryIndices.size(), std::fabs(mRobinCoeff));

    // populate the geometric queries for the absorbing and reflecting boundary
    populateGeometricQueries(mAbsorbingBoundaryPositions, mReflectingBoundaryPositions,
                             mBoundingBox, mMinRobinCoeffValues, mMaxRobinCoeffValues,
                             mPde.areRobinConditionsPureNeumann, mSdfGridForAbsorbingBoundary,
                             mAbsorbingBoundaryHandler, mReflectingNeumannBoundaryHandler,
                             mReflectingRobinBoundaryHandler, mQueries);

    if (mSolveExterior) {
        // invert the exterior problem into an equivalent interior problem
        invertExteriorProblem();

        // populate the geometric queries for the inverted absorbing and reflecting boundary
        populateGeometricQueries(mInvertedAbsorbingBoundaryPositions,
                                 mInvertedReflectingBoundaryPositions,
                                 mInvertedBoundingBox,
                                 mMinRobinCoeffValuesInvertedDomain,
                                 mMaxRobinCoeffValuesInvertedDomain,
                                 mPdeInvertedDomain.areRobinConditionsPureNeumann,
                                 mSdfGridForInvertedAbsorbingBoundary,
                                 mInvertedAbsorbingBoundaryHandler,
                                 mInvertedReflectingNeumannBoundaryHandler,
                                 mInvertedReflectingRobinBoundaryHandler,
                                 mQueriesInvertedDomain);
    }
}

template <typename T>
void ModelProblem<T>::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation)
{
    zombie::loadBoundaryMesh<2>(filename, mPositions, mIndices);
    if (normalize) zombie::normalize<2>(mPositions);
    if (flipOrientation) zombie::flipOrientation<2>(mIndices);
    mBoundingBox = zombie::computeBoundingBox<2>(mPositions, true, 1.0);
}

template <typename T>
T ModelProblem<T>::getPDEValue(const Image<IMAGE_CHANNELS>& image, Vector2 uv) const
{
    if constexpr (std::is_same_v<T, float>) {
        return image.get(uv)[0];

    } else {
        T value = T::Zero();
        value.template head<IMAGE_CHANNELS>() = image.get(uv);
        return value;
    }
}

template <typename T>
void ModelProblem<T>::setupPDE()
{
    Vector2 bMin = mBoundingBox.first;
    Vector2 bMax = mBoundingBox.second;
    float maxLength = (bMax - bMin).maxCoeff();

    mPde.source = [this, bMin, maxLength](const Vector2& x) -> T {
        Vector2 uv = (x - bMin)/maxLength;
        return this->getPDEValue(this->mSourceValue, uv);
    };
    mPde.dirichlet = [this, bMin, maxLength](const Vector2& x,
                                             bool returnBoundaryNormalAlignedValue) -> T {
        Vector2 uv = (x - bMin)/maxLength;
        return returnBoundaryNormalAlignedValue ?
                this->getPDEValue(this->mAbsorbingBoundaryNormalAlignedValue, uv) :
                this->getPDEValue(this->mAbsorbingBoundaryValue, uv);
    };
    mPde.robin = [this, bMin, maxLength](const Vector2& x, const Vector2& n,
                                         bool returnBoundaryNormalAlignedValue) -> T {
        Vector2 uv = (x - bMin)/maxLength;
        return returnBoundaryNormalAlignedValue ?
                this->getPDEValue(this->mReflectingBoundaryNormalAlignedValue, uv) :
                this->getPDEValue(this->mReflectingBoundaryValue, uv);
    };
    mPde.robinCoeff = [this](const Vector2& x, const Vector2& n,
                             bool returnBoundaryNormalAlignedValue) -> float {
        return this->mRobinCoeff;
    };
    mPde.hasReflectingBoundaryConditions = [this, bMin, maxLength](const Vector2& x) -> bool {
        Vector2 uv = (x - bMin)/maxLength;
        return this->mSolveExterior ? this->mReflectingBoundaryValue.get(uv)[0] > 0 :
                                      this->mIsReflectingBoundary.get(uv)[0] > 0;
    };
    mPde.areRobinConditionsPureNeumann = mRobinCoeff == 0.0f;
    mPde.areRobinCoeffsNonnegative = mRobinCoeff >= 0.0f;
    mPde.absorptionCoeff = mAbsorptionCoeff;
}

template <typename T>
void ModelProblem<T>::partitionBoundaryMesh()
{
    // use Zombie's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    zombie::partitionBoundaryMesh<2>(mPde.hasReflectingBoundaryConditions, mPositions, mIndices,
                                     mAbsorbingBoundaryPositions, mAbsorbingBoundaryIndices,
                                     mReflectingBoundaryPositions, mReflectingBoundaryIndices);
}

template <typename T>
void ModelProblem<T>::populateGeometricQueries(const std::vector<Vector2>& absorbingBoundaryPositions,
                                               const std::vector<Vector2>& reflectingBoundaryPositions,
                                               const std::pair<Vector2, Vector2>& boundingBox,
                                               const std::vector<float>& minRobinCoeffValues,
                                               const std::vector<float>& maxRobinCoeffValues,
                                               bool areRobinConditionsPureNeumann,
                                               std::unique_ptr<zombie::SdfGrid<2>>& sdfGridForAbsorbingBoundary,
                                               zombie::FcpwDirichletBoundaryHandler<2>& absorbingBoundaryHandler,
                                               zombie::FcpwNeumannBoundaryHandler<2>& reflectingNeumannBoundaryHandler,
                                               zombie::FcpwRobinBoundaryHandler<2>& reflectingRobinBoundaryHandler,
                                               zombie::GeometricQueries<2>& queries)
{
    // set the domain extent for geometric queries
    queries.domainIsWatertight = mDomainIsWatertight;
    queries.domainMin = boundingBox.first;
    queries.domainMax = boundingBox.second;

    // use an absorbing boundary handler to populate geometric queries for the absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryPositions, mAbsorbingBoundaryIndices);
    zombie::populateGeometricQueriesForDirichletBoundary<2>(absorbingBoundaryHandler, queries);

    if (!mSolveDoubleSided && mUseSdfForAbsorbingBoundary) {
        // override distance queries to use an SDF grid. The user can also use Zombie to build
        // an SDF hierarchy for double-sided problems (ommited here for simplicity)
        sdfGridForAbsorbingBoundary = std::make_unique<zombie::SdfGrid<2>>(queries.domainMin, queries.domainMax);
        Vector2i sdfGridShape = Vector2i::Constant(mSdfGridResolution);
        zombie::populateSdfGrid<2>(absorbingBoundaryHandler, *sdfGridForAbsorbingBoundary, sdfGridShape);
        zombie::populateGeometricQueriesForDirichletBoundary<zombie::SdfGrid<2>, 2>(*sdfGridForAbsorbingBoundary, queries);
    }

    // use a reflecting boundary handler to populate geometric queries for the reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette =
        zombie::getIgnoreCandidateSilhouetteCallback(mSolveDoubleSided);
    std::function<float(float)> branchTraversalWeight = zombie::getBranchTraversalWeightCallback();

    if (areRobinConditionsPureNeumann) {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, mReflectingBoundaryIndices, ignoreCandidateSilhouette);
        zombie::populateGeometricQueriesForNeumannBoundary<2>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, mReflectingBoundaryIndices, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForRobinBoundary<2>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);
    }
}

template <typename T>
void ModelProblem<T>::invertExteriorProblem()
{
    // invert the domain
    std::vector<Vector2> invertedPositions;
    mKelvinTransform.transformPoints(mPositions, invertedPositions);
    mKelvinTransform.transformPoints(mAbsorbingBoundaryPositions, mInvertedAbsorbingBoundaryPositions);
    mKelvinTransform.transformPoints(mReflectingBoundaryPositions, mInvertedReflectingBoundaryPositions);

    // compute the bounding box for the inverted domain
    mInvertedBoundingBox = zombie::computeBoundingBox<2>(invertedPositions, true, 1.0);

    // setup a modified PDE on the inverted domain
    mKelvinTransform.transformPde(mPde, mPdeInvertedDomain);

    if(!mPdeInvertedDomain.areRobinConditionsPureNeumann) {
        // compute the modified Robin coefficients on the inverted domain
        std::vector<float> minRobinCoeffValues(mReflectingBoundaryIndices.size(), mRobinCoeff);
        std::vector<float> maxRobinCoeffValues(mReflectingBoundaryIndices.size(), mRobinCoeff);
        mKelvinTransform.computeRobinCoefficients(mInvertedReflectingBoundaryPositions,
                                                  mReflectingBoundaryIndices,
                                                  minRobinCoeffValues, maxRobinCoeffValues,
                                                  mMinRobinCoeffValuesInvertedDomain,
                                                  mMaxRobinCoeffValuesInvertedDomain);
    }
}

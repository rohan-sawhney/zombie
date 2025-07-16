// This file defines a ModelProblem class, which is used to describe a scalar-valued
// Poisson or screened Poisson PDE on a 2D domain via a boundary mesh, associated
// boundary conditions, source term, and robin and absorption coefficients.
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

class ModelProblem {
public:
    // constructor
    ModelProblem(const json& config, std::string directoryPath);

    // members
    bool solveDoubleSided;
    bool solveExterior;
    std::vector<Vector2i> indices;
    std::vector<Vector2i> absorbingBoundaryIndices;
    std::vector<Vector2i> reflectingBoundaryIndices;

    // members for input domain
    std::vector<Vector2> positions;
    std::vector<Vector2> absorbingBoundaryPositions;
    std::vector<Vector2> reflectingBoundaryPositions;
    std::pair<Vector2, Vector2> boundingBox;
    zombie::PDE<float, 2> pde;
    zombie::GeometricQueries<2> queries;

    // members for inverted domain (needed for solving exterior problems)
    zombie::KelvinTransform<float, 2> kelvinTransform;
    std::vector<Vector2> invertedAbsorbingBoundaryPositions;
    std::vector<Vector2> invertedReflectingBoundaryPositions;
    std::pair<Vector2, Vector2> invertedBoundingBox;
    zombie::PDE<float, 2> pdeInvertedDomain;
    zombie::GeometricQueries<2> queriesInvertedDomain;

protected:
    // loads a boundary mesh from an OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

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
    Image<1> isReflectingBoundary;
    Image<1> absorbingBoundaryValue;
    Image<1> reflectingBoundaryValue;
    Image<1> sourceValue;
    bool domainIsWatertight;
    bool useSdfForAbsorbingBoundary;
    int sdfGridResolution;
    float robinCoeff, absorptionCoeff;

    std::vector<float> minRobinCoeffValues;
    std::vector<float> maxRobinCoeffValues;
    std::unique_ptr<zombie::SdfGrid<2>> sdfGridForAbsorbingBoundary;
    zombie::FcpwDirichletBoundaryHandler<2> absorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<2> reflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<2> reflectingRobinBoundaryHandler;

    std::vector<float> minRobinCoeffValuesInvertedDomain;
    std::vector<float> maxRobinCoeffValuesInvertedDomain;
    std::unique_ptr<zombie::SdfGrid<2>> sdfGridForInvertedAbsorbingBoundary;
    zombie::FcpwDirichletBoundaryHandler<2> invertedAbsorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<2> invertedReflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<2> invertedReflectingRobinBoundaryHandler;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

ModelProblem::ModelProblem(const json& config, std::string directoryPath):
kelvinTransform(Vector2(0.0f, 0.125f)), // ensure origin lies inside default domain for demo, a requirement for exterior problems
sdfGridForAbsorbingBoundary(nullptr),
sdfGridForInvertedAbsorbingBoundary(nullptr)
{
    // load config settings
    std::string geometryFile = directoryPath + getRequired<std::string>(config, "geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    isReflectingBoundary = Image<1>(directoryPath + getRequired<std::string>(config, "isReflectingBoundary"));
    absorbingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "absorbingBoundaryValue"));
    reflectingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "reflectingBoundaryValue"));
    sourceValue = Image<1>(directoryPath + getRequired<std::string>(config, "sourceValue"));
    solveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    solveExterior = getOptional<bool>(config, "solveExterior", false);
    domainIsWatertight = getOptional<bool>(config, "domainIsWatertight", true);
    useSdfForAbsorbingBoundary = getOptional<bool>(config, "useSdfForAbsorbingBoundary", false);
    sdfGridResolution = getOptional<int>(config, "sdfGridResolution", 128);
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);

    // load a boundary mesh from an OBJ file
    loadOBJ(geometryFile, normalize, flipOrientation);

    // setup the PDE
    setupPDE();

    // partition the boundary mesh into absorbing and reflecting boundary elements
    partitionBoundaryMesh();

    // specify the minimum and maximum Robin coefficient values for each reflecting boundary element:
    // we use a constant value for all elements in this demo, but Zombie supports variable coefficients
    minRobinCoeffValues.resize(reflectingBoundaryIndices.size(), std::fabs(robinCoeff));
    maxRobinCoeffValues.resize(reflectingBoundaryIndices.size(), std::fabs(robinCoeff));

    // populate the geometric queries for the absorbing and reflecting boundary
    populateGeometricQueries(absorbingBoundaryPositions, reflectingBoundaryPositions,
                             boundingBox, minRobinCoeffValues, maxRobinCoeffValues,
                             pde.areRobinConditionsPureNeumann, sdfGridForAbsorbingBoundary,
                             absorbingBoundaryHandler, reflectingNeumannBoundaryHandler,
                             reflectingRobinBoundaryHandler, queries);

    if (solveExterior) {
        // invert the exterior problem into an equivalent interior problem
        invertExteriorProblem();

        // populate the geometric queries for the inverted absorbing and reflecting boundary
        populateGeometricQueries(invertedAbsorbingBoundaryPositions, invertedReflectingBoundaryPositions,
                                 invertedBoundingBox, minRobinCoeffValuesInvertedDomain, maxRobinCoeffValuesInvertedDomain,
                                 pdeInvertedDomain.areRobinConditionsPureNeumann, sdfGridForInvertedAbsorbingBoundary,
                                 invertedAbsorbingBoundaryHandler, invertedReflectingNeumannBoundaryHandler,
                                 invertedReflectingRobinBoundaryHandler, queriesInvertedDomain);
    }
}

void ModelProblem::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation)
{
    zombie::loadBoundaryMesh<2>(filename, positions, indices);
    if (normalize) zombie::normalize<2>(positions);
    if (flipOrientation) zombie::flipOrientation<2>(indices);
    boundingBox = zombie::computeBoundingBox<2>(positions, true, 1.0);
}

void ModelProblem::setupPDE()
{
    Vector2 bMin = boundingBox.first;
    Vector2 bMax = boundingBox.second;
    float maxLength = (bMax - bMin).maxCoeff();

    pde.source = [this, bMin, maxLength](const Vector2& x) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->sourceValue.get(uv)[0];
    };
    pde.dirichlet = [this, bMin, maxLength](const Vector2& x, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->absorbingBoundaryValue.get(uv)[0];
    };
    pde.robin = [this, bMin, maxLength](const Vector2& x, const Vector2& n, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->reflectingBoundaryValue.get(uv)[0];
    };
    pde.robinCoeff = [this](const Vector2& x, const Vector2& n, bool _) -> float {
        return this->robinCoeff;
    };
    pde.hasReflectingBoundaryConditions = [this, bMin, maxLength](const Vector2& x) -> bool {
        Vector2 uv = (x - bMin)/maxLength;
        return this->solveExterior ? this->reflectingBoundaryValue.get(uv)[0] > 0 :
                                     this->isReflectingBoundary.get(uv)[0] > 0;
    };
    pde.areRobinConditionsPureNeumann = robinCoeff == 0.0f;
    pde.areRobinCoeffsNonnegative = robinCoeff >= 0.0f;
    pde.absorptionCoeff = absorptionCoeff;
}

void ModelProblem::partitionBoundaryMesh()
{
    // use Zombie's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    zombie::partitionBoundaryMesh<2>(pde.hasReflectingBoundaryConditions, positions, indices,
                                     absorbingBoundaryPositions, absorbingBoundaryIndices,
                                     reflectingBoundaryPositions, reflectingBoundaryIndices);
}

void ModelProblem::populateGeometricQueries(const std::vector<Vector2>& absorbingBoundaryPositions,
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
    queries.domainIsWatertight = domainIsWatertight;
    queries.domainMin = boundingBox.first;
    queries.domainMax = boundingBox.second;

    // use an absorbing boundary handler to populate geometric queries for the absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryPositions, absorbingBoundaryIndices);
    zombie::populateGeometricQueriesForDirichletBoundary<2>(absorbingBoundaryHandler, queries);

    if (!solveDoubleSided && useSdfForAbsorbingBoundary) {
        // override distance queries to use an SDF grid. The user can also use Zombie to build
        // an SDF hierarchy for double-sided problems (ommited here for simplicity)
        sdfGridForAbsorbingBoundary = std::make_unique<zombie::SdfGrid<2>>(queries.domainMin, queries.domainMax);
        zombie::Vector2i sdfGridShape = zombie::Vector2i::Constant(sdfGridResolution);
        zombie::populateSdfGrid<2>(absorbingBoundaryHandler, *sdfGridForAbsorbingBoundary, sdfGridShape);
        zombie::populateGeometricQueriesForDirichletBoundary<zombie::SdfGrid<2>, 2>(*sdfGridForAbsorbingBoundary, queries);
    }

    // use a reflecting boundary handler to populate geometric queries for the reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette = zombie::getIgnoreCandidateSilhouetteCallback(solveDoubleSided);
    std::function<float(float)> branchTraversalWeight = zombie::getBranchTraversalWeightCallback();

    if (areRobinConditionsPureNeumann) {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette);
        zombie::populateGeometricQueriesForNeumannBoundary<2>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryPositions, reflectingBoundaryIndices, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForRobinBoundary<2>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);
    }
}

void ModelProblem::invertExteriorProblem()
{
    // invert the domain
    std::vector<Vector2> invertedPositions;
    kelvinTransform.transformPoints(positions, invertedPositions);
    kelvinTransform.transformPoints(absorbingBoundaryPositions, invertedAbsorbingBoundaryPositions);
    kelvinTransform.transformPoints(reflectingBoundaryPositions, invertedReflectingBoundaryPositions);

    // compute the bounding box for the inverted domain
    invertedBoundingBox = zombie::computeBoundingBox<2>(invertedPositions, true, 1.0);

    // setup a modified PDE on the inverted domain
    kelvinTransform.transformPde(pde, pdeInvertedDomain);

    if(!pdeInvertedDomain.areRobinConditionsPureNeumann) {
        // compute the modified Robin coefficients on the inverted domain
        std::vector<float> minRobinCoeffValues(reflectingBoundaryIndices.size(), robinCoeff);
        std::vector<float> maxRobinCoeffValues(reflectingBoundaryIndices.size(), robinCoeff);
        kelvinTransform.computeRobinCoefficients(invertedReflectingBoundaryPositions,
                                                 reflectingBoundaryIndices,
                                                 minRobinCoeffValues, maxRobinCoeffValues,
                                                 minRobinCoeffValuesInvertedDomain,
                                                 maxRobinCoeffValuesInvertedDomain);
    }
}

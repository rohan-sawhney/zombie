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
    std::vector<Vector2> vertices;
    std::vector<Vector2> absorbingBoundaryVertices;
    std::vector<Vector2> reflectingBoundaryVertices;
    std::vector<Vector2i> segments;
    std::vector<Vector2i> absorbingBoundarySegments;
    std::vector<Vector2i> reflectingBoundarySegments;
    bool solveDoubleSided;
    zombie::PDE<float, 2> pde;
    zombie::GeometricQueries<2> queries;

protected:
    // loads boundary mesh from OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // partitions boundary mesh into absorbing and reflecting parts
    void partitionBoundaryMesh();

    // setup PDE
    void setupPDE();

    // populates geometric queries for boundary mesh
    void populateGeometricQueries();

    // members
    zombie::FcpwDirichletBoundaryHandler<2> absorbingBoundaryHandler;
    zombie::FcpwNeumannBoundaryHandler<2> reflectingNeumannBoundaryHandler;
    zombie::FcpwRobinBoundaryHandler<2> reflectingRobinBoundaryHandler;

    Image<1> isReflectingBoundary;
    Image<1> absorbingBoundaryValue;
    Image<1> reflectingBoundaryValue;
    Image<1> sourceValue;
    float robinCoeff, absorptionCoeff;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

ModelProblem::ModelProblem(const json& config, std::string directoryPath)
{
    // load config settings
    std::string geometryFile = directoryPath + getRequired<std::string>(config, "geometry");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    isReflectingBoundary = Image<1>(directoryPath + getRequired<std::string>(config, "isReflectingBoundary"));
    absorbingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "absorbingBoundaryValue"));
    reflectingBoundaryValue = Image<1>(directoryPath + getRequired<std::string>(config, "reflectingBoundaryValue"));
    sourceValue = Image<1>(directoryPath + getRequired<std::string>(config, "sourceValue"));
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);
    solveDoubleSided = getOptional<bool>(config, "solveDoubleSided", false);
    queries.domainIsWatertight = getOptional<bool>(config, "IsWatertightDomain", true);

    // load boundary mesh, build acceleration structures and set geometric queries and PDE inputs
    loadOBJ(geometryFile, normalize, flipOrientation);
    setupPDE();
    partitionBoundaryMesh();
    populateGeometricQueries();
}

void ModelProblem::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation)
{
    zombie::loadBoundaryMesh<2>(filename, vertices, segments);
    if (normalize) zombie::normalize<2>(vertices);
    if (flipOrientation) zombie::flipOrientation<2>(segments);
    std::pair<Vector2, Vector2> bbox = zombie::computeBoundingBox<2>(vertices, true, 1.0);
    queries.domainMin = bbox.first;
    queries.domainMax = bbox.second;
}

void ModelProblem::setupPDE()
{
    const Vector2& bMin = queries.domainMin;
    const Vector2& bMax = queries.domainMax;
    float maxLength = (bMax - bMin).maxCoeff();

    pde.source = [this, &bMin, maxLength](const Vector2& x) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->sourceValue.get(uv)[0];
    };
    pde.dirichlet = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->absorbingBoundaryValue.get(uv)[0];
    };
    pde.robin = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->reflectingBoundaryValue.get(uv)[0];
    };
    pde.robinCoeff = [this](const Vector2& x, bool _) -> float {
        return this->robinCoeff;
    };
    pde.hasReflectingBoundaryConditions = [this, &bMin, maxLength](const Vector2& x) -> bool {
        Vector2 uv = (x - bMin)/maxLength;
        return this->isReflectingBoundary.get(uv)[0] > 0;
    };
    pde.areRobinConditionsPureNeumann = robinCoeff == 0.0f;
    pde.absorptionCoeff = absorptionCoeff;
}

void ModelProblem::partitionBoundaryMesh()
{
    // use zombie's default partitioning function, which assumes the boundary discretization
    // is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    // and the user might want to override this function for their specific problem
    zombie::partitionBoundaryMesh<2>(pde.hasReflectingBoundaryConditions, vertices, segments,
                                     absorbingBoundaryVertices, absorbingBoundarySegments,
                                     reflectingBoundaryVertices, reflectingBoundarySegments);
}

void ModelProblem::populateGeometricQueries()
{
    // build acceleration structure and populate geometric queries for absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryVertices, absorbingBoundarySegments);
    zombie::populateGeometricQueriesForDirichletBoundary<2>(absorbingBoundaryHandler, queries);

    // build acceleration structure and populate geometric queries for reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette = zombie::getIgnoreCandidateSilhouetteCallback(solveDoubleSided);
    std::function<float(float)> branchTraversalWeight = zombie::getBranchTraversalWeightCallback();

    if (robinCoeff > 0.0f) {
        // despite using a constant Robin coefficient here, the implementation supports
        // varying coefficients over the boundary
        std::vector<float> minRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
        std::vector<float> maxRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryVertices, reflectingBoundarySegments, ignoreCandidateSilhouette,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForRobinBoundary<2>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryVertices, reflectingBoundarySegments, ignoreCandidateSilhouette);
        zombie::populateGeometricQueriesForNeumannBoundary<2>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);
    }
}

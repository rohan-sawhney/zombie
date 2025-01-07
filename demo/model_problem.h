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
    ModelProblem(const json& config);

    // members
    std::vector<Vector2> vertices;
    std::vector<Vector2> absorbingBoundaryVertices;
    std::vector<Vector2> reflectingBoundaryVertices;
    std::vector<Vector2i> segments;
    std::vector<Vector2i> absorbingBoundarySegments;
    std::vector<Vector2i> reflectingBoundarySegments;
    const bool solveDoubleSided;
    zombie::PDE<float, 2> pde;
    zombie::GeometricQueries<2> queries;

protected:
    // loads boundary mesh from OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // setup PDE
    void setupPDE();

    // populates geometric queries for boundary mesh
    void populateGeometricQueries();

    // members
    zombie::FcpwBoundaryHandler<2, false> absorbingBoundaryHandler;
    zombie::FcpwBoundaryHandler<2, false> reflectingNeumannBoundaryHandler;
    zombie::FcpwBoundaryHandler<2, true> reflectingRobinBoundaryHandler;

    std::shared_ptr<Image<1>> isReflectingBoundary;
    std::shared_ptr<Image<1>> absorbingBoundaryValue;
    std::shared_ptr<Image<1>> reflectingBoundaryValue;
    std::shared_ptr<Image<1>> sourceValue;
    float absorptionCoeff, robinCoeff;

    std::function<float(float)> branchTraversalWeight;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

ModelProblem::ModelProblem(const json& config):
solveDoubleSided(getOptional<bool>(config, "solveDoubleSided", false))
{
    // load config settings
    const std::string geometryFile = getRequired<std::string>(config, "geometry");
    const std::string isReflectingBoundaryFile = getRequired<std::string>(config, "isReflectingBoundary");
    const std::string absorbingBoundaryValueFile = getRequired<std::string>(config, "absorbingBoundaryValue");
    const std::string reflectingBoundaryValueFile = getRequired<std::string>(config, "reflectingBoundaryValue");
    const std::string sourceValueFile = getRequired<std::string>(config, "sourceValue");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);
    queries.domainIsWatertight = getOptional<bool>(config, "IsWatertightDomain", true);

    // load images specifying boundary conditions and source term
    isReflectingBoundary = std::make_shared<Image<1>>(isReflectingBoundaryFile);
    absorbingBoundaryValue = std::make_shared<Image<1>>(absorbingBoundaryValueFile);
    reflectingBoundaryValue = std::make_shared<Image<1>>(reflectingBoundaryValueFile);
    sourceValue = std::make_shared<Image<1>>(sourceValueFile);

    // load boundary mesh, build acceleration structures and set geometric queries and PDE inputs
    loadOBJ(geometryFile, normalize, flipOrientation);
    setupPDE();
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
        return this->sourceValue->get(uv)[0];
    };
    pde.dirichlet = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->absorbingBoundaryValue->get(uv)[0];
    };
    pde.robin = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->reflectingBoundaryValue->get(uv)[0];
    };
    pde.robinCoeff = [this](const Vector2& x, bool _) -> float {
        return this->robinCoeff;
    };
    pde.hasReflectingBoundaryConditions = [this, &bMin, maxLength](const Vector2& x) -> bool {
        Vector2 uv = (x - bMin)/maxLength;
        return this->isReflectingBoundary->get(uv)[0] > 0;
    };
    pde.areRobinConditionsPureNeumann = robinCoeff == 0.0f;
    pde.absorptionCoeff = absorptionCoeff;
}

void ModelProblem::populateGeometricQueries()
{
    // partition boundary vertices and indices into absorbing and reflecting parts
    zombie::partitionBoundaryMesh<2>(pde.hasReflectingBoundaryConditions, vertices, segments,
                                     absorbingBoundaryVertices, absorbingBoundarySegments,
                                     reflectingBoundaryVertices, reflectingBoundarySegments);

    // build acceleration structure and populate geometric queries for absorbing boundary
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryVertices, absorbingBoundarySegments);
    zombie::populateGeometricQueriesForAbsorbingBoundary<2>(absorbingBoundaryHandler, queries);

    // build acceleration structure and populate geometric queries for reflecting boundary
    std::function<bool(float, int)> ignoreCandidateSilhouette = zombie::getIgnoreCandidateSilhouetteCallback(solveDoubleSided);
    branchTraversalWeight = zombie::getBranchTraversalWeight();

    if (robinCoeff > 0.0f) {
        std::vector<float> minRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
        std::vector<float> maxRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
        reflectingRobinBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryVertices, reflectingBoundarySegments, ignoreCandidateSilhouette, false,
            minRobinCoeffValues, maxRobinCoeffValues);
        zombie::populateGeometricQueriesForReflectingBoundary<2, true>(
            reflectingRobinBoundaryHandler, branchTraversalWeight, queries);

    } else {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(
            reflectingBoundaryVertices, reflectingBoundarySegments, ignoreCandidateSilhouette, true);
        zombie::populateGeometricQueriesForReflectingBoundary<2, false>(
            reflectingNeumannBoundaryHandler, branchTraversalWeight, queries);
    }
}

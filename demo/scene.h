// This file defines a Scene class, which is used to describe a scalar-valued
// Poisson or screened Poisson PDE on a 2D domain via a boundary mesh, associated
// boundary conditions, source term, and robin and absorption coefficients.
//
// The boundary mesh is read from an OBJ file, while the input PDE data is read
// from images for the purposes of this demo. The user may analogously define
// a Scene class for 3D domains and/or vector-valued PDEs, as all functionality
// in Zombie is templated on the dimension and value type of the PDE.

#pragma once

#include <zombie/core/pde.h>
#include <zombie/utils/fcpw_boundary_handler.h>
#include <fstream>
#include <sstream>
#include "config.h"
#include "image.h"

class Scene {
public:
    // constructor
    Scene(const json& config);

    // check if a point is on a reflecting boundary
    bool onReflectingBoundary(const Vector2& x) const;

    // returns the volume of the solve region
    float getSolveRegionVolume() const;

    // members
    std::pair<Vector2, Vector2> bbox;
    std::vector<Vector2> vertices;
    std::vector<std::vector<size_t>> segments;
    const bool isWatertight;
    const bool isDoubleSided;
    zombie::GeometricQueries<2> queries;
    zombie::PDE<float, 2> pde;

private:
    // loads boundary mesh from OBJ file
    void loadOBJ(const std::string& filename, bool normalize, bool flipOrientation);

    // builds acceleration structures for boundary mesh
    void buildAccelerationStructures();

    // populates geometric queries for boundary mesh
    void populateGeometricQueries();

    // populates PDE inputs
    void populatePDEInputs();

    // members
    std::vector<Vector2> absorbingBoundaryVertices;
    std::vector<Vector2> reflectingBoundaryVertices;
    std::vector<std::vector<size_t>> absorbingBoundarySegments;
    std::vector<std::vector<size_t>> reflectingBoundarySegments;

    zombie::FcpwBoundaryHandler<2, false> absorbingBoundaryHandler;
    zombie::FcpwBoundaryHandler<2, false> reflectingNeumannBoundaryHandler;
    zombie::FcpwBoundaryHandler<2, true> reflectingRobinBoundaryHandler;

    std::shared_ptr<Image<1>> isReflectingBoundary;
    std::shared_ptr<Image<1>> absorbingBoundaryValue;
    std::shared_ptr<Image<1>> reflectingBoundaryValue;
    std::shared_ptr<Image<1>> sourceValue;
    float absorptionCoeff, robinCoeff;

    std::function<bool(float, int)> ignoreCandidateSilhouette;
    zombie::HarmonicGreensFnFreeSpace<3> harmonicGreensFn;
    std::function<float(float)> branchTraversalWeight;
};

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

Scene::Scene(const json& config):
isWatertight(getOptional<bool>(config, "isWatertight", true)),
isDoubleSided(getOptional<bool>(config, "isDoubleSided", false)),
queries(isWatertight) {
    // load config settings
    const std::string boundaryFile = getRequired<std::string>(config, "boundary");
    const std::string isReflectingBoundaryFile = getRequired<std::string>(config, "isReflectingBoundary");
    const std::string absorbingBoundaryValueFile = getRequired<std::string>(config, "absorbingBoundaryValue");
    const std::string reflectingBoundaryValueFile = getRequired<std::string>(config, "reflectingBoundaryValue");
    const std::string sourceValueFile = getRequired<std::string>(config, "sourceValue");
    bool normalize = getOptional<bool>(config, "normalizeDomain", true);
    bool flipOrientation = getOptional<bool>(config, "flipOrientation", true);
    absorptionCoeff = getOptional<float>(config, "absorptionCoeff", 0.0f);
    robinCoeff = getOptional<float>(config, "robinCoeff", 0.0f);

    // load images specifying boundary conditions and source term
    isReflectingBoundary = std::make_shared<Image<1>>(isReflectingBoundaryFile);
    absorbingBoundaryValue = std::make_shared<Image<1>>(absorbingBoundaryValueFile);
    reflectingBoundaryValue = std::make_shared<Image<1>>(reflectingBoundaryValueFile);
    sourceValue = std::make_shared<Image<1>>(sourceValueFile);

    // load boundary mesh, build acceleration structures and set geometric queries and PDE inputs
    loadOBJ(boundaryFile, normalize, flipOrientation);
    buildAccelerationStructures();
    populateGeometricQueries();
    populatePDEInputs();
}

bool Scene::onReflectingBoundary(const Vector2& x) const {
    const Vector2& bMin = bbox.first;
    const Vector2& bMax = bbox.second;
    Vector2 uv = (x - bMin)/(bMax - bMin).maxCoeff();

    return isReflectingBoundary->get(uv)[0] > 0;
}

float Scene::getSolveRegionVolume() const {
    if (isDoubleSided) return (bbox.second - bbox.first).prod();
    return std::fabs(queries.computeSignedDomainVolume());
}

void Scene::loadOBJ(const std::string& filename, bool normalize, bool flipOrientation) {
    zombie::loadBoundaryMesh<2>(filename, vertices, segments);
    if (normalize) zombie::normalize<2>(vertices);
    if (flipOrientation) zombie::flipOrientation(segments);
    bbox = zombie::computeBoundingBox<2>(vertices, true, 1.0);
}

void Scene::buildAccelerationStructures() {
    // separate boundary vertices and indices into absorbing and reflecting parts
    std::vector<size_t> indices(2, -1);
    size_t nAbsorbingBoundaryVerts = 0, nReflectingBoundaryVerts = 0;
    std::unordered_map<size_t, size_t> absorbingBoundaryIndexMap, reflectingBoundaryIndexMap;

    for (int i = 0; i < segments.size(); i++) {
        Vector2 pMid = 0.5f * (vertices[segments[i][0]] + vertices[segments[i][1]]);
        if (onReflectingBoundary(pMid)) {
            for (int j = 0; j < 2; j++) {
                size_t vIndex = segments[i][j];
                if (reflectingBoundaryIndexMap.find(vIndex) == reflectingBoundaryIndexMap.end()) {
                    const Vector2& p = vertices[vIndex];
                    reflectingBoundaryVertices.emplace_back(p);
                    reflectingBoundaryIndexMap[vIndex] = nReflectingBoundaryVerts++;
                }

                indices[j] = reflectingBoundaryIndexMap[vIndex];
            }

            reflectingBoundarySegments.emplace_back(indices);

        } else {
            for (int j = 0; j < 2; j++) {
                size_t vIndex = segments[i][j];
                if (absorbingBoundaryIndexMap.find(vIndex) == absorbingBoundaryIndexMap.end()) {
                    const Vector2& p = vertices[vIndex];
                    absorbingBoundaryVertices.emplace_back(p);
                    absorbingBoundaryIndexMap[vIndex] = nAbsorbingBoundaryVerts++;
                }

                indices[j] = absorbingBoundaryIndexMap[vIndex];
            }

            absorbingBoundarySegments.emplace_back(indices);
        }
    }

    // build acceleration structures for absorbing and reflecting boundaries
    absorbingBoundaryHandler.buildAccelerationStructure(absorbingBoundaryVertices, absorbingBoundarySegments);
    ignoreCandidateSilhouette = [this](float dihedralAngle, int index) -> bool {
        // ignore convex vertices/edges for closest silhouette point tests when solving an interior problem;
        // NOTE: for complex scenes with both open and closed meshes, the primitive index argument
        // (of an adjacent line segment/triangle in the scene) can be used to determine whether a
        // vertex/edge should be ignored as a candidate for silhouette tests.
        return this->isDoubleSided ? false : dihedralAngle < 1e-3f;
    };
    if (robinCoeff > 0.0f) {
        std::vector<float> minRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
        std::vector<float> maxRobinCoeffValues(reflectingBoundarySegments.size(), robinCoeff);
        reflectingRobinBoundaryHandler.buildAccelerationStructure(reflectingBoundaryVertices,
                                                                    reflectingBoundarySegments,
                                                                    ignoreCandidateSilhouette, false,
                                                                    minRobinCoeffValues, maxRobinCoeffValues);

    } else {
        reflectingNeumannBoundaryHandler.buildAccelerationStructure(reflectingBoundaryVertices,
                                                                    reflectingBoundarySegments,
                                                                    ignoreCandidateSilhouette, true);
    }
}

void Scene::populateGeometricQueries() {
    branchTraversalWeight = [this](float r2) -> float {
        float r = std::max(std::sqrt(r2), 1e-2f);
        return std::fabs(this->harmonicGreensFn.evaluate(r));
    };

    if (robinCoeff > 0.0f) {
        zombie::populateGeometricQueries<2, true>(absorbingBoundaryHandler,
                                                  reflectingRobinBoundaryHandler,
                                                  branchTraversalWeight, bbox, queries);

    } else {
        zombie::populateGeometricQueries<2, false>(absorbingBoundaryHandler,
                                                   reflectingNeumannBoundaryHandler,
                                                   branchTraversalWeight, bbox, queries);
    }
}

void Scene::populatePDEInputs() {
    const Vector2& bMin = this->bbox.first;
    const Vector2& bMax = this->bbox.second;
    float maxLength = (bMax - bMin).maxCoeff();

    pde.source = [this, &bMin, maxLength](const Vector2& x) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->sourceValue->get(uv)[0];
    };
    pde.dirichlet = [this, &bMin, maxLength](const Vector2& x) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->absorbingBoundaryValue->get(uv)[0];
    };
    pde.dirichletDoubleSided = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
        Vector2 uv = (x - bMin)/maxLength;
        return this->absorbingBoundaryValue->get(uv)[0];
    };
    if (robinCoeff > 0.0f) {
        pde.robin = [this, &bMin, maxLength](const Vector2& x) -> float {
            Vector2 uv = (x - bMin)/maxLength;
            return this->reflectingBoundaryValue->get(uv)[0];
        };
        pde.robinCoeff = [this](const Vector2& x) -> float {
            return this->robinCoeff;
        };
        pde.robinDoubleSided = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
            Vector2 uv = (x - bMin)/maxLength;
            return this->reflectingBoundaryValue->get(uv)[0];
        };
        pde.robinCoeffDoubleSided = [this](const Vector2& x, bool boundaryNormalAligned) -> float {
            return this->robinCoeff;
        };

    } else {
        pde.neumann = [this, &bMin, maxLength](const Vector2& x) -> float {
            Vector2 uv = (x - bMin)/maxLength;
            return this->reflectingBoundaryValue->get(uv)[0];
        };
        pde.neumannDoubleSided = [this, &bMin, maxLength](const Vector2& x, bool _) -> float {
            Vector2 uv = (x - bMin)/maxLength;
            return this->reflectingBoundaryValue->get(uv)[0];
        };
    }
    pde.absorption = absorptionCoeff;
}
// This file contains helper functions for creating 2D grids and writing
// PDE solutions estimated on these grids to disk.

#pragma once

#include <filesystem>
#include <zombie/core/pde.h>
#include "config.h"
#include "colormap.h"

void writeSolution(const std::string& filename,
                   std::shared_ptr<Image<3>> solution,
                   std::shared_ptr<Image<3>> boundaryDistance,
                   std::shared_ptr<Image<3>> boundaryData,
                   bool saveDebug, bool saveColormapped,
                   std::string colormap, float minVal, float maxVal) {
    // write solution to disk
    std::filesystem::path path(filename);
    std::filesystem::create_directories(path.parent_path());
    solution->write(filename);

    // write colormapped and debug images to disk
    std::string basePath = (path.parent_path() / path.stem()).string();
    std::string ext = path.extension().string();

    if (saveColormapped) {
        getColormappedImage(solution, colormap, minVal, maxVal)->write(basePath + "_color" + ext);
    }

    if (saveDebug) {
        boundaryDistance->write(basePath + "_geometry" + ext);
        boundaryData->write(basePath + "_pde" + ext);
    }
}

void createSolutionGrid(std::vector<zombie::SamplePoint<float, 2>>& samplePts,
                        const zombie::GeometricQueries<2>& queries,
                        const Vector2& bMin, const Vector2& bMax,
                        const int gridRes) {
    // create a grid of sample points
    Vector2 extent = bMax - bMin;
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            Vector2 pt((i/float(gridRes))*extent.x() + bMin.x(),
                       (j/float(gridRes))*extent.y() + bMin.y());
            float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
            float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

            samplePts.emplace_back(zombie::SamplePoint<float, 2>(pt, Vector2::Zero(),
                                                                 zombie::SampleType::InDomain,
                                                                 1.0f, distToAbsorbingBoundary,
                                                                 distToReflectingBoundary, 0.0f));
        }
    }
}

void saveSolutionGrid(const std::vector<zombie::SamplePoint<float, 2>>& samplePts,
                      const zombie::PDE<float, 2>& pde,
                      const zombie::GeometricQueries<2>& queries,
                      const bool isDoubleSided, const json& config) {
    // read settings from config
    const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
    const int gridRes = getRequired<int>(config, "gridRes");
    const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0f);

    const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
    const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
    const std::string colormap = getOptional<std::string>(config, "colormap", "");
    const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0f);
    const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0f);

    // extract solution data
    std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
    std::shared_ptr<Image<3>> boundaryDistance = std::make_shared<Image<3>>(gridRes, gridRes);
    std::shared_ptr<Image<3>> boundaryData = std::make_shared<Image<3>>(gridRes, gridRes);
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            int idx = i*gridRes + j;

            // scene data
            float inDomain = queries.insideDomain(samplePts[idx].pt, true) ? 1 : 0;
            float distToAbsorbingBoundary = samplePts[idx].distToAbsorbingBoundary;
            float distToReflectingBoundary = samplePts[idx].distToReflectingBoundary;
            boundaryDistance->get(j, i) = Array3(distToAbsorbingBoundary, distToReflectingBoundary, inDomain);

            float dirichletVal = pde.dirichlet(samplePts[idx].pt, false);
            float robinVal = pde.robin ? pde.robin(samplePts[idx].pt, false) : pde.neumann(samplePts[idx].pt, false);
            float sourceVal = pde.source(samplePts[idx].pt);
            boundaryData->get(j, i) = Array3(dirichletVal, robinVal, sourceVal);

            // solution data
            float value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedSolution(): 0.0f;
            bool maskOutValue = (!inDomain && !isDoubleSided) || std::min(std::abs(distToAbsorbingBoundary),
                                                                          std::abs(distToReflectingBoundary))
                                                                          < boundaryDistanceMask;
            solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
        }
    }

    // write to disk
    writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
                  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

template <typename EvaluationPointType>
void createEvaluationGrid(std::vector<EvaluationPointType>& evalPts,
                          const zombie::GeometricQueries<2>& queries,
                          const Vector2& bMin, const Vector2& bMax,
                          const int gridRes) {
    // create a grid of evaluation points
    Vector2 extent = bMax - bMin;
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            Vector2 pt((i/float(gridRes))*extent.x() + bMin.x(),
                       (j/float(gridRes))*extent.y() + bMin.y());
            float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
            float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

            evalPts.emplace_back(EvaluationPointType(pt, Vector2::Zero(),
                                                     zombie::SampleType::InDomain,
                                                     distToAbsorbingBoundary,
                                                     distToReflectingBoundary, 0.0f));
        }
    }
}

void saveEvaluationGrid(const std::vector<zombie::bvc::EvaluationPoint<float, 2>>& evalPts,
                        const zombie::PDE<float, 2>& pde,
                        const zombie::GeometricQueries<2>& queries,
                        const bool isDoubleSided, const json& config) {
    // read settings from config
    const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
    const int gridRes = getRequired<int>(config, "gridRes");
    const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0f);

    const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
    const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
    const std::string colormap = getOptional<std::string>(config, "colormap", "");
    const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0f);
    const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0f);

    // extract solution data
    std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
    std::shared_ptr<Image<3>> boundaryDistance = std::make_shared<Image<3>>(gridRes, gridRes);
    std::shared_ptr<Image<3>> boundaryData = std::make_shared<Image<3>>(gridRes, gridRes);
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            int idx = i*gridRes + j;

            // scene data
            float inDomain = queries.insideDomain(evalPts[idx].pt, true) ? 1 : 0;
            float distToAbsorbingBoundary = evalPts[idx].distToAbsorbingBoundary;
            float distToReflectingBoundary = evalPts[idx].distToReflectingBoundary;
            boundaryDistance->get(j, i) = Array3(distToAbsorbingBoundary, distToReflectingBoundary, inDomain);

            float dirichletVal = pde.dirichlet(evalPts[idx].pt, false);
            float robinVal = pde.robin ? pde.robin(evalPts[idx].pt, false) : pde.neumann(evalPts[idx].pt, false);
            float sourceVal = pde.source(evalPts[idx].pt);
            boundaryData->get(j, i) = Array3(dirichletVal, robinVal, sourceVal);

            // solution data
            float value = evalPts[idx].getEstimatedSolution();
            bool maskOutValue = (!inDomain && !isDoubleSided) || std::min(std::abs(distToAbsorbingBoundary),
                                                                          std::abs(distToReflectingBoundary))
                                                                          < boundaryDistanceMask;
            solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
        }
    }

    // write to disk
    writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
                  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

void saveEvaluationGrid(const std::vector<zombie::rws::EvaluationPoint<float, 2>>& evalPts,
                        int nAbsorbingBoundarySamples, int nAbsorbingBoundaryNormalAlignedSamples,
                        int nReflectingBoundarySamples, int nReflectingBoundaryNormalAlignedSamples,
                        int nSourceSamples, const zombie::PDE<float, 2>& pde,
                        const zombie::GeometricQueries<2>& queries,
                        const bool isDoubleSided, const json& config) {
    // read settings from config
    const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
    const int gridRes = getRequired<int>(config, "gridRes");
    const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0f);

    const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
    const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
    const std::string colormap = getOptional<std::string>(config, "colormap", "");
    const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0f);
    const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0f);

    // extract solution data
    std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
    std::shared_ptr<Image<3>> boundaryDistance = std::make_shared<Image<3>>(gridRes, gridRes);
    std::shared_ptr<Image<3>> boundaryData = std::make_shared<Image<3>>(gridRes, gridRes);
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            int idx = i*gridRes + j;

            // scene data
            float inDomain = queries.insideDomain(evalPts[idx].pt, true) ? 1 : 0;
            float distToAbsorbingBoundary = evalPts[idx].distToAbsorbingBoundary;
            float distToReflectingBoundary = evalPts[idx].distToReflectingBoundary;
            boundaryDistance->get(j, i) = Array3(distToAbsorbingBoundary, distToReflectingBoundary, inDomain);

            float dirichletVal = pde.dirichlet(evalPts[idx].pt, false);
            float robinVal = pde.robin ? pde.robin(evalPts[idx].pt, false) : pde.neumann(evalPts[idx].pt, false);
            float sourceVal = pde.source(evalPts[idx].pt);
            boundaryData->get(j, i) = Array3(dirichletVal, robinVal, sourceVal);

            // solution data
            float value = evalPts[idx].getEstimatedSolution(nAbsorbingBoundarySamples,
                                                            nAbsorbingBoundaryNormalAlignedSamples,
                                                            nReflectingBoundarySamples,
                                                            nReflectingBoundaryNormalAlignedSamples,
                                                            nSourceSamples, 0.0f);
            bool maskOutValue = (!inDomain && !isDoubleSided) || std::min(std::abs(distToAbsorbingBoundary),
                                                                          std::abs(distToReflectingBoundary))
                                                                          < boundaryDistanceMask;
            solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
        }
    }

    // write to disk
    writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
                  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

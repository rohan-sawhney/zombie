// This file contains helper functions for creating 2D grids and writing
// PDE solutions estimated on these grids to disk.

#pragma once

#include <filesystem>
#include <zombie/zombie.h>
#include "config.h"
#include "colormap.h"

void writeSolution(const std::string& filename,
                   std::shared_ptr<Image<3>> solution,
                   std::shared_ptr<Image<3>> boundaryDistance,
                   std::shared_ptr<Image<3>> boundaryData,
                   bool saveDebug, bool saveColormapped,
                   std::string colormap, float minVal, float maxVal)
{
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
                        bool solveDoubleSided, const int gridRes)
{
    // create a grid of sample points
    Vector2 extent = queries.domainMax - queries.domainMin;
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            Vector2 pt((i/float(gridRes))*extent.x() + queries.domainMin.x(),
                       (j/float(gridRes))*extent.y() + queries.domainMin.y());
            zombie::EstimationQuantity estimationQuantity = solveDoubleSided || queries.insideDomain(pt) ?
                                                            zombie::EstimationQuantity::Solution:
                                                            zombie::EstimationQuantity::None;
            float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
            float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

            samplePts.emplace_back(zombie::SamplePoint<float, 2>(pt, Vector2::Zero(),
                                                                 zombie::SampleType::InDomain,
                                                                 estimationQuantity, 1.0f,
                                                                 distToAbsorbingBoundary,
                                                                 distToReflectingBoundary));
        }
    }
}

void saveSolutionGrid(const std::vector<zombie::SamplePoint<float, 2>>& samplePts,
                      const zombie::GeometricQueries<2>& queries,
                      const zombie::PDE<float, 2>& pde,
                      const bool solveDoubleSided, const json& config)
{
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

            // model problem data
            bool inDomain = solveDoubleSided || queries.insideDomain(samplePts[idx].pt);
            float distToAbsorbingBoundary = samplePts[idx].distToAbsorbingBoundary;
            float distToReflectingBoundary = samplePts[idx].distToReflectingBoundary;
            boundaryDistance->get(j, i) = Array3(distToAbsorbingBoundary,
                                                 distToReflectingBoundary,
                                                 inDomain ? 1.0f : 0.0f);

            float dirichletVal = pde.dirichlet(samplePts[idx].pt, false);
            float robinVal = pde.robin(samplePts[idx].pt, false);
            float sourceVal = pde.source(samplePts[idx].pt);
            boundaryData->get(j, i) = Array3(dirichletVal, robinVal, sourceVal);

            // solution data
            float value = samplePts[idx].statistics.getEstimatedSolution();
            bool maskOutValue = !inDomain || std::min(std::abs(distToAbsorbingBoundary),
                                                      std::abs(distToReflectingBoundary)) < boundaryDistanceMask;
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
                          const int gridRes)
{
    // create a grid of evaluation points
    Vector2 extent = queries.domainMax - queries.domainMin;
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            Vector2 pt((i/float(gridRes))*extent.x() + queries.domainMin.x(),
                       (j/float(gridRes))*extent.y() + queries.domainMin.y());
            float distToAbsorbingBoundary = queries.computeDistToAbsorbingBoundary(pt, false);
            float distToReflectingBoundary = queries.computeDistToReflectingBoundary(pt, false);

            evalPts.emplace_back(EvaluationPointType(pt, Vector2::Zero(),
                                                     zombie::SampleType::InDomain,
                                                     distToAbsorbingBoundary,
                                                     distToReflectingBoundary));
        }
    }
}

void saveEvaluationGrid(const std::vector<zombie::bvc::EvaluationPoint<float, 2>>& evalPts,
                        const zombie::GeometricQueries<2>& queries,
                        const zombie::PDE<float, 2>& pde,
                        const bool solveDoubleSided, const json& config)
{
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

            // model problem data
            bool inDomain = solveDoubleSided || queries.insideDomain(evalPts[idx].pt);
            float distToAbsorbingBoundary = evalPts[idx].distToAbsorbingBoundary;
            float distToReflectingBoundary = evalPts[idx].distToReflectingBoundary;
            boundaryDistance->get(j, i) = Array3(distToAbsorbingBoundary,
                                                 distToReflectingBoundary,
                                                 inDomain ? 1.0f : 0.0f);

            float dirichletVal = pde.dirichlet(evalPts[idx].pt, false);
            float robinVal = pde.robin(evalPts[idx].pt, false);
            float sourceVal = pde.source(evalPts[idx].pt);
            boundaryData->get(j, i) = Array3(dirichletVal, robinVal, sourceVal);

            // solution data
            float value = evalPts[idx].getEstimatedSolution();
            bool maskOutValue = !inDomain || std::min(std::abs(distToAbsorbingBoundary),
                                                      std::abs(distToReflectingBoundary)) < boundaryDistanceMask;
            solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
        }
    }

    // write to disk
    writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
                  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

void saveEvaluationGrid(const std::vector<zombie::rws::EvaluationPoint<float, 2>>& evalPts,
                        const std::vector<int>& sampleCounts,
                        const zombie::GeometricQueries<2>& queries,
                        const zombie::PDE<float, 2>& pde,
                        const bool solveDoubleSided, const json& config)
{
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
    int absorbingBoundarySampleCount = sampleCounts[0];
    int absorbingBoundaryNormalAlignedSampleCount = sampleCounts[1];
    int reflectingBoundarySampleCount = sampleCounts[2];
    int reflectingBoundaryNormalAlignedSampleCount = sampleCounts[3];
    int domainSampleCount = sampleCounts[4];

    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            int idx = i*gridRes + j;

            // model problem data
            bool inDomain = solveDoubleSided || queries.insideDomain(evalPts[idx].pt);
            float distToAbsorbingBoundary = evalPts[idx].distToAbsorbingBoundary;
            float distToReflectingBoundary = evalPts[idx].distToReflectingBoundary;
            boundaryDistance->get(j, i) = Array3(distToAbsorbingBoundary,
                                                 distToReflectingBoundary,
                                                 inDomain ? 1.0f : 0.0f);

            float dirichletVal = pde.dirichlet(evalPts[idx].pt, false);
            float robinVal = pde.robin(evalPts[idx].pt, false);
            float sourceVal = pde.source(evalPts[idx].pt);
            boundaryData->get(j, i) = Array3(dirichletVal, robinVal, sourceVal);

            // solution data
            float value = evalPts[idx].getEstimatedSolution(absorbingBoundarySampleCount,
                                                            absorbingBoundaryNormalAlignedSampleCount,
                                                            reflectingBoundarySampleCount,
                                                            reflectingBoundaryNormalAlignedSampleCount,
                                                            domainSampleCount);
            bool maskOutValue = !inDomain || std::min(std::abs(distToAbsorbingBoundary),
                                                      std::abs(distToReflectingBoundary)) < boundaryDistanceMask;
            solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
        }
    }

    // write to disk
    writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
                  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

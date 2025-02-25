// This file contains helper functions for creating 2D grids and writing
// PDE solutions estimated on these grids to disk.

#pragma once

#include <filesystem>
#include "config.h"
#include "image.h"
#include "colormap.h"

struct DistanceInfo {
    // constructor
    DistanceInfo(): inValidSolveRegion(false),
                    distToAbsorbingBoundary(0.0f),
                    distToReflectingBoundary(0.0f) {}

    // members
    bool inValidSolveRegion;
    float distToAbsorbingBoundary;
    float distToReflectingBoundary;
};

void createGridPoints(const json& config,
                      std::pair<Vector2, Vector2> boundingBox,
                      std::vector<Vector2>& gridPoints)
{
    const int gridRes = getRequired<int>(config, "gridRes");
    Vector2 gridMin = boundingBox.first;
    Vector2 gridMax = boundingBox.second;
    Vector2 extent = gridMax - gridMin;
    gridPoints.resize(gridRes*gridRes);

    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            int index = i*gridRes + j;
            gridPoints[index][0] = (i/float(gridRes))*extent[0] + gridMin[0];
            gridPoints[index][1] = (j/float(gridRes))*extent[1] + gridMin[1];
        }
    }
}

void saveGridValues(const json& config, std::string directoryPath,
                    const std::vector<DistanceInfo>& distanceInfo,
                    const std::vector<float>& values)
{
    const int gridRes = getRequired<int>(config, "gridRes");
    const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0f);
    const std::string solutionFile = directoryPath + getOptional<std::string>(config, "solutionFile", "solution.pfm");
    const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
    const std::string colormap = getOptional<std::string>(config, "colormap", "");
    const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0f);
    const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0f);

    // fill grid values
    std::shared_ptr<Image<3>> gridValues = std::make_shared<Image<3>>(gridRes, gridRes);
    for (int i = 0; i < gridRes; i++) {
        for (int j = 0; j < gridRes; j++) {
            int index = i*gridRes + j;
            bool inValidSolveRegion = distanceInfo[index].inValidSolveRegion;
            float distToBoundary = std::min(distanceInfo[index].distToAbsorbingBoundary,
                                            distanceInfo[index].distToReflectingBoundary);

            bool maskOutValue = !inValidSolveRegion || distToBoundary < boundaryDistanceMask;
            gridValues->get(j, i) = Array3(maskOutValue ? 0.0f : values[index]);
        }
    }

    // write solution to disk as grayscale image
    std::filesystem::path path(solutionFile);
    std::filesystem::create_directories(path.parent_path());
    gridValues->write(solutionFile);

    // write solution to disk as colormapped image
    std::string basePath = (path.parent_path() / path.stem()).string();
    std::string ext = path.extension().string();

    if (saveColormapped) {
        getColormappedImage(gridValues, colormap, colormapMinVal, colormapMaxVal)->write(basePath + "_color" + ext);
    }
}

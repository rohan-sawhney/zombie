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
	std::filesystem::path path(filename);
	std::filesystem::create_directories(path.parent_path());

	solution->write(filename);

	std::string basePath = (path.parent_path() / path.stem()).string();
	std::string ext = path.extension();

	if (saveColormapped) {
		getColormappedImage(solution, colormap, minVal, maxVal)->write(basePath + "_color" + ext);
	}

	if (saveDebug) {
		boundaryDistance->write(basePath + "_geometry" + ext);
		boundaryData->write(basePath + "_pde" + ext);
	}
}

void createSolutionGrid(std::vector<zombie::SamplePoint<float, 2>>& samplePts,
						const zombie::GeometricQueries<2> &queries,
						const Vector2 &bMin, const Vector2 &bMax,
						const int gridRes) {
	Vector2 extent = bMax - bMin;
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			Vector2 pt((i / float(gridRes)) * extent.x() + bMin.x(),
					   (j / float(gridRes)) * extent.y() + bMin.y());
			float dDist = queries.computeDistToDirichlet(pt, false);
			float nDist = queries.computeDistToNeumann(pt, false);
			samplePts.emplace_back(zombie::SamplePoint<float, 2>(pt, Vector2::Zero(),
																 zombie::SampleType::InDomain,
																 1.0f, dDist, nDist, 0.0f));
		}
	}
}

void saveSolutionGrid(const std::vector<zombie::SamplePoint<float, 2>>& samplePts,
					  const zombie::PDE<float, 2>& pde,
					  const zombie::GeometricQueries<2> &queries,
					  const bool isDoubleSided, const json &config) {
	const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
	const int gridRes = getRequired<int>(config, "gridRes");
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
	const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
	const std::string colormap = getOptional<std::string>(config, "colormap", "");
	const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0);
	const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0);

	std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> boundaryDistance = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> boundaryData = std::make_shared<Image<3>>(gridRes, gridRes);
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			int idx = i * gridRes + j;

			// debug / scene data
			float inDomain  = queries.insideDomain(samplePts[idx].pt) ? 1 : 0;
			float dirichletDist = samplePts[idx].dirichletDist;
			float neumannDist = samplePts[idx].neumannDist;
			boundaryDistance->get(j, i) = Array3(dirichletDist, neumannDist, inDomain);

			float dirichletVal = pde.dirichlet(samplePts[idx].pt);
			float neumannVal = pde.neumann(samplePts[idx].pt);
			float sourceVal = pde.source(samplePts[idx].pt);
			boundaryData->get(j, i) = Array3(dirichletVal, neumannVal, sourceVal);

			// solution data
			float value = samplePts[idx].statistics ? samplePts[idx].statistics->getEstimatedSolution(): 0.0f;
			bool maskOutValue = (!inDomain && !isDoubleSided) ||
								std::min(std::abs(dirichletDist), std::abs(neumannDist)) < boundaryDistanceMask;
			solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
		}
	}

	writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
				  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

void createEvaluationGrid(std::vector<zombie::EvaluationPoint<float, 2>>& evalPts,
						  const zombie::GeometricQueries<2> &queries,
						  const Vector2 &bMin, const Vector2 &bMax,
						  const int gridRes) {
	Vector2 extent = bMax - bMin;
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			Vector2 pt((i / float(gridRes)) * extent.x() + bMin.x(),
					   (j / float(gridRes)) * extent.y() + bMin.y());
			float dDist = queries.computeDistToDirichlet(pt, false);
			float nDist = queries.computeDistToNeumann(pt, false);
			evalPts.emplace_back(zombie::EvaluationPoint<float, 2>(pt, Vector2::Zero(),
																   zombie::SampleType::InDomain,
																   dDist, nDist, 0.0f));
		}
	}
}

void saveEvaluationGrid(const std::vector<zombie::EvaluationPoint<float, 2>>& evalPts,
						const zombie::PDE<float, 2>& pde,
						const zombie::GeometricQueries<2> &queries,
						const bool isDoubleSided, const json &config) {
	const std::string solutionFile = getOptional<std::string>(config, "solutionFile", "solution.pfm");
	const int gridRes = getRequired<int>(config, "gridRes");
	const float boundaryDistanceMask = getOptional<float>(config, "boundaryDistanceMask", 0.0);

	const bool saveDebug = getOptional<bool>(config, "saveDebug", false);
	const bool saveColormapped = getOptional<bool>(config, "saveColormapped", true);
	const std::string colormap = getOptional<std::string>(config, "colormap", "");
	const float colormapMinVal = getOptional<float>(config, "colormapMinVal", 0.0);
	const float colormapMaxVal = getOptional<float>(config, "colormapMaxVal", 1.0);

	std::shared_ptr<Image<3>> solution = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> boundaryDistance = std::make_shared<Image<3>>(gridRes, gridRes);
	std::shared_ptr<Image<3>> boundaryData = std::make_shared<Image<3>>(gridRes, gridRes);
	for (int i = 0; i < gridRes; i++) {
		for (int j = 0; j < gridRes; j++) {
			int idx = i * gridRes + j;

			// debug / scene data
			float inDomain  = queries.insideDomain(evalPts[idx].pt) ? 1 : 0;
			float dirichletDist = evalPts[idx].dirichletDist;
			float neumannDist = evalPts[idx].neumannDist;
			boundaryDistance->get(j, i) = Array3(dirichletDist, neumannDist, inDomain);

			float dirichletVal = pde.dirichlet(evalPts[idx].pt);
			float neumannVal = pde.neumann(evalPts[idx].pt);
			float sourceVal = pde.source(evalPts[idx].pt);
			boundaryData->get(j, i) = Array3(dirichletVal, neumannVal, sourceVal);

			// solution data
			float value = evalPts[idx].getEstimatedSolution();
			bool maskOutValue = (!inDomain && !isDoubleSided) ||
								std::min(std::abs(dirichletDist), std::abs(neumannDist)) < boundaryDistanceMask;
			solution->get(j, i) = Array3(maskOutValue ? 0.0f : value);
		}
	}

	writeSolution(solutionFile, solution, boundaryDistance, boundaryData, saveDebug,
				  saveColormapped, colormap, colormapMinVal, colormapMaxVal);
}

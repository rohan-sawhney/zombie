'''
This file is the entry point for a 2D demo application demonstrating how to use Zombie.
It reads a 'model problem' description from a JSON file, runs the WalkOnStars, BoundaryValueCaching
or ReverseWalkonOnStars solvers, and saves the result as a PNG file.

The full Zombie API can be viewed using the following commands in the Python console:
>>> import zombie
>>> help(zombie)
'''

import argparse
import json
import os
import numpy as np
import zombie
import matplotlib.pyplot as plt
from typing import Callable, Optional
from PIL import Image

##############################################################################################
# Image I/O

def load_image_buffer(image_file):
    # PIL does not support pfm files, so default to png
    if image_file.endswith(".pfm"):
        image_file = image_file.replace(".pfm", ".png")

    # load image and convert to grayscale
    image = Image.open(image_file)
    image = image.transpose(Image.Transpose.TRANSPOSE)
    image_shape = np.array([image.height, image.width], dtype=np.int32)
    image = image.convert("L")
    image = np.array(image).flatten().astype(np.float32)/255.0

    return image, image_shape

def save_image_buffer(image_file, image_buffer, save_colormapped, colormap,
                      colormap_min_val, colormap_max_val):
    # PIL does not support pfm files, so default to png
    if image_file.endswith(".pfm"):
        image_file = image_file.replace(".pfm", ".png")

    # save grayscale image
    image = np.clip(image_buffer, 0.0, 1.0)
    grayscale_image = (image*255.0).astype(np.uint8)
    grayscale_image = Image.fromarray(grayscale_image)
    grayscale_image.save(image_file)

    # save colormapped image
    if save_colormapped:
        image = np.clip((image_buffer - colormap_min_val)/(colormap_max_val - colormap_min_val), 0.0, 1.0)
        cmap = plt.get_cmap(colormap)
        colormapped_image = cmap(image, bytes=True)
        colormapped_image = np.clip(colormapped_image[:, :, :3].astype(np.uint8), 0, 255)
        colormapped_image = Image.fromarray(colormapped_image)
        base, ext = os.path.splitext(image_file)
        colormap_file = f"{base}_color{ext}"
        colormapped_image.save(colormap_file)

##############################################################################################
# Problem specification - geometry loading and PDE setup

def load_boundary_mesh(model_problem_config, normalize=True, flip_orientation=True):
    # load the model problem configuration
    obj_file = model_problem_config["geometry"]

    # load obj file, and optionally normalize and flip mesh orientation
    positions = zombie.float2_list()
    indices = zombie.int2_list()
    zombie.utils.load_boundary_mesh_2d(obj_file, positions, indices)

    if normalize:
        zombie.utils.normalize_2d(positions)

    if flip_orientation:
        zombie.utils.flip_orientation_2d(indices)

    bounding_box = zombie.utils.compute_bounding_box_2d(positions, True, 1.0)

    return positions, indices, bounding_box

def setup_pde(model_problem_config, bounding_box):
    # load the model problem configuration
    source_value_buffer, source_value_shape =\
        load_image_buffer(model_problem_config["sourceValue"])
    absorbing_boundary_value_buffer, absorbing_boundary_value_shape =\
        load_image_buffer(model_problem_config["absorbingBoundaryValue"])
    reflecting_boundary_value_buffer, reflecting_boundary_value_shape =\
        load_image_buffer(model_problem_config["reflectingBoundaryValue"])
    is_reflecting_boundary_buffer, is_reflecting_boundary_shape =\
        load_image_buffer(model_problem_config["isReflectingBoundary"])
    robin_coeff = model_problem_config["robinCoeff"]\
        if "robinCoeff" in model_problem_config else 0.0
    absorption_coeff = model_problem_config["absorptionCoeff"]\
        if "absorptionCoeff" in model_problem_config else 0.0
    domain_min = bounding_box[0]
    domain_max = bounding_box[1]

    # setup the PDE
    pde = zombie.core.pde_float_2d()
    pde.source = zombie.utils.get_dense_grid_source_callback_float_2d(
        source_value_buffer, source_value_shape, domain_min, domain_max)
    pde.dirichlet = zombie.utils.get_dense_grid_boundary_condition_callback_float_2d(
        absorbing_boundary_value_buffer, absorbing_boundary_value_shape, domain_min, domain_max)
    pde.robin = zombie.utils.get_dense_grid_boundary_condition_callback_float_2d(
        reflecting_boundary_value_buffer, reflecting_boundary_value_shape, domain_min, domain_max)
    pde.has_reflecting_boundary_conditions = zombie.utils.get_dense_grid_indicator_callback_2d(
        is_reflecting_boundary_buffer, is_reflecting_boundary_shape, domain_min, domain_max)
    pde.robin_coeff = zombie.core.get_constant_robin_coefficient_callback_2d(robin_coeff)
    pde.absorption_coeff = absorption_coeff
    pde.are_robin_conditions_pure_neumann = robin_coeff == 0.0

    return pde

def partition_boundary_mesh(positions, indices, on_reflecting_boundary):
    # use Zombie's default partitioning function, which assumes the boundary discretization
    # is perfectly adapted to the boundary conditions; this isn't always a correct assumption
    # and the user might want to override this function for their specific problem
    absorbing_boundary_positions = zombie.float2_list()
    absorbing_boundary_indices = zombie.int2_list()
    reflecting_boundary_positions = zombie.float2_list()
    reflecting_boundary_indices = zombie.int2_list()
    zombie.utils.partition_boundary_mesh_2d(on_reflecting_boundary, positions, indices,
                                            absorbing_boundary_positions, absorbing_boundary_indices,
                                            reflecting_boundary_positions, reflecting_boundary_indices)

    return absorbing_boundary_positions, absorbing_boundary_indices,\
           reflecting_boundary_positions, reflecting_boundary_indices

def populate_geometric_queries(model_problem_config, absorbing_boundary_positions,
                               absorbing_boundary_indices, reflecting_boundary_positions,
                               reflecting_boundary_indices, bounding_box):
    # load the model problem configuration
    domain_is_watertight = model_problem_config["domainIsWatertight"]\
        if "domainIsWatertight" in model_problem_config else True
    solve_double_sided = model_problem_config["solveDoubleSided"]\
        if "solveDoubleSided" in model_problem_config else False
    robin_coeff = model_problem_config["robinCoeff"]\
        if "robinCoeff" in model_problem_config else 0.0

    # create the geometric queries object
    domain_min = bounding_box[0]
    domain_max = bounding_box[1]
    geometric_queries = zombie.core.geometric_queries_2d(domain_is_watertight, domain_min, domain_max)

    # use an absorbing boundary handler to populate geometric queries for the absorbing boundary
    dirichlet_boundary_handler = zombie.utils.fcpw_dirichlet_boundary_handler_2d()
    dirichlet_boundary_handler.build_acceleration_structure(absorbing_boundary_positions,
                                                            absorbing_boundary_indices)
    zombie.utils.populate_geometric_queries_for_dirichlet_boundary_2d(dirichlet_boundary_handler,
                                                                      geometric_queries)

    # use a reflecting boundary handler to populate geometric queries for the reflecting boundary
    ignore_candidate_silhouette = zombie.utils.get_ignore_candidate_silhouette_callback(solve_double_sided)
    branch_traversal_weight = zombie.utils.get_branch_traversal_weight_callback()

    if robin_coeff > 0.0:
        # despite using a constant Robin coefficient here, the implementation supports
        # varying coefficients over the boundary
        min_robin_coeff_values = zombie.float_list([robin_coeff]*len(reflecting_boundary_indices))
        max_robin_coeff_values = zombie.float_list([robin_coeff]*len(reflecting_boundary_indices))
        robin_boundary_handler = zombie.utils.fcpw_robin_boundary_handler_2d()
        robin_boundary_handler.build_acceleration_structure(reflecting_boundary_positions,
                                                            reflecting_boundary_indices,
                                                            ignore_candidate_silhouette,
                                                            min_robin_coeff_values,
                                                            max_robin_coeff_values)
        zombie.utils.populate_geometric_queries_for_robin_boundary_2d(robin_boundary_handler,
                                                                      branch_traversal_weight,
                                                                      geometric_queries)

        return geometric_queries, dirichlet_boundary_handler, robin_boundary_handler

    else:
        neumann_boundary_handler = zombie.utils.fcpw_neumann_boundary_handler_2d()
        neumann_boundary_handler.build_acceleration_structure(reflecting_boundary_positions,
                                                              reflecting_boundary_indices,
                                                              ignore_candidate_silhouette)
        zombie.utils.populate_geometric_queries_for_neumann_boundary_2d(neumann_boundary_handler,
                                                                        branch_traversal_weight,
                                                                        geometric_queries)

        return geometric_queries, dirichlet_boundary_handler, neumann_boundary_handler

##############################################################################################
# Walk on Stars solver - note that this solver is a strict generalization of Walk on Spheres,
# and reduces to it when the PDE only has Dirichlet boundary conditions

def create_sample_point_grid(grid_res, geometric_queries, solve_double_sided):
    sample_pts = [None]*(grid_res*grid_res)
    extent = geometric_queries.domain_max - geometric_queries.domain_min

    for i in range(grid_res):
        for j in range(grid_res):
            pt = np.zeros(2)
            pt[0] = (i/float(grid_res))*extent[0] + geometric_queries.domain_min[0]
            pt[1] = (j/float(grid_res))*extent[1] + geometric_queries.domain_min[1]
            normal = np.zeros(2)
            sample_type = zombie.solvers.sample_type.in_domain
            estimation_quantity = zombie.solvers.estimation_quantity.solution\
                                    if solve_double_sided or geometric_queries.inside_domain(pt)\
                                        else zombie.solvers.estimation_quantity.none
            pdf = 1.0
            dist_to_absorbing_boundary = geometric_queries.compute_dist_to_absorbing_boundary(pt, False)
            dist_to_reflecting_boundary = geometric_queries.compute_dist_to_reflecting_boundary(pt, False)

            sample_pts[i*grid_res + j] = zombie.solvers.sample_point_float_2d(pt, normal, sample_type,
                                                                              estimation_quantity, pdf,
                                                                              dist_to_absorbing_boundary,
                                                                              dist_to_reflecting_boundary)

    return zombie.solvers.sample_point_float_2d_list(sample_pts)

def run_walk_on_stars(solver_config, sample_pts, geometric_queries, pde, solve_double_sided):
    # load config settings
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.0

    n_walks = solver_config["nWalks"]\
        if "nWalks" in solver_config else 128
    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 1024
    steps_before_applying_tikhonov = solver_config["stepsBeforeApplyingTikhonov"]\
        if "stepsBeforeApplyingTikhonov" in solver_config else 0
    steps_before_using_maximal_spheres = solver_config["stepsBeforeUsingMaximalSpheres"]\
        if "stepsBeforeUsingMaximalSpheres" in solver_config else max_walk_length

    disable_gradient_control_variates = solver_config["disableGradientControlVariates"]\
        if "disableGradientControlVariates" in solver_config else False
    disable_gradient_antithetic_variates = solver_config["disableGradientAntitheticVariates"]\
        if "disableGradientAntitheticVariates" in solver_config else False
    use_cosine_sampling_for_directional_derivatives = solver_config["useCosineSamplingForDirectionalDerivatives"]\
        if "useCosineSamplingForDirectionalDerivatives" in solver_config else False
    ignore_absorbing_boundary_contribution = solver_config["ignoreAbsorbingBoundaryContribution"]\
        if "ignoreAbsorbingBoundaryContribution" in solver_config else False
    ignore_reflecting_boundary_contribution = solver_config["ignoreReflectingBoundaryContribution"]\
        if "ignoreReflectingBoundaryContribution" in solver_config else False
    ignore_source_contribution = solver_config["ignoreSourceContribution"]\
        if "ignoreSourceContribution" in solver_config else False
    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    run_single_threaded = solver_config["runSingleThreaded"]\
        if "runSingleThreaded" in solver_config else False

    # initialize solver and estimate solution
    progress_bar = zombie.utils.progress_bar(len(sample_pts))
    report_progress = zombie.utils.get_report_progress_callback(progress_bar)

    walk_settings = zombie.solvers.walk_settings(epsilon_shell_for_absorbing_boundary,
                                                 epsilon_shell_for_reflecting_boundary,
                                                 silhouette_precision, russian_roulette_threshold,
                                                 max_walk_length, steps_before_applying_tikhonov,
                                                 steps_before_using_maximal_spheres, solve_double_sided,
                                                 not disable_gradient_control_variates,
                                                 not disable_gradient_antithetic_variates,
                                                 use_cosine_sampling_for_directional_derivatives,
                                                 ignore_absorbing_boundary_contribution,
                                                 ignore_reflecting_boundary_contribution,
                                                 ignore_source_contribution, print_logs)
    n_walks_list = zombie.int_list([n_walks]*len(sample_pts))
    walk_on_stars = zombie.solvers.walk_on_stars_float_2d(geometric_queries)
    walk_on_stars.solve(pde, walk_settings, n_walks_list, sample_pts, run_single_threaded, report_progress)
    progress_bar.finish()

def get_solution_from_sample_point_grid(sample_pts, geometric_queries, solve_double_sided,
                                        grid_res, boundary_dist_mask):
    solution = np.zeros((grid_res, grid_res))

    for i in range(grid_res):
        for j in range(grid_res):
            sample_pt = sample_pts[i*grid_res + j]
            in_domain = solve_double_sided or geometric_queries.inside_domain(sample_pt.pt)
            dist_to_boundary = min(abs(sample_pt.dist_to_absorbing_boundary),
                                   abs(sample_pt.dist_to_reflecting_boundary))

            if in_domain and dist_to_boundary > boundary_dist_mask:
                solution[j][i] = sample_pt.statistics.get_estimated_solution()

    return solution

##############################################################################################
# Boundary Value Caching solver

def create_domain_sampler(geometric_queries, solve_double_sided):
    solve_region_min = geometric_queries.domain_min
    solve_region_max = geometric_queries.domain_max

    if solve_double_sided:
        solve_region_volume = np.prod(solve_region_max - solve_region_min)
        return zombie.samplers.create_uniform_domain_sampler_float_2d(
                geometric_queries, geometric_queries.inside_bounding_domain,
                solve_region_min, solve_region_max, solve_region_volume)

    else:
        solve_region_volume = np.abs(geometric_queries.compute_domain_signed_volume())
        return zombie.samplers.create_uniform_domain_sampler_float_2d(
                geometric_queries, geometric_queries.inside_domain,
                solve_region_min, solve_region_max, solve_region_volume)

def create_boundary_sampler(positions, indices, geometric_queries,
                            normal_offset_for_boundary, solve_double_sided):
    boundary_sampler = zombie.samplers.create_uniform_line_segment_boundary_sampler_float_2d(
                        positions, indices, geometric_queries, geometric_queries.inside_bounding_domain)
    boundary_sampler.initialize(normal_offset_for_boundary, solve_double_sided)

    return boundary_sampler

def create_bvc_evaluation_point_grid(grid_res, geometric_queries, solve_double_sided):
    evaluation_pts = [None]*(grid_res*grid_res)
    extent = geometric_queries.domain_max - geometric_queries.domain_min

    for i in range(grid_res):
        for j in range(grid_res):
            pt = np.zeros(2)
            pt[0] = (i/float(grid_res))*extent[0] + geometric_queries.domain_min[0]
            pt[1] = (j/float(grid_res))*extent[1] + geometric_queries.domain_min[1]
            normal = np.zeros(2)
            sample_type = zombie.solvers.sample_type.in_domain
            dist_to_absorbing_boundary = geometric_queries.compute_dist_to_absorbing_boundary(pt, False)
            dist_to_reflecting_boundary = geometric_queries.compute_dist_to_reflecting_boundary(pt, False)

            evaluation_pts[i*grid_res + j] = zombie.solvers.bvc_evaluation_point_float_2d(pt, normal, sample_type,
                                                                                          dist_to_absorbing_boundary,
                                                                                          dist_to_reflecting_boundary)

    return zombie.solvers.bvc_evaluation_point_float_2d_list(evaluation_pts)

def run_boundary_value_caching(solver_config, evaluation_pts,
                               absorbing_boundary_positions, absorbing_boundary_indices,
                               reflecting_boundary_positions, reflecting_boundary_indices,
                               geometric_queries, pde, solve_double_sided):
    # load config settings for walk on stars
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.0

    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 1024
    steps_before_applying_tikhonov = solver_config["stepsBeforeApplyingTikhonov"]\
        if "stepsBeforeApplyingTikhonov" in solver_config else 0
    steps_before_using_maximal_spheres = solver_config["stepsBeforeUsingMaximalSpheres"]\
        if "stepsBeforeUsingMaximalSpheres" in solver_config else max_walk_length

    disable_gradient_control_variates = solver_config["disableGradientControlVariates"]\
        if "disableGradientControlVariates" in solver_config else False
    disable_gradient_antithetic_variates = solver_config["disableGradientAntitheticVariates"]\
        if "disableGradientAntitheticVariates" in solver_config else False
    use_cosine_sampling_for_directional_derivatives = solver_config["useCosineSamplingForDirectionalDerivatives"]\
        if "useCosineSamplingForDirectionalDerivatives" in solver_config else False
    ignore_absorbing_boundary_contribution = solver_config["ignoreAbsorbingBoundaryContribution"]\
        if "ignoreAbsorbingBoundaryContribution" in solver_config else False
    ignore_reflecting_boundary_contribution = solver_config["ignoreReflectingBoundaryContribution"]\
        if "ignoreReflectingBoundaryContribution" in solver_config else False
    ignore_source_contribution = solver_config["ignoreSourceContribution"]\
        if "ignoreSourceContribution" in solver_config else False
    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    run_single_threaded = solver_config["runSingleThreaded"]\
        if "runSingleThreaded" in solver_config else False

    # load config settings for boundary value caching
    n_walks_for_cached_solution_estimates = solver_config["nWalksForCachedSolutionEstimates"]\
        if "nWalksForCachedSolutionEstimates" in solver_config else 128
    n_walks_for_cached_gradient_estimates = solver_config["nWalksForCachedGradientEstimates"]\
        if "nWalksForCachedGradientEstimates" in solver_config else 640
    absorbing_boundary_cache_size = solver_config["absorbingBoundaryCacheSize"]\
        if "absorbingBoundaryCacheSize" in solver_config else 1024
    reflecting_boundary_cache_size = solver_config["reflectingBoundaryCacheSize"]\
        if "reflectingBoundaryCacheSize" in solver_config else 1024
    domain_cache_size = solver_config["domainCacheSize"]\
        if "domainCacheSize" in solver_config else 1024

    use_finite_differences_for_boundary_derivatives = solver_config["useFiniteDifferencesForBoundaryDerivatives"]\
        if "useFiniteDifferencesForBoundaryDerivatives" in solver_config else False

    robin_coeff_cutoff_for_normal_derivative = solver_config["robinCoeffCutoffForNormalDerivative"]\
        if "robinCoeffCutoffForNormalDerivative" in solver_config else np.inf
    normal_offset_for_absorbing_boundary = solver_config["normalOffsetForAbsorbingBoundary"]\
        if "normalOffsetForAbsorbingBoundary" in solver_config else 5.0*epsilon_shell_for_absorbing_boundary
    normal_offset_for_reflecting_boundary = solver_config["normalOffsetForReflectingBoundary"]\
        if "normalOffsetForReflectingBoundary" in solver_config else 0.0
    radius_clamp_for_kernels = solver_config["radiusClampForKernels"]\
        if "radiusClampForKernels" in solver_config else 0.0
    regularization_for_kernels = solver_config["regularizationForKernels"]\
        if "regularizationForKernels" in solver_config else 0.0

    # initialize boundary samplers
    absorbing_boundary_sampler = create_boundary_sampler(
        absorbing_boundary_positions, absorbing_boundary_indices, geometric_queries,
        normal_offset_for_absorbing_boundary, solve_double_sided)
    reflecting_boundary_sampler = create_boundary_sampler(
        reflecting_boundary_positions, reflecting_boundary_indices, geometric_queries,
        normal_offset_for_reflecting_boundary, solve_double_sided)

    # initialize domain sampler
    domain_sampler = create_domain_sampler(geometric_queries, solve_double_sided)
    if ignore_source_contribution:
        domain_cache_size = 0

    # solve using boundary value caching
    total_work = 2*(absorbing_boundary_cache_size + reflecting_boundary_cache_size) + domain_cache_size
    progress_bar = zombie.utils.progress_bar(total_work)
    report_progress = zombie.utils.get_report_progress_callback(progress_bar)

    boundary_value_caching = zombie.solvers.boundary_value_caching_float_2d(
        geometric_queries, absorbing_boundary_sampler, reflecting_boundary_sampler, domain_sampler)

    # generate boundary and domain samples
    boundary_value_caching.generate_samples(absorbing_boundary_cache_size, reflecting_boundary_cache_size,
                                            domain_cache_size, normal_offset_for_absorbing_boundary,
                                            normal_offset_for_reflecting_boundary, solve_double_sided)

    # compute sample estimates
    walk_settings = zombie.solvers.walk_settings(epsilon_shell_for_absorbing_boundary,
                                                 epsilon_shell_for_reflecting_boundary,
                                                 silhouette_precision, russian_roulette_threshold,
                                                 max_walk_length, steps_before_applying_tikhonov,
                                                 steps_before_using_maximal_spheres, solve_double_sided,
                                                 not disable_gradient_control_variates,
                                                 not disable_gradient_antithetic_variates,
                                                 use_cosine_sampling_for_directional_derivatives,
                                                 ignore_absorbing_boundary_contribution,
                                                 ignore_reflecting_boundary_contribution,
                                                 ignore_source_contribution, print_logs)
    boundary_value_caching.compute_sample_estimates(pde, walk_settings,
                                                    n_walks_for_cached_solution_estimates,
                                                    n_walks_for_cached_gradient_estimates,
                                                    robin_coeff_cutoff_for_normal_derivative,
                                                    use_finite_differences_for_boundary_derivatives,
                                                    run_single_threaded, report_progress)

    # splat boundary sample estimates and domain data to evaluation points
    boundary_value_caching.splat(pde, radius_clamp_for_kernels,
                                 regularization_for_kernels,
                                 robin_coeff_cutoff_for_normal_derivative,
                                 normal_offset_for_absorbing_boundary,
                                 normal_offset_for_reflecting_boundary,
                                 evaluation_pts, report_progress)

    # estimate solution near boundary
    boundary_value_caching.estimate_solution_near_boundary(pde, walk_settings,
                                                           normal_offset_for_absorbing_boundary,
                                                           normal_offset_for_reflecting_boundary,
                                                           n_walks_for_cached_solution_estimates,
                                                           evaluation_pts, run_single_threaded)
    progress_bar.finish()

def get_solution_from_bvc_evaluation_point_grid(evaluation_pts, geometric_queries, solve_double_sided,
                                                grid_res, boundary_dist_mask):
    solution = np.zeros((grid_res, grid_res))

    for i in range(grid_res):
        for j in range(grid_res):
            eval_pt = evaluation_pts[i*grid_res + j]
            in_domain = solve_double_sided or geometric_queries.inside_domain(eval_pt.pt)
            dist_to_boundary = min(abs(eval_pt.dist_to_absorbing_boundary),
                                   abs(eval_pt.dist_to_reflecting_boundary))

            if in_domain and dist_to_boundary > boundary_dist_mask:
                solution[j][i] = eval_pt.get_estimated_solution()

    return solution

##############################################################################################
# Reverse Walk on Stars solver

def create_rws_evaluation_point_grid(grid_res, geometric_queries, solve_double_sided):
    evaluation_pts = [None]*(grid_res*grid_res)
    extent = geometric_queries.domain_max - geometric_queries.domain_min

    for i in range(grid_res):
        for j in range(grid_res):
            pt = np.zeros(2)
            pt[0] = (i/float(grid_res))*extent[0] + geometric_queries.domain_min[0]
            pt[1] = (j/float(grid_res))*extent[1] + geometric_queries.domain_min[1]
            normal = np.zeros(2)
            sample_type = zombie.solvers.sample_type.in_domain
            dist_to_absorbing_boundary = geometric_queries.compute_dist_to_absorbing_boundary(pt, False)
            dist_to_reflecting_boundary = geometric_queries.compute_dist_to_reflecting_boundary(pt, False)

            evaluation_pts[i*grid_res + j] = zombie.solvers.rws_evaluation_point_float_2d(pt, normal, sample_type,
                                                                                          dist_to_absorbing_boundary,
                                                                                          dist_to_reflecting_boundary)

    return zombie.solvers.rws_evaluation_point_float_2d_list(evaluation_pts)

def run_reverse_walk_on_stars(solver_config, evaluation_pts,
                              absorbing_boundary_positions, absorbing_boundary_indices,
                              reflecting_boundary_positions, reflecting_boundary_indices,
                              geometric_queries, pde, solve_double_sided):
    # load config settings for reverse walk on stars
    epsilon_shell_for_absorbing_boundary = solver_config["epsilonShellForAbsorbingBoundary"]\
        if "epsilonShellForAbsorbingBoundary" in solver_config else 1e-3
    epsilon_shell_for_reflecting_boundary = solver_config["epsilonShellForReflectingBoundary"]\
        if "epsilonShellForReflectingBoundary" in solver_config else 1e-3
    silhouette_precision = solver_config["silhouettePrecision"]\
        if "silhouettePrecision" in solver_config else 1e-3
    russian_roulette_threshold = solver_config["russianRouletteThreshold"]\
        if "russianRouletteThreshold" in solver_config else 0.0

    max_walk_length = solver_config["maxWalkLength"]\
        if "maxWalkLength" in solver_config else 1024
    steps_before_applying_tikhonov = solver_config["stepsBeforeApplyingTikhonov"]\
        if "stepsBeforeApplyingTikhonov" in solver_config else 0
    steps_before_using_maximal_spheres = solver_config["stepsBeforeUsingMaximalSpheres"]\
        if "stepsBeforeUsingMaximalSpheres" in solver_config else max_walk_length

    ignore_absorbing_boundary_contribution = solver_config["ignoreAbsorbingBoundaryContribution"]\
        if "ignoreAbsorbingBoundaryContribution" in solver_config else False
    ignore_reflecting_boundary_contribution = solver_config["ignoreReflectingBoundaryContribution"]\
        if "ignoreReflectingBoundaryContribution" in solver_config else False
    ignore_source_contribution = solver_config["ignoreSourceContribution"]\
        if "ignoreSourceContribution" in solver_config else False
    print_logs = solver_config["printLogs"]\
        if "printLogs" in solver_config else False
    run_single_threaded = solver_config["runSingleThreaded"]\
        if "runSingleThreaded" in solver_config else False

    # load config settings for reverse walk splatting
    absorbing_boundary_sample_count = solver_config["absorbingBoundarySampleCount"]\
        if "absorbingBoundarySampleCount" in solver_config else 1024
    reflecting_boundary_sample_count = solver_config["reflectingBoundarySampleCount"]\
        if "reflectingBoundarySampleCount" in solver_config else 1024
    domain_sample_count = solver_config["domainSampleCount"]\
        if "domainSampleCount" in solver_config else 1024

    normal_offset_for_absorbing_boundary = solver_config["normalOffsetForAbsorbingBoundary"]\
        if "normalOffsetForAbsorbingBoundary" in solver_config else 5.0*epsilon_shell_for_absorbing_boundary
    radius_clamp_for_kernels = solver_config["radiusClampForKernels"]\
        if "radiusClampForKernels" in solver_config else 0.0
    regularization_for_kernels = solver_config["regularizationForKernels"]\
        if "regularizationForKernels" in solver_config else 0.0

    # initialize boundary samplers
    absorbing_boundary_sampler = create_boundary_sampler(
        absorbing_boundary_positions, absorbing_boundary_indices, geometric_queries,
        normal_offset_for_absorbing_boundary, solve_double_sided)
    if ignore_absorbing_boundary_contribution:
        absorbing_boundary_sample_count = 0

    reflecting_boundary_sampler = create_boundary_sampler(
        reflecting_boundary_positions, reflecting_boundary_indices, geometric_queries,
        0.0, solve_double_sided)
    if ignore_reflecting_boundary_contribution:
        reflecting_boundary_sample_count = 0

    # initialize domain sampler
    domain_sampler = create_domain_sampler(geometric_queries, solve_double_sided)
    if ignore_source_contribution:
        domain_sample_count = 0

    # solve using reverse walk on stars
    total_work = absorbing_boundary_sample_count + reflecting_boundary_sample_count + domain_sample_count
    progress_bar = zombie.utils.progress_bar(total_work)
    report_progress = zombie.utils.get_report_progress_callback(progress_bar)

    reverse_walk_on_stars = zombie.solvers.reverse_walk_on_stars_float_2d(absorbing_boundary_sampler,
                                                                          reflecting_boundary_sampler,
                                                                          domain_sampler)

    # generate boundary and domain samples
    reverse_walk_on_stars.generate_samples(absorbing_boundary_sample_count,
                                           reflecting_boundary_sample_count,
                                           domain_sample_count,
                                           normal_offset_for_absorbing_boundary,
                                           solve_double_sided)

    # splat contributions to evaluation points
    walk_settings = zombie.solvers.walk_settings(epsilon_shell_for_absorbing_boundary,
                                                 epsilon_shell_for_reflecting_boundary,
                                                 silhouette_precision, russian_roulette_threshold,
                                                 max_walk_length, steps_before_applying_tikhonov,
                                                 steps_before_using_maximal_spheres,
                                                 solve_double_sided, False, False, False,
                                                 ignore_absorbing_boundary_contribution,
                                                 ignore_reflecting_boundary_contribution,
                                                 ignore_source_contribution, print_logs)
    reverse_walk_on_stars.solve(pde, geometric_queries, walk_settings, normal_offset_for_absorbing_boundary,
                                radius_clamp_for_kernels, regularization_for_kernels, evaluation_pts,
                                True, run_single_threaded, report_progress)
    progress_bar.finish()

    return [reverse_walk_on_stars.get_absorbing_boundary_sample_count(False),
            reverse_walk_on_stars.get_absorbing_boundary_sample_count(True),
            reverse_walk_on_stars.get_reflecting_boundary_sample_count(False),
            reverse_walk_on_stars.get_reflecting_boundary_sample_count(True),
            reverse_walk_on_stars.get_domain_sample_count()]

def get_solution_from_rws_evaluation_point_grid(evaluation_pts, sample_counts,
                                                geometric_queries, solve_double_sided,
                                                grid_res, boundary_dist_mask):
    solution = np.zeros((grid_res, grid_res))
    absorbing_boundary_sample_count = sample_counts[0]
    absorbing_boundary_normal_aligned_sample_count = sample_counts[1]
    reflecting_boundary_sample_count = sample_counts[2]
    reflecting_boundary_normal_aligned_sample_count = sample_counts[3]
    domain_sample_count = sample_counts[4]

    for i in range(grid_res):
        for j in range(grid_res):
            eval_pt = evaluation_pts[i*grid_res + j]
            in_domain = solve_double_sided or geometric_queries.inside_domain(eval_pt.pt)
            dist_to_boundary = min(abs(eval_pt.dist_to_absorbing_boundary),
                                   abs(eval_pt.dist_to_reflecting_boundary))

            if in_domain and dist_to_boundary > boundary_dist_mask:
                solution[j][i] = eval_pt.get_estimated_solution(absorbing_boundary_sample_count,
                                                                absorbing_boundary_normal_aligned_sample_count,
                                                                reflecting_boundary_sample_count,
                                                                reflecting_boundary_normal_aligned_sample_count,
                                                                domain_sample_count)

    return solution

##############################################################################################
# Solver execution

def run_solver(model_problem_config, solver_type, solver_config, output_config,
               absorbing_boundary_positions, absorbing_boundary_indices,
               reflecting_boundary_positions, reflecting_boundary_indices,
               geometric_queries, pde):
    grid_res = output_config["gridRes"]
    boundary_distance_mask = output_config["boundaryDistanceMask"]\
        if "boundaryDistanceMask" in output_config else 0.0
    solution_file = output_config["solutionFile"]\
        if "solutionFile" in output_config else "solution.png"
    colormap = output_config["colormap"]\
        if "colormap" in output_config else "turbo"
    colormap_min_val = output_config["colormapMinVal"]\
        if "colormapMinVal" in output_config else 0.0
    colormap_max_val = output_config["colormapMaxVal"]\
        if "colormapMaxVal" in output_config else 1.0
    save_colormapped = output_config["saveColormapped"]\
        if "saveColormapped" in output_config else True
    solve_double_sided = model_problem_config["solveDoubleSided"]\
        if "solveDoubleSided" in model_problem_config else False

    if solver_type == "wost":
        # create sample points on grid to compute solution on
        sample_pts = create_sample_point_grid(grid_res, geometric_queries, solve_double_sided)

        # run walk on stars
        run_walk_on_stars(solver_config, sample_pts, geometric_queries, pde, solve_double_sided)

        # save solution to disk
        solution = get_solution_from_sample_point_grid(sample_pts, geometric_queries,
                                                       solve_double_sided, grid_res,
                                                       boundary_distance_mask)
        save_image_buffer(solution_file, solution, save_colormapped, colormap,
                          colormap_min_val, colormap_max_val)

    elif solver_type == "bvc":
        # create evaluation points on grid to compute solution on
        evaluation_pts = create_bvc_evaluation_point_grid(grid_res, geometric_queries,
                                                          solve_double_sided)

        # run boundary value caching
        run_boundary_value_caching(solver_config, evaluation_pts,
                                   absorbing_boundary_positions, absorbing_boundary_indices,
                                   reflecting_boundary_positions, reflecting_boundary_indices,
                                   geometric_queries, pde, solve_double_sided)

        # save solution to disk
        solution = get_solution_from_bvc_evaluation_point_grid(evaluation_pts, geometric_queries,
                                                               solve_double_sided, grid_res,
                                                               boundary_distance_mask)
        save_image_buffer(solution_file, solution, save_colormapped, colormap,
                          colormap_min_val, colormap_max_val)

    elif solver_type == "rws":
        # create evaluation points on grid to compute solution on
        evaluation_pts = create_rws_evaluation_point_grid(grid_res, geometric_queries,
                                                          solve_double_sided)

        # run reverse walk on stars
        sample_counts = run_reverse_walk_on_stars(solver_config, evaluation_pts,
                                                  absorbing_boundary_positions, absorbing_boundary_indices,
                                                  reflecting_boundary_positions, reflecting_boundary_indices,
                                                  geometric_queries, pde, solve_double_sided)

        # save solution to disk
        solution = get_solution_from_rws_evaluation_point_grid(evaluation_pts, sample_counts,
                                                               geometric_queries, solve_double_sided,
                                                               grid_res, boundary_distance_mask)
        save_image_buffer(solution_file, solution, save_colormapped, colormap,
                          colormap_min_val, colormap_max_val)

if __name__ == "__main__":
    # parse arguments
    parser = argparse.ArgumentParser(description="zombie 2d demo application")
    parser.add_argument("--config", type=str, help="path to the configuration file")
    args = parser.parse_args()

    try:
        # load the configuration file
        with open(args.config, 'r') as file:
            config = json.load(file)

        # load a boundary mesh
        model_problem_config = config["modelProblem"]
        positions, indices, bounding_box = load_boundary_mesh(model_problem_config)

        # setup the PDE
        pde = setup_pde(model_problem_config, bounding_box)

        # partition the boundary mesh
        absorbing_boundary_positions, absorbing_boundary_indices,\
            reflecting_boundary_positions, reflecting_boundary_indices =\
                partition_boundary_mesh(positions, indices, pde.has_reflecting_boundary_conditions)

        # populate geometric queries
        geometric_queries, absorbing_boundary_handler, reflecting_boundary_handler =\
            populate_geometric_queries(model_problem_config, absorbing_boundary_positions,
                                       absorbing_boundary_indices, reflecting_boundary_positions,
                                       reflecting_boundary_indices, bounding_box)

        # run the solver
        solver_type = config["solverType"]
        solver_config = config["solver"]
        output_config = config["output"]
        run_solver(model_problem_config, solver_type, solver_config, output_config,
                   absorbing_boundary_positions, absorbing_boundary_indices,
                   reflecting_boundary_positions, reflecting_boundary_indices,
                   geometric_queries, pde)

    except FileNotFoundError:
        print("Configuration file not found")

    except json.JSONDecodeError:
        print("Invalid configuration file")

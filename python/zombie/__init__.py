# This module provides a clean Python API for Zombie by wrapping the C++ nanobind
# extension with factory functions that accept `dim` and `channels` keyword arguments,
# hiding the internal suffixed class names.
#
# The dispatch functions are added directly onto the C++ submodules (Core, Solvers, etc.)
# so that help() on those submodules shows both the dispatched names and the full nanobind
# documentation for the underlying C++ classes.

_VALID_DIMS = {2, 3}
_VALID_CHANNELS = {1, 4, 16, 64, 256}

def _validate_dim(dim):
    if dim not in _VALID_DIMS:
        raise ValueError(f"dim must be one of {sorted(_VALID_DIMS)}, got {dim}")

def _validate_channels(channels):
    if channels not in _VALID_CHANNELS:
        raise ValueError(f"channels must be one of {sorted(_VALID_CHANNELS)}, got {channels}")

def _resolve_dim_channels_name(cpp_base_name, dim, channels):
    if channels == 1:
        return f"{cpp_base_name}_float_{dim}d"
    return f"{cpp_base_name}_float{channels}_{dim}d"

def _resolve_dim_name(cpp_base_name, dim):
    return f"{cpp_base_name}_{dim}d"

def _dim_channels_factory(cpp_submodule, cpp_base_name):
    def factory(*args, dim, channels=1, **kwargs):
        _validate_dim(dim)
        _validate_channels(channels)
        cls = getattr(cpp_submodule, _resolve_dim_channels_name(cpp_base_name, dim, channels))
        return cls(*args, **kwargs)
    factory.__name__ = factory.__qualname__ = cpp_base_name
    factory.__doc__ = (
        f"Creates a {cpp_base_name} specialized for the given dim and channels.\n\n"
        f"dim must be 2 or 3. channels must be one of {{1, 4, 16, 64, 256}} (default: 1).\n"
        f"Additional arguments are forwarded to the {cpp_base_name} constructor."
    )
    return factory

def _dim_factory(cpp_submodule, cpp_base_name):
    def factory(*args, dim, **kwargs):
        _validate_dim(dim)
        cls = getattr(cpp_submodule, _resolve_dim_name(cpp_base_name, dim))
        return cls(*args, **kwargs)
    factory.__name__ = factory.__qualname__ = cpp_base_name
    factory.__doc__ = (
        f"Creates a {cpp_base_name} specialized for the given dim.\n\n"
        f"dim must be 2 or 3.\n"
        f"Additional arguments are forwarded to the {cpp_base_name} constructor."
    )
    return factory

def _dim_channels_func(cpp_submodule, cpp_base_name):
    def func(*args, dim, channels=1, **kwargs):
        _validate_dim(dim)
        _validate_channels(channels)
        fn = getattr(cpp_submodule, _resolve_dim_channels_name(cpp_base_name, dim, channels))
        return fn(*args, **kwargs)
    func.__name__ = func.__qualname__ = cpp_base_name
    func.__doc__ = (
        f"Calls {cpp_base_name} for the given dim and channels.\n\n"
        f"dim must be 2 or 3. channels must be one of {{1, 4, 16, 64, 256}} (default: 1).\n"
        f"Additional arguments are forwarded to {cpp_base_name}."
    )
    return func

def _dim_func(cpp_submodule, cpp_base_name):
    def func(*args, dim, **kwargs):
        _validate_dim(dim)
        fn = getattr(cpp_submodule, _resolve_dim_name(cpp_base_name, dim))
        return fn(*args, **kwargs)
    func.__name__ = func.__qualname__ = cpp_base_name
    func.__doc__ = (
        f"Calls {cpp_base_name} for the given dim.\n\n"
        f"dim must be 2 or 3.\n"
        f"Additional arguments are forwarded to {cpp_base_name}."
    )
    return func

try:
    from . import _zombie

    # ==================== Top-level types (non-templated) ====================

    BoolList = _zombie.BoolList
    IntList = _zombie.IntList
    UintList = _zombie.UintList
    FloatList = _zombie.FloatList
    convert_list_to_numpy_array = _zombie.convert_list_to_numpy_array

    # Dim-specific list types (keep direct names for convenience)
    Float2List = _zombie.Float2List
    Float3List = _zombie.Float3List
    Int2List = _zombie.Int2List
    Int3List = _zombie.Int3List

    # Dim-dispatched float list constructor
    def FloatNList(*args, dim, **kwargs):
        _validate_dim(dim)
        cls = _zombie.Float2List if dim == 2 else _zombie.Float3List
        return cls(*args, **kwargs)

    # Dim-dispatched int list constructor
    def IntNList(*args, dim, **kwargs):
        _validate_dim(dim)
        cls = _zombie.Int2List if dim == 2 else _zombie.Int3List
        return cls(*args, **kwargs)

    # ==================== Expose C++ submodules directly ====================
    # We add dispatch functions onto the existing C++ submodules so that
    # help() shows full nanobind docs for all suffixed types AND the new
    # dispatched names side by side.

    Core = _zombie.Core
    Solvers = _zombie.Solvers
    Samplers = _zombie.Samplers
    Utils = _zombie.Utils
    OpaqueTypes = _zombie.OpaqueTypes

    # ==================== Core dispatchers ====================

    Core.PDE = _dim_channels_factory(Core, "PDE")
    Core.get_constant_source_callback = _dim_channels_func(Core, "get_constant_source_callback")
    Core.get_constant_dirichlet_callback = _dim_channels_func(Core, "get_constant_dirichlet_callback")
    Core.get_constant_robin_callback = _dim_channels_func(Core, "get_constant_robin_callback")

    Core.IntersectionPoint = _dim_factory(Core, "IntersectionPoint")
    Core.GeometricQueries = _dim_factory(Core, "GeometricQueries")

    Core.get_constant_indicator_callback = _dim_func(Core, "get_constant_indicator_callback")
    Core.get_constant_robin_coefficient_callback = _dim_func(Core, "get_constant_robin_coefficient_callback")

    # ==================== Solvers dispatchers ====================

    for _name in [
        "WalkState",
        "SampleStatistics",
        "SampleStatisticsList",
        "SamplePoint",
        "SamplePointList",
        "WalkOnSpheres",
        "WalkOnStars",
        "BVCEvaluationPoint",
        "BVCEvaluationPointList",
        "BoundaryValueCaching",
        "RWSEvaluationPoint",
        "RWSEvaluationPointList",
        "ReverseWalkOnStars",
        "KelvinTransform",
    ]:
        setattr(Solvers, _name, _dim_channels_factory(Solvers, _name))

    Solvers.create_sample_statistics_list = _dim_channels_func(
        Solvers, "create_sample_statistics_list")
    Solvers.get_empty_walk_state_callback = _dim_channels_func(
        Solvers, "get_empty_walk_state_callback")
    Solvers.get_empty_terminal_contribution_callback = _dim_channels_func(
        Solvers, "get_empty_terminal_contribution_callback")

    # ==================== Samplers dispatchers ====================

    Samplers.BoundarySampler = _dim_channels_factory(Samplers, "BoundarySampler")
    Samplers.DomainSampler = _dim_channels_factory(Samplers, "DomainSampler")

    Samplers.create_uniform_line_segment_boundary_sampler = _dim_channels_func(
        Samplers, "create_uniform_line_segment_boundary_sampler")
    Samplers.create_uniform_triangle_boundary_sampler = _dim_channels_func(
        Samplers, "create_uniform_triangle_boundary_sampler")
    Samplers.create_uniform_domain_sampler = _dim_channels_func(
        Samplers, "create_uniform_domain_sampler")

    # ==================== Utils dispatchers ====================

    Utils.FloatDenseGrid = _dim_channels_factory(Utils, "DenseGrid")
    Utils.BoolDenseGrid = _dim_factory(Utils, "DenseGrid_bool")

    for _name in [
        "FcpwDirichletBoundaryHandler",
        "FcpwNeumannBoundaryHandler",
        "FcpwRobinBoundaryHandler",
        "SDFGrid",
    ]:
        setattr(Utils, _name, _dim_factory(Utils, _name))

    for _name in [
        "load_boundary_mesh",
        "load_textured_boundary_mesh",
        "normalize",
        "apply_shift",
        "flip_orientation",
        "compute_bounding_box",
        "add_bounding_box_to_boundary_mesh",
        "compute_signed_volume",
        "compute_dist_to_boundary",
        "partition_boundary_mesh",
        "populate_sdf_grid",
        "populate_geometric_queries_for_dirichlet_boundary",
        "populate_geometric_queries_for_neumann_boundary",
        "populate_geometric_queries_for_robin_boundary",
        "get_spatially_sorted_point_indices",
    ]:
        setattr(Utils, _name, _dim_func(Utils, _name))

    for _name in [
        "get_dense_grid_source_callback",
        "get_dense_grid_dirichlet_callback",
        "get_dense_grid_robin_callback",
    ]:
        setattr(Utils, _name, _dim_channels_func(Utils, _name))

    Utils.get_dense_grid_indicator_callback = _dim_func(Utils, "get_dense_grid_indicator_callback")
    Utils.get_dense_grid_robin_coefficient_callback = _dim_func(Utils, "get_dense_grid_robin_coefficient_callback")

    # ==================== Clean up ====================

    del _name

    __all__ = [
        "BoolList",
        "IntList",
        "UintList",
        "FloatList",
        "Float2List",
        "Float3List",
        "Int2List",
        "Int3List",
        "FloatNList",
        "IntNList",
        "convert_list_to_numpy_array",
        "Core",
        "Solvers",
        "Samplers",
        "Utils",
        "OpaqueTypes",
    ]

except ImportError as e:
    import warnings
    warnings.warn(
        f"Failed to import Zombie extension module: {e}\n"
        "This usually means:\n"
        "  1. The package was not built with bindings enabled (ZOMBIE_BUILD_BINDINGS=ON)\n"
        "  2. The extension module is missing from the installation\n"
        "  3. There are missing dependencies (e.g., Slang library for GPU support)",
        ImportWarning
    )
    __all__ = []

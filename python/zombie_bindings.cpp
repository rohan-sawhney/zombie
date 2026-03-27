// This file creates python bindings for Zombie using nanobind.

#include "zombie_bindings.h"

// Helper function to bind all callback, PDE, solver, and sampler resources for a given type T and dimension DIM.
// The typeStr is used internally by nanobind to create unique Python class names (hidden from user by __init__.py).
template <typename T, size_t DIM>
void bindTypedResources(nb::module_ m, const std::string& typeStr, const std::string& dimStr) {
    std::string suffix = "_" + typeStr + "_" + dimStr;

    // bind opaque callback types
    bindFloatNToTypeFunc<T, DIM>(m, typeStr);
    bindFloatNBoolToTypeFunc<T, DIM>(m, typeStr);
    bindFloatNFloatNBoolToTypeFunc<T, DIM>(m, typeStr);
    bindWalkStateFuncs<T, DIM>(m, typeStr);

    // bind dense grid
    bindDenseGrid<T, DIM>(m, suffix);

    // bind PDE resources
    bindPDESouceCallbacks<T, DIM>(m, suffix);
    bindPDEDirichletCallbacks<T, DIM>(m, suffix);
    bindPDERobinCallbacks<T, DIM>(m, suffix);
    bindPDEStructure<T, DIM>(m, suffix);

    // bind solver and sampler resources
    bindRandomWalkStructures<T, DIM>(m, suffix);
    bindWalkOnSpheresSolver<T, DIM>(m, suffix);
    bindWalkOnStarsSolver<T, DIM>(m, suffix);
    bindSamplers<T, DIM>(m, suffix);
    bindBoundaryValueCachingSolver<T, DIM>(m, suffix);
    bindReverseWalkOnStarsSolver<T, DIM>(m, suffix);
    bindKelvinTransform<T, DIM>(m, suffix);
}

// Helper function to bind all channel variants for a given dimension.
template <size_t DIM>
void bindAllChannels(nb::module_ m, const std::string& dimStr) {
    bindTypedResources<float, DIM>(m, "float", dimStr);
    bindTypedResources<zombie::Array<float, 4>, DIM>(m, "float4", dimStr);
    bindTypedResources<zombie::Array<float, 16>, DIM>(m, "float16", dimStr);
    bindTypedResources<zombie::Array<float, 64>, DIM>(m, "float64", dimStr);
    bindTypedResources<zombie::Array<float, 256>, DIM>(m, "float256", dimStr);
}

NB_MODULE(_zombie, m) {
    m.doc() = "Zombie Python bindings";

    // bind non-templated resources
    bindNonTemplatedLibraryResources(m);

    // bind dim-only opaque callback types
    bindIntersectBoundaryFuncs<2>(m);
    bindIntersectBoundaryFuncs<3>(m);

    // bind bool callback types
    bindFloatNToTypeFunc<bool, 2>(m, "bool");
    bindFloatNToTypeFunc<bool, 3>(m, "bool");

    // bind all channel variants for 2D and 3D
    // NOTE: this must come before bindGeometryUtilityFunctions because SDFGrid inherits
    // from DenseGrid<float, 1, DIM>, which is registered inside bindAllChannels.
    bindAllChannels<2>(m, "2d");
    bindAllChannels<3>(m, "3d");

    // bind bool dense grids
    bindDenseGrid<bool, 2>(m, "_bool_2d");
    bindDenseGrid<bool, 3>(m, "_bool_3d");

    // bind dim-only geometry resources
    bindCoreGeometryStructures<2>(m, "_2d");
    bindCoreGeometryStructures<3>(m, "_3d");
    bindGeometryUtilityFunctions<2>(m, "_2d");
    bindGeometryUtilityFunctions<3>(m, "_3d");

    // bind dim-only PDE opaque callback types
    bindPDEIndicatorCallbacks<2>(m, "_2d");
    bindPDEIndicatorCallbacks<3>(m, "_3d");
    bindPDECoefficientCallbacks<2>(m, "_2d");
    bindPDECoefficientCallbacks<3>(m, "_3d");
}

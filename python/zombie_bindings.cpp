// This file creates python bindings for Zombie using nanobind.

#include "zombie_bindings.h"

NB_MODULE(py, m) {
    m.doc() = "Zombie Python bindings";

    // bind non-templated resources
    bindNonTemplatedLibraryResources(m);

    // bind templated geometry, PDE and solver callbacks
    bindIntersectBoundaryFuncs<2>(m);
    bindFloatNToTypeFunc<bool, 2>(m, "bool");

    bindFloatNToTypeFunc<float, 2>(m, "float");
    bindFloatNBoolToTypeFunc<float, 2>(m, "float");
    bindFloatNFloatNBoolToTypeFunc<float, 2>(m, "float");
    bindWalkStateFuncs<float, 2>(m, "float");

    bindFloatNToTypeFunc<zombie::Array<float, 3>, 2>(m, "float3");
    bindFloatNBoolToTypeFunc<zombie::Array<float, 3>, 2>(m, "float3");
    bindWalkStateFuncs<zombie::Array<float, 3>, 2>(m, "float3");

    bindIntersectBoundaryFuncs<3>(m);
    bindFloatNToTypeFunc<bool, 3>(m, "bool");

    bindFloatNToTypeFunc<float, 3>(m, "float");
    bindFloatNBoolToTypeFunc<float, 3>(m, "float");
    bindFloatNFloatNBoolToTypeFunc<float, 3>(m, "float");
    bindWalkStateFuncs<float, 3>(m, "float");

    bindFloatNToTypeFunc<zombie::Array<float, 3>, 3>(m, "float3");
    bindFloatNBoolToTypeFunc<zombie::Array<float, 3>, 3>(m, "float3");
    bindWalkStateFuncs<zombie::Array<float, 3>, 3>(m, "float3");

    // bind dense grid
    bindDenseGrid<bool, 2>(m, "_bool_2d");
    bindDenseGrid<float, 2>(m, "_float_2d");
    bindDenseGrid<zombie::Array<float, 3>, 2>(m, "_float3_2d");

    bindDenseGrid<bool, 3>(m, "_bool_3d");
    bindDenseGrid<float, 3>(m, "_float_3d");
    bindDenseGrid<zombie::Array<float, 3>, 3>(m, "_float3_3d");

    // bind geometry resources
    bindCoreGeometryStructures<2>(m, "_2d");
    bindGeometryUtilityFunctions<2>(m, "_2d");

    bindCoreGeometryStructures<3>(m, "_3d");
    bindGeometryUtilityFunctions<3>(m, "_3d");

    // bind PDE resources
    bindPDEIndicatorCallbacks<2>(m, "_2d");
    bindPDECoefficientCallbacks<2>(m, "_2d");

    bindPDESouceCallbacks<float, 2>(m, "_float_2d");
    bindPDEBoundaryConditionCallbacks<float, 2>(m, "_float_2d");
    bindPDEStructure<float, 2>(m, "_float_2d");

    bindPDESouceCallbacks<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindPDEBoundaryConditionCallbacks<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindPDEStructure<zombie::Array<float, 3>, 2>(m, "_float3_2d");

    bindPDEIndicatorCallbacks<3>(m, "_3d");
    bindPDECoefficientCallbacks<3>(m, "_3d");

    bindPDESouceCallbacks<float, 3>(m, "_float_3d");
    bindPDEBoundaryConditionCallbacks<float, 3>(m, "_float_3d");
    bindPDEStructure<float, 3>(m, "_float_3d");

    bindPDESouceCallbacks<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindPDEBoundaryConditionCallbacks<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindPDEStructure<zombie::Array<float, 3>, 3>(m, "_float3_3d");

    // bind solver resources
    bindRandomWalkStructures<float, 2>(m, "_float_2d");
    bindWalkOnSpheresSolver<float, 2>(m, "_float_2d");
    bindWalkOnStarsSolver<float, 2>(m, "_float_2d");
    bindSamplers<float, 2>(m, "_float_2d");
    bindBoundaryValueCachingSolver<float, 2>(m, "_float_2d");
    bindReverseWalkOnStarsSolver<float, 2>(m, "_float_2d");

    bindRandomWalkStructures<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindWalkOnSpheresSolver<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindWalkOnStarsSolver<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindSamplers<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindBoundaryValueCachingSolver<zombie::Array<float, 3>, 2>(m, "_float3_2d");
    bindReverseWalkOnStarsSolver<zombie::Array<float, 3>, 2>(m, "_float3_2d");

    bindRandomWalkStructures<float, 3>(m, "_float_3d");
    bindWalkOnSpheresSolver<float, 3>(m, "_float_3d");
    bindWalkOnStarsSolver<float, 3>(m, "_float_3d");
    bindSamplers<float, 3>(m, "_float_3d");
    bindBoundaryValueCachingSolver<float, 3>(m, "_float_3d");
    bindReverseWalkOnStarsSolver<float, 3>(m, "_float_3d");

    bindRandomWalkStructures<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindWalkOnSpheresSolver<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindWalkOnStarsSolver<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindSamplers<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindBoundaryValueCachingSolver<zombie::Array<float, 3>, 3>(m, "_float3_3d");
    bindReverseWalkOnStarsSolver<zombie::Array<float, 3>, 3>(m, "_float3_3d");
}

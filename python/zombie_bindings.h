// This file contains helper functions to create python bindings for Zombie using nanobind.
// Functions are templated on a value type T which can be, e.g., a float for scalar-valued
// PDE problems or an Array<float, CHANNELS> for vector-valued problems. The template parameter
// DIM specifies the dimension of the problem, and is typically 2 or 3. Users can use these
// functions to create bindings specialized to value types and dimensions of their choice.

#pragma once

#include <nanobind/nanobind.h>
#include <nanobind/stl/bind_vector.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/string.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/ndarray.h>
#include <zombie/zombie.h>

namespace nb = nanobind;
using namespace nb::literals;

// vector types
using BoolList = std::vector<bool>;
using IntList = std::vector<int>;
using UintList = std::vector<uint32_t>;
using FloatList = std::vector<float>;

template <size_t DIM>
using IntNList = std::vector<zombie::Vectori<DIM>>;
template <size_t DIM>
using FloatNList = std::vector<zombie::Vector<DIM>>;

template <typename T, size_t DIM>
using SamplePointList = std::vector<zombie::SamplePoint<T, DIM>>;
template <typename T, size_t DIM>
using SampleStatisticsList = std::vector<zombie::SampleStatistics<T, DIM>>;
template <typename T, size_t DIM>
using BVCEvaluationPointList = std::vector<zombie::bvc::EvaluationPoint<T, DIM>>;
template <typename T, size_t DIM>
using RWSEvaluationPointList = std::vector<zombie::rws::EvaluationPoint<T, DIM>>;

// opaque types
using FloatIntToBoolFunc = std::function<bool(float, int)>;
using FloatToFloatFunc = std::function<float(float)>;
using IntIntToVoidFunc = std::function<void(int, int)>;
using VoidToFloatFunc = std::function<float()>;
template <size_t DIM>
using IntersectBoundaryFunc = std::function<bool(const zombie::Vector<DIM>&, const zombie::Vector<DIM>&,
                                                 const zombie::Vector<DIM>&, float, bool,
                                                 zombie::IntersectionPoint<DIM>&)>;
template <size_t DIM>
using IntersectUnionedBoundaryFunc = std::function<bool(const zombie::Vector<DIM>&, const zombie::Vector<DIM>&,
                                                        const zombie::Vector<DIM>&, float, bool, bool,
                                                        zombie::IntersectionPoint<DIM>&)>;
template <typename T, size_t DIM>
using FloatNToTypeFunc = std::function<T(const zombie::Vector<DIM>&)>;
template <typename T, size_t DIM>
using FloatNBoolToTypeFunc = std::function<T(const zombie::Vector<DIM>&, bool)>;
template <typename T, size_t DIM>
using FloatNFloatNBoolToTypeFunc = std::function<T(const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool)>;
template <typename T, size_t DIM>
using WalkStateToVoidFunc = std::function<void(const zombie::WalkState<T, DIM>&)>;
template <typename T, size_t DIM>
using WalkCodeStateToTypeFunc = std::function<T(zombie::WalkCompletionCode, const zombie::WalkState<T, DIM>&)>;

NB_MAKE_OPAQUE(FloatIntToBoolFunc)                                     // ignoreCandidateSilhouette
NB_MAKE_OPAQUE(FloatToFloatFunc)                                       // branchTraversalWeight
NB_MAKE_OPAQUE(IntIntToVoidFunc)                                       // reportProgress
NB_MAKE_OPAQUE(VoidToFloatFunc)                                        // computeSignedVolume
NB_MAKE_OPAQUE(IntersectBoundaryFunc<2>)                               // intersectAbsorbingBoundary, intersectReflectingBoundary
NB_MAKE_OPAQUE(IntersectBoundaryFunc<3>)                               // intersectAbsorbingBoundary, intersectReflectingBoundary
NB_MAKE_OPAQUE(IntersectUnionedBoundaryFunc<2>)                        // intersectBoundary
NB_MAKE_OPAQUE(IntersectUnionedBoundaryFunc<3>)                        // intersectBoundary
NB_MAKE_OPAQUE(FloatNToTypeFunc<bool, 2>)                              // insideDomain, insideBoundingDomain, outsideBoundingDomain, hasReflectingBoundaryConditions, insideSolveRegion
NB_MAKE_OPAQUE(FloatNToTypeFunc<bool, 3>)                              // insideDomain, insideBoundingDomain, outsideBoundingDomain, hasReflectingBoundaryConditions, insideSolveRegion
NB_MAKE_OPAQUE(FloatNToTypeFunc<float, 2>)                             // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<float, 3>)                             // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<zombie::Array<float, 4>, 2>)           // source
NB_MAKE_OPAQUE(FloatNToTypeFunc<zombie::Array<float, 4>, 3>)           // source
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<float, 2>)                         // computeDistToBoundary, dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<float, 3>)                         // computeDistToBoundary, dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<zombie::Array<float, 4>, 2>)       // dirichlet
NB_MAKE_OPAQUE(FloatNBoolToTypeFunc<zombie::Array<float, 4>, 3>)       // dirichlet
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<float, 2>)                   // robinCoeff, robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<float, 3>)                   // robinCoeff, robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<zombie::Array<float, 4>, 2>) // robin
NB_MAKE_OPAQUE(FloatNFloatNBoolToTypeFunc<zombie::Array<float, 4>, 3>) // robin
NB_MAKE_OPAQUE(WalkStateToVoidFunc<float, 2>)                          // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<float, 3>)                          // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<zombie::Array<float, 4>, 2>)        // walkStateCallback
NB_MAKE_OPAQUE(WalkStateToVoidFunc<zombie::Array<float, 4>, 3>)        // walkStateCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<float, 2>)                      // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<float, 3>)                      // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<zombie::Array<float, 4>, 2>)    // terminalContributionCallback
NB_MAKE_OPAQUE(WalkCodeStateToTypeFunc<zombie::Array<float, 4>, 3>)    // terminalContributionCallback

// binding functions
void bindNonTemplatedLibraryResources(nb::module_ m);

template <size_t DIM>
void bindIntersectBoundaryFuncs(nb::module_ m);
template <typename T, size_t DIM>
void bindFloatNToTypeFunc(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindFloatNFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindWalkStateFuncs(nb::module_ m, std::string typeStr="");

template <typename T, size_t DIM>
void bindDenseGrid(nb::module_ m, std::string typeStr="");

template <size_t DIM>
void bindCoreGeometryStructures(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindGeometryUtilityFunctions(nb::module_ m, std::string typeStr=""); // currently valid only for 2D or 3D

template <size_t DIM>
void bindPDEIndicatorCallbacks(nb::module_ m, std::string typeStr="");
template <size_t DIM>
void bindPDECoefficientCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDESouceCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDEDirichletCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDERobinCallbacks(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindPDEStructure(nb::module_ m, std::string typeStr="");

template <typename T, size_t DIM>
void bindRandomWalkStructures(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindWalkOnSpheresSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindWalkOnStarsSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindSamplers(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindBoundaryValueCachingSolver(nb::module_ m, std::string typeStr="");
template <typename T, size_t DIM>
void bindReverseWalkOnStarsSolver(nb::module_ m, std::string typeStr="");

/////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// Implementation

template <typename List, typename T>
nb::ndarray<nb::numpy, T> convertListToNumpyArray(const List& v)
{
    std::cerr << "convertListToNumpyArray: Unsupported type" << std::endl;
    exit(EXIT_FAILURE);
}

template <>
nb::ndarray<nb::numpy, bool> convertListToNumpyArray(const BoolList& v)
{
    // allocate a memory region an initialize it
    bool *data = new bool[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (bool *)p;
    });

    return nb::ndarray<nb::numpy, bool>(data, { v.size() }, owner);
}

template <>
nb::ndarray<nb::numpy, int> convertListToNumpyArray(const IntList& v)
{
    // allocate a memory region an initialize it
    int *data = new int[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (int *)p;
    });

    return nb::ndarray<nb::numpy, int>(data, { v.size() }, owner);
}

template <>
nb::ndarray<nb::numpy, uint32_t> convertListToNumpyArray(const UintList& v)
{
    // allocate a memory region an initialize it
    uint32_t *data = new uint32_t[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (uint32_t *)p;
    });

    return nb::ndarray<nb::numpy, uint32_t>(data, { v.size() }, owner);
}

template <>
nb::ndarray<nb::numpy, float> convertListToNumpyArray(const FloatList& v)
{
    // allocate a memory region an initialize it
    float *data = new float[v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[i] = v[i];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *)p;
    });

    return nb::ndarray<nb::numpy, float>(data, { v.size() }, owner);
}

template <>
nb::ndarray<nb::numpy, int> convertListToNumpyArray(const IntNList<2>& v)
{
    // allocate a memory region an initialize it
    int *data = new int[2*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[2*i + 0] = v[i][0];
        data[2*i + 1] = v[i][1];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (int *)p;
    });

    return nb::ndarray<nb::numpy, int>(data, { v.size(), 2 }, owner);
}

template <>
nb::ndarray<nb::numpy, int> convertListToNumpyArray(const IntNList<3>& v)
{
    // allocate a memory region an initialize it
    int *data = new int[3*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[3*i + 0] = v[i][0];
        data[3*i + 1] = v[i][1];
        data[3*i + 2] = v[i][2];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (int *)p;
    });

    return nb::ndarray<nb::numpy, int>(data, { v.size(), 3 }, owner);
}

template <>
nb::ndarray<nb::numpy, float> convertListToNumpyArray(const FloatNList<2>& v)
{
    // allocate a memory region an initialize it
    float *data = new float[2*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[2*i + 0] = v[i][0];
        data[2*i + 1] = v[i][1];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *)p;
    });

    return nb::ndarray<nb::numpy, float>(data, { v.size(), 2 }, owner);
}

template <>
nb::ndarray<nb::numpy, float> convertListToNumpyArray(const FloatNList<3>& v)
{
    // allocate a memory region an initialize it
    float *data = new float[3*v.size()];
    for (size_t i = 0; i < v.size(); i++) {
        data[3*i + 0] = v[i][0];
        data[3*i + 1] = v[i][1];
        data[3*i + 2] = v[i][2];
    }

    // delete 'data' when the 'owner' capsule expires
    nb::capsule owner(data, [](void *p) noexcept {
        delete[] (float *)p;
    });

    return nb::ndarray<nb::numpy, float>(data, { v.size(), 3 }, owner);
}

void bindNonTemplatedLibraryResources(nb::module_ m)
{
    nb::bind_vector<BoolList>(m, "bool_list");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<BoolList, bool>);

    nb::bind_vector<IntList>(m, "int_list");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<IntList, int>);

    nb::bind_vector<UintList>(m, "uint_list");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<UintList, uint32_t>);

    nb::bind_vector<FloatList>(m, "float_list");
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<FloatList, float>);

    nb::module_ opaque_types_m = m.def_submodule("opaque_types", "Opaque types module");

    nb::class_<FloatIntToBoolFunc>(opaque_types_m, "func<bool(float, int)>")
        .def("__call__", [](const FloatIntToBoolFunc& callback,
                            float a, int b) -> bool {
            return callback(a, b);
        });

    nb::class_<FloatToFloatFunc>(opaque_types_m, "func<float(float)>")
        .def("__call__", [](const FloatToFloatFunc& callback,
                            float a) -> float {
            return callback(a);
        });

    nb::class_<IntIntToVoidFunc>(opaque_types_m, "func<void(int, int)>")
        .def("__call__", [](const IntIntToVoidFunc& callback,
                            int a, int b) -> void {
            callback(a, b);
        });

    nb::class_<VoidToFloatFunc>(opaque_types_m, "func<float(void)>")
        .def("__call__", [](const VoidToFloatFunc& callback) -> float {
            return callback();
        });

    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::enum_<zombie::SampleType>(solvers_m, "sample_type")
        .value("in_domain", zombie::SampleType::InDomain)
        .value("on_absorbing_boundary", zombie::SampleType::OnAbsorbingBoundary)
        .value("on_reflecting_boundary", zombie::SampleType::OnReflectingBoundary);

    nb::enum_<zombie::EstimationQuantity>(solvers_m, "estimation_quantity")
        .value("solution", zombie::EstimationQuantity::Solution)
        .value("solution_and_gradient", zombie::EstimationQuantity::SolutionAndGradient)
        .value("none", zombie::EstimationQuantity::None);

    nb::enum_<zombie::WalkCompletionCode>(solvers_m, "walk_completion_code")
        .value("reached_absorbing_boundary", zombie::WalkCompletionCode::ReachedAbsorbingBoundary)
        .value("terminated_with_russian_roulette", zombie::WalkCompletionCode::TerminatedWithRussianRoulette)
        .value("exceeded_max_walk_length", zombie::WalkCompletionCode::ExceededMaxWalkLength)
        .value("escaped_domain", zombie::WalkCompletionCode::EscapedDomain);

    nb::class_<zombie::WalkSettings>(solvers_m, "walk_settings")
        .def(nb::init<float, float, int, bool>(),
            "epsilon_shell_for_absorbing_boundary"_a, "epsilon_shell_for_reflecting_boundary"_a,
            "max_walk_length"_a, "solve_double_sided"_a)
        .def(nb::init<float, float, float, float, float, int, int, int,
                      bool, bool, bool, bool, bool, bool, bool, bool>(),
            "epsilon_shell_for_absorbing_boundary"_a, "epsilon_shell_for_reflecting_boundary"_a,
            "silhouette_precision"_a, "russian_roulette_threshold"_a, "splitting_threshold"_a,
            "max_walk_length"_a, "steps_before_applying_tikhonov"_a,
            "steps_before_using_maximal_spheres"_a, "solve_double_sided"_a,
            "use_gradient_control_variates"_a, "use_gradient_antithetic_variates"_a,
            "use_cosine_sampling_for_derivatives"_a, "ignore_absorbing_boundary_contribution"_a,
            "ignore_reflecting_boundary_contribution"_a, "ignore_source_contribution"_a, "print_logs"_a)
        .def_rw("epsilon_shell_for_absorbing_boundary", &zombie::WalkSettings::epsilonShellForAbsorbingBoundary)
        .def_rw("epsilon_shell_for_reflecting_boundary", &zombie::WalkSettings::epsilonShellForReflectingBoundary)
        .def_rw("silhouette_precision", &zombie::WalkSettings::silhouettePrecision)
        .def_rw("russian_roulette_threshold", &zombie::WalkSettings::russianRouletteThreshold)
        .def_rw("splitting_threshold", &zombie::WalkSettings::splittingThreshold)
        .def_rw("max_walk_length", &zombie::WalkSettings::maxWalkLength)
        .def_rw("steps_before_applying_tikhonov", &zombie::WalkSettings::stepsBeforeApplyingTikhonov)
        .def_rw("steps_before_using_maximal_spheres", &zombie::WalkSettings::stepsBeforeUsingMaximalSpheres)
        .def_rw("solve_double_sided", &zombie::WalkSettings::solveDoubleSided)
        .def_rw("use_gradient_control_variates", &zombie::WalkSettings::useGradientControlVariates)
        .def_rw("use_gradient_antithetic_variates", &zombie::WalkSettings::useGradientAntitheticVariates)
        .def_rw("use_cosine_sampling_for_derivatives", &zombie::WalkSettings::useCosineSamplingForDerivatives)
        .def_rw("ignore_absorbing_boundary_contribution", &zombie::WalkSettings::ignoreAbsorbingBoundaryContribution)
        .def_rw("ignore_reflecting_boundary_contribution", &zombie::WalkSettings::ignoreReflectingBoundaryContribution)
        .def_rw("ignore_source_contribution", &zombie::WalkSettings::ignoreSourceContribution)
        .def_rw("print_logs", &zombie::WalkSettings::printLogs);

    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    utils_m.def("get_ignore_candidate_silhouette_callback",
               &zombie::getIgnoreCandidateSilhouetteCallback,
               "solve_double_sided"_a=false, "silhouette_precision"_a=1e-3f,
               "Returns a callback that ignores silhouette candidates on the reflecting boundary based on the given parameters.");

    utils_m.def("get_branch_traversal_weight_callback",
               &zombie::getBranchTraversalWeightCallback,
               "min_radial_distance"_a=1e-2f,
               "Returns a branch traversal weight function for a reflecting boundary.");

    nb::class_<ProgressBar>(utils_m, "progress_bar")
        .def(nb::init<int, int>(),
            "total_work"_a, "display_width"_a=80)
        .def("report", &ProgressBar::report,
            "Reports progress on the progress bar.",
            "new_work_completed"_a, "thread_id"_a)
        .def("finish", &ProgressBar::finish,
            "Finishes the progress bar.");

    utils_m.def("get_report_progress_callback",
               &getReportProgressCallback,
               "progress_bar"_a,
               "Returns a callback that reports progress using a progress bar.");
}

template <size_t DIM>
void bindIntersectBoundaryFuncs(nb::module_ m)
{
    nb::module_ opaque_types_m = m.def_submodule("opaque_types", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string intersectionPointStr = "intersection_point_" + std::to_string(DIM) + "d";
    std::string funcStr = "func<bool(" + vectorStr + ", " + vectorStr + ", " + vectorStr + ", float, bool, " + intersectionPointStr + ")>";
    nb::class_<IntersectBoundaryFunc<DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const IntersectBoundaryFunc<DIM>& callback,
                            const zombie::Vector<DIM>& a, const zombie::Vector<DIM>& b,
                            const zombie::Vector<DIM>& c, float d, bool e,
                            zombie::IntersectionPoint<DIM>& f) -> bool {
            return callback(a, b, c, d, e, f);
        });

    funcStr = "func<bool(" + vectorStr + ", " + vectorStr + ", " + vectorStr + ", float, bool, bool, " + intersectionPointStr + ")>";
    nb::class_<IntersectUnionedBoundaryFunc<DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const IntersectUnionedBoundaryFunc<DIM>& callback,
                            const zombie::Vector<DIM>& a, const zombie::Vector<DIM>& b,
                            const zombie::Vector<DIM>& c, float d, bool e, bool f,
                            zombie::IntersectionPoint<DIM>& g) -> bool {
            return callback(a, b, c, d, e, f, g);
        });
}

template <typename T, size_t DIM>
void bindFloatNToTypeFunc(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("opaque_types", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string funcStr = "func<" + typeStr + "(" + vectorStr + ")>";
    nb::class_<FloatNToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const FloatNToTypeFunc<T, DIM>& callback,
                            const zombie::Vector<DIM>& a) -> T {
            return callback(a);
        });
}

template <typename T, size_t DIM>
void bindFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("opaque_types", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string funcStr = "func<" + typeStr + "(" + vectorStr + ", bool)>";
    nb::class_<FloatNBoolToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const FloatNBoolToTypeFunc<T, DIM>& callback,
                            const zombie::Vector<DIM>& a, bool b) -> T {
            return callback(a, b);
        });
}

template <typename T, size_t DIM>
void bindFloatNFloatNBoolToTypeFunc(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("opaque_types", "Opaque types module");

    std::string vectorStr = "float" + std::to_string(DIM);
    std::string funcStr = "func<" + typeStr + "(" + vectorStr + ", " + vectorStr + ", bool)>";
    nb::class_<FloatNFloatNBoolToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const FloatNFloatNBoolToTypeFunc<T, DIM>& callback,
                            const zombie::Vector<DIM>& a,
                            const zombie::Vector<DIM>& b, bool c) -> T {
            return callback(a, b, c);
        });
}

template <typename T, size_t DIM>
void bindWalkStateFuncs(nb::module_ m, std::string typeStr)
{
    nb::module_ opaque_types_m = m.def_submodule("opaque_types", "Opaque types module");

    std::string funcStr = "func<void(walk_state_" + typeStr + "_" + std::to_string(DIM) + "d)>";
    nb::class_<WalkStateToVoidFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const WalkStateToVoidFunc<T, DIM>& callback,
                            const zombie::WalkState<T, DIM>& a) -> void {
            callback(a);
        });

    funcStr = "func<" + typeStr + "(walk_completion_code, walk_state_" + typeStr + "_" + std::to_string(DIM) + "d)>";
    nb::class_<WalkCodeStateToTypeFunc<T, DIM>>(opaque_types_m, funcStr.c_str())
        .def("__call__", [](const WalkCodeStateToTypeFunc<T, DIM>& callback,
                            zombie::WalkCompletionCode a,
                            const zombie::WalkState<T, DIM>& b) -> T {
            return callback(a, b);
        });
}

template <size_t DIM>
void bindCoreGeometryStructures(nb::module_ m, std::string typeStr)
{
    nb::bind_vector<IntNList<DIM>>(m, ("int" + std::to_string(DIM) + "_list").c_str());
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<IntNList<DIM>, int>);

    nb::bind_vector<FloatNList<DIM>>(m, ("float" + std::to_string(DIM) + "_list").c_str());
    m.def("convert_list_to_numpy_array", &convertListToNumpyArray<FloatNList<DIM>, float>);

    nb::module_ core_m = m.def_submodule("core", "Core module");

    nb::class_<zombie::IntersectionPoint<DIM>>(core_m, ("intersection_point" + typeStr).c_str())
        .def(nb::init<>())
        .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, float>(),
            "pt"_a, "normal"_a, "dist"_a)
        .def_rw("pt", &zombie::IntersectionPoint<DIM>::pt)
        .def_rw("normal", &zombie::IntersectionPoint<DIM>::normal)
        .def_rw("dist", &zombie::IntersectionPoint<DIM>::dist);

    nb::class_<zombie::GeometricQueries<DIM>>(core_m, ("geometric_queries" + typeStr).c_str())
        .def(nb::init<>())
        .def(nb::init<bool, const zombie::Vector<DIM>&, const zombie::Vector<DIM>&>(),
            "domain_is_watertight"_a, "domain_min"_a, "domain_max"_a)
        .def_ro("has_non_empty_absorbing_boundary", &zombie::GeometricQueries<DIM>::hasNonEmptyAbsorbingBoundary)
        .def_ro("has_non_empty_reflecting_boundary", &zombie::GeometricQueries<DIM>::hasNonEmptyReflectingBoundary)
        .def_ro("domain_is_watertight", &zombie::GeometricQueries<DIM>::domainIsWatertight)
        .def_ro("domain_min", &zombie::GeometricQueries<DIM>::domainMin)
        .def_ro("domain_max", &zombie::GeometricQueries<DIM>::domainMax)
        .def_ro("compute_dist_to_absorbing_boundary", &zombie::GeometricQueries<DIM>::computeDistToAbsorbingBoundary)
        .def_ro("compute_dist_to_reflecting_boundary", &zombie::GeometricQueries<DIM>::computeDistToReflectingBoundary)
        .def_ro("compute_dist_to_boundary", &zombie::GeometricQueries<DIM>::computeDistToBoundary)
        .def_ro("intersect_absorbing_boundary", &zombie::GeometricQueries<DIM>::intersectAbsorbingBoundary)
        .def_ro("intersect_reflecting_boundary", &zombie::GeometricQueries<DIM>::intersectReflectingBoundary)
        .def_ro("intersect_boundary", &zombie::GeometricQueries<DIM>::intersectBoundary)
        .def_ro("inside_domain", &zombie::GeometricQueries<DIM>::insideDomain)
        .def_ro("inside_bounding_domain", &zombie::GeometricQueries<DIM>::insideBoundingDomain)
        .def_ro("outside_bounding_domain", &zombie::GeometricQueries<DIM>::outsideBoundingDomain)
        .def_ro("compute_domain_signed_volume", &zombie::GeometricQueries<DIM>::computeDomainSignedVolume);
}

template <typename T, size_t DIM>
void bindDenseGrid(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value ||
                  std::is_integral<T>::value ||
                  std::is_same<T, bool>::value) {
        nb::class_<zombie::DenseGrid<T, 1, DIM>>(utils_m, ("dense_grid" + typeStr).c_str())
            .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool>(),
                "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false)
            .def(nb::init<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                          const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                          const zombie::Vector<DIM>&, bool>(),
                "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                "enable_interpolation"_a=false)
            .def("set", nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                          const zombie::Vectori<DIM>&>(
                &zombie::DenseGrid<T, 1, DIM>::set),
                "sets the grid data.",
                "grid_data"_a, "grid_shape"_a)
            .def("set", nb::overload_cast<int, const zombie::Array<T, 1>&>(
                &zombie::DenseGrid<T, 1, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("set", nb::overload_cast<const zombie::Vectori<DIM>&,
                                          const zombie::Array<T, 1>&>(
                &zombie::DenseGrid<T, 1, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("get", nb::overload_cast<int>(
                &zombie::DenseGrid<T, 1, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("get", nb::overload_cast<const zombie::Vectori<DIM>&>(
                &zombie::DenseGrid<T, 1, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("__call__", &zombie::DenseGrid<T, 1, DIM>::operator(),
                "returns the grid value at a point.",
                "x"_a)
            .def("min", &zombie::DenseGrid<T, 1, DIM>::min,
                "returns the minimum grid value.")
            .def("max", &zombie::DenseGrid<T, 1, DIM>::max,
                "returns the maximum grid value.")
            .def_ro("data", &zombie::DenseGrid<T, 1, DIM>::data)
            .def_ro("shape", &zombie::DenseGrid<T, 1, DIM>::shape)
            .def_ro("origin", &zombie::DenseGrid<T, 1, DIM>::origin)
            .def_ro("extent", &zombie::DenseGrid<T, 1, DIM>::extent);

    } else {
        nb::class_<zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>>(utils_m, ("dense_grid" + typeStr).c_str())
            .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, bool>(),
                "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false)
            .def(nb::init<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                          const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                          const zombie::Vector<DIM>&, bool>(),
                "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                "enable_interpolation"_a=false)
            .def("set", nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                          const zombie::Vectori<DIM>&>(
                &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::set),
                "sets the grid data.",
                "grid_data"_a, "grid_shape"_a)
            .def("set", nb::overload_cast<int, const zombie::Array<float, T::RowsAtCompileTime>&>(
                &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("set", nb::overload_cast<const zombie::Vectori<DIM>&,
                                          const zombie::Array<float, T::RowsAtCompileTime>&>(
                &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::set),
                "sets the grid value at an index.",
                "index"_a, "value"_a)
            .def("get", nb::overload_cast<int>(
                &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("get", nb::overload_cast<const zombie::Vectori<DIM>&>(
                &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::get, nb::const_),
                "returns the world-space position for an index.",
                "index"_a)
            .def("__call__", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::operator(),
                "returns the grid value at a point.",
                "x"_a)
            .def("min", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::min,
                "returns the minimum grid value.")
            .def("max", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::max,
                "returns the maximum grid value.")
            .def_ro("data", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::data)
            .def_ro("shape", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::shape)
            .def_ro("origin", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::origin)
            .def_ro("extent", &zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>::extent);
    }
}

template <size_t DIM>
void bindGeometryUtilityFunctions(nb::module_ m, std::string typeStr)
{
    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    utils_m.def(("load_boundary_mesh" + typeStr).c_str(),
               &zombie::loadBoundaryMesh<DIM>,
               "obj_file"_a, "positions"_a, "indices"_a,
               "Loads boundary mesh from OBJ file.");

    if (DIM == 3) {
        utils_m.def(("load_textured_boundary_mesh" + typeStr).c_str(),
                    &zombie::loadTexturedBoundaryMesh<DIM>,
                    "obj_file"_a, "positions"_a, "texture_coordinates"_a,
                    "indices"_a, "texture_indices"_a,
                    "Loads textured boundary mesh from OBJ file.");
    }

    utils_m.def(("normalize" + typeStr).c_str(),
               &zombie::normalize<DIM>,
               "positions"_a,
               "Normalizes positions to the unit sphere.");

    utils_m.def(("apply_shift" + typeStr).c_str(),
               &zombie::applyShift<DIM>,
               "shift"_a, "positions"_a,
               "Applies a shift to a set of positions.");

    utils_m.def(("flip_orientation" + typeStr).c_str(),
               &zombie::flipOrientation<DIM>,
               "indices"_a,
               "Flips the orientation of a boundary mesh.");

    utils_m.def(("compute_bounding_box" + typeStr).c_str(),
               &zombie::computeBoundingBox<DIM>,
               "positions"_a, "make_square"_a, "scale"_a,
               "Computes the bounding box of a boundary mesh.");

    utils_m.def(("add_bounding_box_to_boundary_mesh" + typeStr).c_str(),
               &zombie::addBoundingBoxToBoundaryMesh<DIM>,
               "bounding_box_min"_a, "bounding_box_max"_a, "positions"_a, "indices"_a,
               "Adds a bounding box to a boundary mesh.");

    utils_m.def(("partition_boundary_mesh" + typeStr).c_str(),
               &zombie::partitionBoundaryMesh<DIM>,
               "on_reflecting_boundary"_a, "positions"_a, "indices"_a,
               "absorbing_positions"_a, "absorbing_indices"_a,
               "reflecting_positions"_a, "reflecting_indices"_a,
               "Partitions a boundary mesh into absorbing and reflecting parts using primitive centroids---\nthis assumes the boundary discretization is perfectly adapted to the boundary conditions,\nwhich isn't always a correct assumption.");

    nb::class_<zombie::FcpwDirichletBoundaryHandler<DIM>>(utils_m, ("fcpw_dirichlet_boundary_handler" + typeStr).c_str())
        .def(nb::init<>())
        .def("build_acceleration_structure", &zombie::FcpwDirichletBoundaryHandler<DIM>::buildAccelerationStructure,
            "Builds an FCPW acceleration structure (specifically a BVH) from a set of positions and indices.\nUses a simple list of mesh faces for brute-force geometric queries when build_bvh is false.",
            "positions"_a, "indices"_a, "build_bvh"_a=true, "enable_bvh_vectorization"_a=false);

    nb::class_<zombie::FcpwNeumannBoundaryHandler<DIM>>(utils_m, ("fcpw_neumann_boundary_handler" + typeStr).c_str())
        .def(nb::init<>())
        .def("build_acceleration_structure", &zombie::FcpwNeumannBoundaryHandler<DIM>::buildAccelerationStructure,
            "Builds an FCPW acceleration structure (specifically a BVH) from a set of positions and indices.\nUses a simple list of mesh faces for brute-force geometric queries when build_bvh is false.",
            "positions"_a, "indices"_a, "ignore_candidate_silhouette"_a, "build_bvh"_a=true, "enable_bvh_vectorization"_a=false);

    nb::class_<zombie::FcpwRobinBoundaryHandler<DIM>>(utils_m, ("fcpw_robin_boundary_handler" + typeStr).c_str())
        .def(nb::init<>())
        .def("build_acceleration_structure", &zombie::FcpwRobinBoundaryHandler<DIM>::buildAccelerationStructure,
            "Builds an FCPW acceleration structure (specifically a BVH) from a set of positions, indices, and min and max absolute coefficient values per mesh face.\nUses a simple list of mesh faces for brute-force geometric queries when build_bvh is false.",
            "positions"_a, "indices"_a, "ignore_candidate_silhouette"_a,
            "min_robin_coeff_values"_a, "max_robin_coeff_values"_a,
            "build_bvh"_a=true, "enable_bvh_vectorization"_a=false)
        .def("update_coefficient_values", &zombie::FcpwRobinBoundaryHandler<DIM>::updateCoefficientValues,
            "updates the Robin coefficients on the boundary mesh.",
            "min_robin_coeff_values"_a, "max_robin_coeff_values"_a);

    nb::class_<zombie::SdfGrid<DIM>, zombie::DenseGrid<float, 1, DIM>>(utils_m, ("sdf_grid" + typeStr).c_str())
        .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&>(),
            "grid_min"_a, "grid_max"_a)
        .def(nb::init<const Eigen::VectorXf&, const zombie::Vectori<DIM>&,
                      const zombie::Vector<DIM>&, const zombie::Vector<DIM>&>(),
            "sdf_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a)
        .def("compute_gradient", &zombie::SdfGrid<DIM>::computeGradient,
            "Computes the gradient of the SDF at the given point.",
            "x"_a)
        .def("compute_normal", &zombie::SdfGrid<DIM>::computeNormal,
            "Computes the normal of the level set at the given point.", 
            "x"_a)
        .def("project_to_zero_level_set", &zombie::SdfGrid<DIM>::projectToZeroLevelSet,
            "Projects the given point to the zero level set.",
            "x"_a, "normal"_a, "max_iterations"_a=8, "epsilon"_a=1e-6f)
        .def("intersect_zero_level_set", &zombie::SdfGrid<DIM>::intersectZeroLevelSet,
            "Intersects the zero level set with the given ray.",
            "origin"_a, "direction"_a, "t_max"_a, "intersection_pt"_a,
            "max_iterations"_a=128, "epsilon"_a=1e-6f);

    utils_m.def(("populate_sdf_grid" + typeStr).c_str(),
               &zombie::populateSdfGrid<DIM>,
               "dirichlet_boundary_handler"_a, "sdf_grid"_a, "grid_shape"_a,
               "compute_unsigned_distance"_a=false,
               "Populates an SDF grid from a Dirichlet boundary handler.");

    utils_m.def(("populate_geometric_queries_for_dirichlet_boundary" + typeStr).c_str(),
               nb::overload_cast<zombie::FcpwDirichletBoundaryHandler<DIM>&,
                                 zombie::GeometricQueries<DIM>&>(
               &zombie::populateGeometricQueriesForDirichletBoundary<DIM>),
               "fcpw_dirichlet_boundary_handler"_a, "geometric_queries"_a,
               "Populates geometric queries for an absorbing Dirichlet boundary.");

    utils_m.def(("populate_geometric_queries_for_dirichlet_boundary" + typeStr).c_str(),
               nb::overload_cast<const zombie::SdfGrid<DIM>&, zombie::GeometricQueries<DIM>&>(
               &zombie::populateGeometricQueriesForDirichletBoundary<zombie::SdfGrid<DIM>, DIM>),
               "sdf_grid"_a, "geometric_queries"_a,
               "Populates geometric queries for an absorbing Dirichlet boundary.");

    utils_m.def(("populate_geometric_queries_for_neumann_boundary" + typeStr).c_str(),
               &zombie::populateGeometricQueriesForNeumannBoundary<DIM>,
               "fcpw_neumann_boundary_handler"_a, "branch_traversal_weight"_a, "geometric_queries"_a,
               "Populates geometric queries for a reflecting Neumann boundary.");

    utils_m.def(("populate_geometric_queries_for_robin_boundary" + typeStr).c_str(),
               &zombie::populateGeometricQueriesForRobinBoundary<DIM>,
               "fcpw_robin_boundary_handler"_a, "branch_traversal_weight"_a, "geometric_queries"_a,
               "Populates geometric queries for a reflecting Robin boundary.");

    utils_m.def(("get_spatially_sorted_point_indices" + typeStr).c_str(),
               &zombie::getSpatiallySortedPointIndices<DIM>,
               "points"_a, "out_indices"_a,
               "Outputs a list of indices that spatially sort the input points.");
}

template <size_t DIM>
void bindPDEIndicatorCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("core", "Core module");

    core_m.def(("get_constant_indicator_callback" + typeStr).c_str(),
              [](bool value) -> FloatNToTypeFunc<bool, DIM> {
                return [value](const zombie::Vector<DIM>& a) -> bool { return value; };
              },
              "value"_a,
              "Returns a constant indicator callback.");

    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    utils_m.def(("get_dense_grid_indicator_callback" + typeStr).c_str(),
               nb::overload_cast<const zombie::DenseGrid<bool, 1, DIM>&>(
               &zombie::getDenseGridCallback0<bool, bool, 1, DIM>),
               "grid"_a,
               "Returns a dense grid indicator callback.");
    utils_m.def(("get_dense_grid_indicator_callback" + typeStr).c_str(),
               nb::overload_cast<const Eigen::Matrix<bool, Eigen::Dynamic, 1>&,
                                 const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                 const zombie::Vector<DIM>&, bool>(
               &zombie::getDenseGridCallback0<bool, bool, 1, DIM>),
               "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
               "enable_interpolation"_a=false,
               "Returns a dense grid indicator callback.");
}

template <size_t DIM>
void bindPDECoefficientCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("core", "Core module");

    core_m.def(("get_constant_robin_coefficient_callback" + typeStr).c_str(),
              [](float value) -> FloatNFloatNBoolToTypeFunc<float, DIM> {
                return [value](const zombie::Vector<DIM>& a,
                               const zombie::Vector<DIM>& b, bool c) -> float {
                    return value;
                };
              },
              "value"_a,
              "Returns a constant coefficient callback.");
    core_m.def(("get_constant_robin_coefficient_callback" + typeStr).c_str(),
              [](float value, float valueBoundaryNormalAligned) -> FloatNFloatNBoolToTypeFunc<float, DIM> {
                return [value, valueBoundaryNormalAligned](const zombie::Vector<DIM>& a,
                                                           const zombie::Vector<DIM>& b, bool c) -> float {
                    return c ? valueBoundaryNormalAligned : value;
                };
              },
              "value"_a, "value_boundary_normal_aligned"_a,
              "Returns a constant coefficient callback.");

    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const zombie::DenseGrid<float, 1, DIM>&>(
               &zombie::getDenseGridCallback3<float, float, 1, DIM>),
               "grid"_a,
               "Returns a dense grid coefficient callback.");
    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, 1>&,
                                 const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                 const zombie::Vector<DIM>&, bool>(
               &zombie::getDenseGridCallback3<float, float, 1, DIM>),
               "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
               "enable_interpolation"_a=false,
               "Returns a dense grid coefficient callback.");

    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const zombie::DenseGrid<float, 1, DIM>&,
                                 const zombie::DenseGrid<float, 1, DIM>&>(
               &zombie::getDenseGridCallback4<float, float, 1, DIM>),
               "grid"_a, "grid_boundary_normal_aligned"_a,
               "Returns a dense grid coefficient callback.");
    utils_m.def(("get_dense_grid_robin_coefficient_callback" + typeStr).c_str(),
               nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, 1>&,
                                 const Eigen::Matrix<float, Eigen::Dynamic, 1>&,
                                 const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                 const zombie::Vector<DIM>&, bool>(
               &zombie::getDenseGridCallback4<float, float, 1, DIM>),
               "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
               "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
               "Returns a dense grid coefficient callback.");
}

template <typename T, size_t DIM>
void bindPDESouceCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("core", "Core module");

    core_m.def(("get_constant_source_callback" + typeStr).c_str(),
              [](float value) -> FloatNToTypeFunc<T, DIM> {
                return [value](const zombie::Vector<DIM>& a) -> T { return T(value); };
              },
              "value"_a,
              "Returns a constant source callback.");

    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value) {
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<T, 1, DIM>&>(
                   &zombie::getDenseGridCallback0<T, T, 1, DIM>),
                   "grid"_a,
                   "Returns a dense grid source callback.");
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback0<T, T, 1, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid source callback.");

    } else {
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &zombie::getDenseGridCallback0<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a,
                   "Returns a dense grid source callback.");
        utils_m.def(("get_dense_grid_source_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback0<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid source callback.");
    }
}

template <typename T, size_t DIM>
void bindPDEDirichletCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("core", "Core module");

    core_m.def(("get_constant_dirichlet_callback" + typeStr).c_str(),
              [](float value) -> FloatNBoolToTypeFunc<T, DIM> {
                return [value](const zombie::Vector<DIM>& a, bool b) -> T { return T(value); };
              },
              "value"_a,
              "Returns a constant dirichlet boundary condition callback.");
    core_m.def(("get_constant_dirichlet_callback" + typeStr).c_str(),
              [](float value, float valueBoundaryNormalAligned) -> FloatNBoolToTypeFunc<T, DIM> {
                return [value, valueBoundaryNormalAligned](const zombie::Vector<DIM>& a, bool b) -> T {
                    return b ? T(valueBoundaryNormalAligned) : T(value);
                };
              },
              "value"_a, "value_boundary_normal_aligned"_a,
              "Returns a constant dirichlet boundary condition callback.");

    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value) {
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<T, 1, DIM>&>(
                   &zombie::getDenseGridCallback1<T, T, 1, DIM>),
                   "grid"_a,
                   "Returns a dense grid dirichlet boundary condition callback.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback1<T, T, 1, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");

        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<T, 1, DIM>&,
                                     const zombie::DenseGrid<T, 1, DIM>&>(
                   &zombie::getDenseGridCallback2<T, T, 1, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid dirichlet boundary condition callback.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback2<T, T, 1, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");

    } else {
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &zombie::getDenseGridCallback1<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a,
                   "Returns a dense grid dirichlet boundary condition callback.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback1<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");

        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&,
                                     const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &zombie::getDenseGridCallback2<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid dirichlet boundary condition callback.");
        utils_m.def(("get_dense_grid_dirichlet_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback2<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid dirichlet boundary condition callback.");
    }
}

template <typename T, size_t DIM>
void bindPDERobinCallbacks(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("core", "Core module");

    core_m.def(("get_constant_robin_callback" + typeStr).c_str(),
              [](float value) -> FloatNFloatNBoolToTypeFunc<T, DIM> {
                return [value](const zombie::Vector<DIM>& a,
                               const zombie::Vector<DIM>& b, bool c) -> T { return T(value); };
              },
              "value"_a,
              "Returns a constant robin boundary condition callback.");
    core_m.def(("get_constant_robin_callback" + typeStr).c_str(),
              [](float value, float valueBoundaryNormalAligned) -> FloatNFloatNBoolToTypeFunc<T, DIM> {
                return [value, valueBoundaryNormalAligned](const zombie::Vector<DIM>& a,
                                                           const zombie::Vector<DIM>& b, bool c) -> T {
                    return c ? T(valueBoundaryNormalAligned) : T(value);
                };
              },
              "value"_a, "value_boundary_normal_aligned"_a,
              "Returns a constant robin boundary condition callback.");

    nb::module_ utils_m = m.def_submodule("utils", "Utilities module");

    if constexpr (std::is_floating_point<T>::value) {
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<T, 1, DIM>&>(
                   &zombie::getDenseGridCallback3<T, T, 1, DIM>),
                   "grid"_a,
                   "Returns a dense grid robin boundary condition callback.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback3<T, T, 1, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");

        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<T, 1, DIM>&,
                                     const zombie::DenseGrid<T, 1, DIM>&>(
                   &zombie::getDenseGridCallback4<T, T, 1, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid robin boundary condition callback.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const Eigen::Matrix<T, Eigen::Dynamic, 1>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback4<T, T, 1, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");

    } else {
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &zombie::getDenseGridCallback3<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a,
                   "Returns a dense grid robin boundary condition callback.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback3<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_shape"_a, "grid_min"_a, "grid_max"_a,
                   "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");

        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&,
                                     const zombie::DenseGrid<float, T::RowsAtCompileTime, DIM>&>(
                   &zombie::getDenseGridCallback4<T, float, T::RowsAtCompileTime, DIM>),
                   "grid"_a, "grid_boundary_normal_aligned"_a,
                   "Returns a dense grid robin boundary condition callback.");
        utils_m.def(("get_dense_grid_robin_callback" + typeStr).c_str(),
                   nb::overload_cast<const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const Eigen::Matrix<float, Eigen::Dynamic, T::RowsAtCompileTime>&,
                                     const zombie::Vectori<DIM>&, const zombie::Vector<DIM>&,
                                     const zombie::Vector<DIM>&, bool>(
                   &zombie::getDenseGridCallback4<T, float, T::RowsAtCompileTime, DIM>),
                   "grid_data"_a, "grid_data_boundary_normal_aligned"_a, "grid_shape"_a,
                   "grid_min"_a, "grid_max"_a, "enable_interpolation"_a=false,
                   "Returns a dense grid robin boundary condition callback.");
    }
}

template <typename T, size_t DIM>
void bindPDEStructure(nb::module_ m, std::string typeStr)
{
    nb::module_ core_m = m.def_submodule("core", "Core module");

    nb::class_<zombie::PDE<T, DIM>>(core_m, ("pde" + typeStr).c_str())
        .def(nb::init<>())
        .def_rw("absorption_coeff", &zombie::PDE<T, DIM>::absorptionCoeff)
        .def_rw("are_robin_conditions_pure_neumann", &zombie::PDE<T, DIM>::areRobinConditionsPureNeumann)
        .def_rw("are_robin_coeffs_nonnegative", &zombie::PDE<T, DIM>::areRobinCoeffsNonnegative)
        .def_rw("source", &zombie::PDE<T, DIM>::source)
        .def_rw("dirichlet", &zombie::PDE<T, DIM>::dirichlet)
        .def_rw("robin", &zombie::PDE<T, DIM>::robin)
        .def_rw("robin_coeff", &zombie::PDE<T, DIM>::robinCoeff)
        .def_rw("has_reflecting_boundary_conditions", &zombie::PDE<T, DIM>::hasReflectingBoundaryConditions);
}

template <typename T, size_t DIM>
void bindRandomWalkStructures(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::class_<zombie::WalkState<T, DIM>>(solvers_m, ("walk_state" + typeStr).c_str())
        .def_ro("current_pt", &zombie::WalkState<T, DIM>::currentPt)
        .def_ro("current_normal", &zombie::WalkState<T, DIM>::currentNormal)
        .def_ro("prev_direction", &zombie::WalkState<T, DIM>::prevDirection)
        .def_ro("prev_distance", &zombie::WalkState<T, DIM>::prevDistance)
        .def_ro("walk_length", &zombie::WalkState<T, DIM>::walkLength)
        .def_ro("on_reflecting_boundary", &zombie::WalkState<T, DIM>::onReflectingBoundary);

    nb::class_<zombie::SampleStatistics<T, DIM>>(solvers_m, ("sample_statistics" + typeStr).c_str())
        .def(nb::init<>())
        .def("reset", &zombie::SampleStatistics<T, DIM>::reset,
            "Resets statistics.")
        .def("get_estimated_solution", &zombie::SampleStatistics<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.")
        .def("get_estimated_gradient", nb::overload_cast<int>(
            &zombie::SampleStatistics<T, DIM>::getEstimatedGradient, nb::const_),
            "Returns estimated gradient for specified channel.",
            "channel"_a)
        .def("get_estimated_derivative", &zombie::SampleStatistics<T, DIM>::getEstimatedDerivative,
            "Returns estimated directional derivative.")
        .def("get_solution_estimate_count", &zombie::SampleStatistics<T, DIM>::getSolutionEstimateCount,
            "Returns number of solution estimates.")
        .def("get_gradient_estimate_count", &zombie::SampleStatistics<T, DIM>::getGradientEstimateCount,
            "Returns number of gradient estimates.")
        .def("get_mean_walk_length", &zombie::SampleStatistics<T, DIM>::getMeanWalkLength,
            "Returns mean walk length.");

    nb::class_<zombie::SamplePoint<T, DIM>>(solvers_m, ("sample_point" + typeStr).c_str())
        .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, zombie::SampleType,
                      zombie::EstimationQuantity, float, float, float>(),
            "pt"_a, "normal"_a, "type"_a, "estimation_quantity"_a, "pdf"_a,
            "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("reset", &zombie::SamplePoint<T, DIM>::reset,
            "Resets the solution data.")
        .def_rw("pt", &zombie::SamplePoint<T, DIM>::pt)
        .def_rw("normal", &zombie::SamplePoint<T, DIM>::normal)
        .def_rw("direction_for_derivative", &zombie::SamplePoint<T, DIM>::directionForDerivative)
        .def_rw("type", &zombie::SamplePoint<T, DIM>::type)
        .def_rw("estimation_quantity", &zombie::SamplePoint<T, DIM>::estimationQuantity)
        .def_rw("pdf", &zombie::SamplePoint<T, DIM>::pdf)
        .def_rw("dist_to_absorbing_boundary", &zombie::SamplePoint<T, DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &zombie::SamplePoint<T, DIM>::distToReflectingBoundary)
        .def_ro("first_sphere_radius", &zombie::SamplePoint<T, DIM>::firstSphereRadius)
        .def_rw("estimate_boundary_normal_aligned", &zombie::SamplePoint<T, DIM>::estimateBoundaryNormalAligned);

    nb::bind_vector<SamplePointList<T, DIM>>(solvers_m, ("sample_point" + typeStr + "_list").c_str());
    nb::bind_vector<SampleStatisticsList<T, DIM>>(solvers_m, ("sample_statistics" + typeStr + "_list").c_str());

    solvers_m.def(("get_empty_walk_state_callback" + typeStr).c_str(),
                 []() -> WalkStateToVoidFunc<T, DIM> { return {}; },
                 "Returns an empty walk state callback.");

    solvers_m.def(("get_empty_terminal_contribution_callback" + typeStr).c_str(),
                 []() -> WalkCodeStateToTypeFunc<T, DIM> { return {}; },
                 "Returns an empty terminal contribution callback.");
}

template <typename T, size_t DIM>
void bindWalkOnSpheresSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::class_<zombie::WalkOnSpheres<T, DIM>>(solvers_m, ("walk_on_spheres" + typeStr).c_str())
        .def(nb::init<const zombie::GeometricQueries<DIM>&>(),
            "geometric_queries"_a)
        .def(nb::init<const zombie::GeometricQueries<DIM>&, WalkStateToVoidFunc<T, DIM>, WalkCodeStateToTypeFunc<T, DIM>>(),
            "geometric_queries"_a, "walk_state_callback"_a, "terminal_contribution_callback"_a)
        .def("solve", nb::overload_cast<const zombie::PDE<T, DIM>&, const zombie::WalkSettings&, int,
                                        zombie::SamplePoint<T, DIM>&, zombie::SampleStatistics<T, DIM>&>(
            &zombie::WalkOnSpheres<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input point.\nAssumes the point does not lie on the boundary when estimating the gradient.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pt"_a, "statistics"_a)
        .def("solve", nb::overload_cast<const zombie::PDE<T, DIM>&, const zombie::WalkSettings&, const IntList&,
                                        SamplePointList<T, DIM>&, SampleStatisticsList<T, DIM>&, bool, IntIntToVoidFunc>(
            &zombie::WalkOnSpheres<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input points.\nAssumes points do not lie on the boundary when estimating gradients.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pts"_a, "statistics"_a,
            "run_single_threaded"_a=false, "report_progress"_a.none());
}

template <typename T, size_t DIM>
void bindWalkOnStarsSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::class_<zombie::WalkOnStars<T, DIM>>(solvers_m, ("walk_on_stars" + typeStr).c_str())
        .def(nb::init<const zombie::GeometricQueries<DIM>&>(),
            "geometric_queries"_a)
        .def(nb::init<const zombie::GeometricQueries<DIM>&, WalkStateToVoidFunc<T, DIM>, WalkCodeStateToTypeFunc<T, DIM>>(),
            "geometric_queries"_a, "walk_state_callback"_a, "terminal_contribution_callback"_a)
        .def("solve", nb::overload_cast<const zombie::PDE<T, DIM>&, const zombie::WalkSettings&, int,
                                        zombie::SamplePoint<T, DIM>&, zombie::SampleStatistics<T, DIM>&>(
            &zombie::WalkOnStars<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input point.\nAssumes the point does not lie on the boundary when estimating the gradient.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pt"_a, "statistics"_a)
        .def("solve", nb::overload_cast<const zombie::PDE<T, DIM>&, const zombie::WalkSettings&, const IntList&,
                                        SamplePointList<T, DIM>&, SampleStatisticsList<T, DIM>&, bool, IntIntToVoidFunc>(
            &zombie::WalkOnStars<T, DIM>::solve, nb::const_),
            "Solves the given PDE at the input points.\nAssumes points do not lie on the boundary when estimating gradients.",
            "pde"_a, "walk_settings"_a, "n_walks"_a, "sample_pts"_a, "statistics"_a,
            "run_single_threaded"_a=false, "report_progress"_a.none());
}

template <typename T, size_t DIM>
void bindSamplers(nb::module_ m, std::string typeStr)
{
    nb::module_ samplers_m = m.def_submodule("samplers", "Samplers module");

    nb::class_<zombie::BoundarySampler<T, DIM>>(samplers_m, ("boundary_sampler" + typeStr).c_str())
        .def("initialize", &zombie::BoundarySampler<T, DIM>::initialize,
            "Performs any sampler specific initialization.",
            "normal_offset_for_boundary"_a, "solve_double_sided"_a)
        .def("get_sample_count", &zombie::BoundarySampler<T, DIM>::getSampleCount,
            "Returns the number of sample points to be generated on the user-specified side of the boundary.",
            "n_total_samples"_a, "boundary_normal_aligned_samples"_a=false)
        .def("generate_samples", &zombie::BoundarySampler<T, DIM>::generateSamples,
            "Generates sample points on the boundary.",
            "n_samples"_a, "sample_type"_a, "normal_offset_for_boundary"_a,
            "geometric_queries"_a, "sample_pts"_a,
            "generate_boundary_normal_aligned_samples"_a=false);

    if (DIM == 2) {
        samplers_m.def(("create_uniform_line_segment_boundary_sampler" + typeStr).c_str(),
                      &zombie::createUniformLineSegmentBoundarySampler<T>,
                      "positions"_a, "indices"_a, "inside_solve_region"_a,
                      "compute_weighted_normals"_a=false,
                      "Creates a uniform line segment boundary sampler.");

    } else if (DIM == 3) {
        samplers_m.def(("create_uniform_triangle_boundary_sampler" + typeStr).c_str(),
                      &zombie::createUniformTriangleBoundarySampler<T>,
                      "positions"_a, "indices"_a, "inside_solve_region"_a,
                      "compute_weighted_normals"_a=false,
                      "Creates a uniform triangle boundary sampler.");
    }

    nb::class_<zombie::DomainSampler<T, DIM>>(samplers_m, ("domain_sampler" + typeStr).c_str())
        .def("generate_samples", &zombie::DomainSampler<T, DIM>::generateSamples,
            "Generates sample points inside the user-specified solve region.",
            "n_samples"_a, "geometric_queries"_a, "sample_pts"_a);

    samplers_m.def(("create_uniform_domain_sampler" + typeStr).c_str(),
                  &zombie::createUniformDomainSampler<T, DIM>,
                  "inside_solve_region"_a, "solve_region_min"_a,
                  "solve_region_max"_a, "solve_region_volume"_a,
                  "Creates a uniform domain sampler.");
}

template <typename T, size_t DIM>
void bindBoundaryValueCachingSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::class_<zombie::bvc::EvaluationPoint<T, DIM>>(solvers_m, ("bvc_evaluation_point" + typeStr).c_str())
        .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, zombie::SampleType, float, float>(),
            "pt"_a, "normal"_a, "type"_a, "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("get_estimated_solution", &zombie::bvc::EvaluationPoint<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.")
        .def("get_estimated_gradient", nb::overload_cast<int>(
            &zombie::bvc::EvaluationPoint<T, DIM>::getEstimatedGradient, nb::const_),
            "Returns estimated gradient for specified channel.",
            "channel"_a)
        .def("reset", &zombie::bvc::EvaluationPoint<T, DIM>::reset,
            "Resets statistics.")
        .def_rw("pt", &zombie::bvc::EvaluationPoint<T, DIM>::pt)
        .def_rw("normal", &zombie::bvc::EvaluationPoint<T, DIM>::normal)
        .def_rw("type", &zombie::bvc::EvaluationPoint<T, DIM>::type)
        .def_rw("dist_to_absorbing_boundary", &zombie::bvc::EvaluationPoint<T, DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &zombie::bvc::EvaluationPoint<T, DIM>::distToReflectingBoundary);

    nb::bind_vector<BVCEvaluationPointList<T, DIM>>(solvers_m, ("bvc_evaluation_point" + typeStr + "_list").c_str());

    nb::class_<zombie::bvc::BoundaryValueCachingSolver<T, DIM>>(solvers_m, ("boundary_value_caching" + typeStr).c_str())
        .def(nb::init<const zombie::GeometricQueries<DIM>&,
                      std::shared_ptr<zombie::BoundarySampler<T, DIM>>,
                      std::shared_ptr<zombie::BoundarySampler<T, DIM>>,
                      std::shared_ptr<zombie::DomainSampler<T, DIM>>>(),
            "geometric_queries"_a, "absorbing_boundary_sampler"_a, "reflecting_boundary_sampler"_a, "domain_sampler"_a)
        .def("generate_samples", &zombie::bvc::BoundaryValueCachingSolver<T, DIM>::generateSamples,
            "Generates boundary and domain samples.",
            "absorbing_boundary_cache_size"_a, "reflecting_boundary_cache_size"_a, "domain_cache_size"_a,
            "normal_offset_for_absorbing_boundary"_a, "normal_offset_for_reflecting_boundary"_a, "solve_double_sided"_a)
        .def("compute_sample_estimates", &zombie::bvc::BoundaryValueCachingSolver<T, DIM>::computeSampleEstimates,
            "Computes sample estimates on the boundary.",
            "pde"_a, "walk_settings"_a, "n_walks_for_solution_estimates"_a, "n_walks_for_gradient_estimates"_a,
            "robin_coeff_cutoff_for_normal_derivative"_a, "use_finite_differences"_a=false,
            "run_single_threaded"_a=false, "report_progress"_a.none())
        .def("splat", &zombie::bvc::BoundaryValueCachingSolver<T, DIM>::splat,
            "Splat solution and gradient estimates into the interior.",
            "pde"_a, "radius_clamp"_a, "kernel_regularization"_a, "robin_coeff_cutoff_for_normal_derivative"_a,
            "cutoff_dist_to_absorbing_boundary"_a, "cutoff_dist_to_reflecting_boundary"_a, "eval_pts"_a,
            "report_progress"_a.none())
        .def("estimate_solution_near_boundary",
            &zombie::bvc::BoundaryValueCachingSolver<T, DIM>::estimateSolutionNearBoundary,
            "Estimates the solution at the input evaluation points near the boundary.",
            "pde"_a, "walk_settings"_a, "cutoff_dist_to_absorbing_boundary"_a, "cutoff_dist_to_reflecting_boundary"_a,
            "n_walks_for_solution_estimates"_a, "eval_pts"_a, "run_single_threaded"_a=false);
}

template <typename T, size_t DIM>
void bindReverseWalkOnStarsSolver(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::class_<zombie::rws::EvaluationPoint<T, DIM>>(solvers_m, ("rws_evaluation_point" + typeStr).c_str())
        .def(nb::init<const zombie::Vector<DIM>&, const zombie::Vector<DIM>&, zombie::SampleType, float, float>(),
            "pt"_a, "normal"_a, "type"_a, "dist_to_absorbing_boundary"_a, "dist_to_reflecting_boundary"_a)
        .def("get_estimated_solution", &zombie::rws::EvaluationPoint<T, DIM>::getEstimatedSolution,
            "Returns estimated solution.",
            "n_absorbing_boundary_samples"_a, "n_absorbing_boundary_normal_aligned_samples"_a,
            "n_reflecting_boundary_samples"_a, "n_reflecting_boundary_normal_aligned_samples"_a,
            "n_source_samples"_a)
        .def("reset", &zombie::rws::EvaluationPoint<T, DIM>::reset,
            "Resets state.")
        .def_rw("pt", &zombie::rws::EvaluationPoint<T, DIM>::pt)
        .def_rw("normal", &zombie::rws::EvaluationPoint<T, DIM>::normal)
        .def_rw("type", &zombie::rws::EvaluationPoint<T, DIM>::type)
        .def_rw("dist_to_absorbing_boundary", &zombie::rws::EvaluationPoint<T, DIM>::distToAbsorbingBoundary)
        .def_rw("dist_to_reflecting_boundary", &zombie::rws::EvaluationPoint<T, DIM>::distToReflectingBoundary);

    nb::bind_vector<RWSEvaluationPointList<T, DIM>>(solvers_m, ("rws_evaluation_point" + typeStr + "_list").c_str());

    nb::class_<zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>>>(
        solvers_m, ("reverse_walk_on_stars" + typeStr).c_str())
        .def(nb::init<const zombie::GeometricQueries<DIM>&,
                      std::shared_ptr<zombie::BoundarySampler<T, DIM>>,
                      std::shared_ptr<zombie::BoundarySampler<T, DIM>>,
                      std::shared_ptr<zombie::DomainSampler<T, DIM>>>(),
            "geometric_queries"_a, "absorbing_boundary_sampler"_a, "reflecting_boundary_sampler"_a, "domain_sampler"_a)
        .def("generate_samples",
            &zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>>::generateSamples,
            "Generates boundary and domain samples.",
            "absorbing_boundary_sample_count"_a, "reflecting_boundary_sample_count"_a, "domain_sample_count"_a,
            "normal_offset_for_absorbing_boundary"_a, "solve_double_sided"_a)
        .def("solve",
            &zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>>::solve,
            "Solves the PDE using the reverse walk on stars algorithm.",
            "pde"_a, "walk_settings"_a, "normal_offset_for_absorbing_boundary"_a, "radius_clamp"_a,
            "kernel_regularization"_a, "eval_pts"_a, "updated_eval_pt_locations"_a=true,
            "run_single_threaded"_a=false, "report_progress"_a.none())
        .def("get_absorbing_boundary_sample_count",
            &zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>>::getAbsorbingBoundarySampleCount,
            "Returns the number of absorbing boundary sample points.",
            "return_boundary_normal_aligned"_a=false)
        .def("get_reflecting_boundary_sample_count",
            &zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>>::getReflectingBoundarySampleCount,
            "Returns the number of reflecting boundary sample points.",
            "return_boundary_normal_aligned"_a=false)
        .def("get_domain_sample_count",
            &zombie::rws::ReverseWalkOnStarsSolver<T, DIM, zombie::NearestNeighborFinder<DIM>>::getDomainSampleCount,
            "Returns the number of domain sample points.");
}

template <typename T, size_t DIM>
void bindKelvinTransform(nb::module_ m, std::string typeStr)
{
    nb::module_ solvers_m = m.def_submodule("solvers", "Solvers module");

    nb::class_<zombie::KelvinTransform<T, DIM>>(solvers_m, ("kelvin_transform" + typeStr).c_str())
        .def(nb::init<const zombie::Vector<DIM>&>(),
            "origin"_a=zombie::Vector<DIM>::Zero())
        .def("set_origin", &zombie::KelvinTransform<T, DIM>::setOrigin,
            "Sets the origin of the Kelvin transform.",
            "origin"_a)
        .def("get_origin", &zombie::KelvinTransform<T, DIM>::getOrigin,
            "Returns the origin of the Kelvin transform.")
        .def("transform_point", &zombie::KelvinTransform<T, DIM>::transformPoint,
            "Applies the Kelvin transform to a point in the exterior domain.",
            "x"_a)
        .def("transform_pde", &zombie::KelvinTransform<T, DIM>::transformPde,
            "Sets up the PDE for the inverted domain given the PDE for the exterior domain.",
            "pde_exterior_domain"_a, "pde_inverted_domain"_a)
        .def("transform_solution_estimate", &zombie::KelvinTransform<T, DIM>::transformSolutionEstimate,
            "Returns the estimated solution in the exterior domain, given the solution estimate at a transformed point.",
            "V"_a, "y"_a)
        .def("transform_points", &zombie::KelvinTransform<T, DIM>::transformPoints,
            "Applies the Kelvin transform to a set of points in the exterior domain.",
            "points"_a, "transformed_points"_a)
        .def("compute_robin_coefficients", &zombie::KelvinTransform<T, DIM>::computeRobinCoefficients,
            "Computes the modified Robin coefficients for the transformed reflecting boundary represented by line segments in 2D and triangles in 3D:\na boundary with Neumann conditions typically has non-zero Robin coefficients on the inverted domain in 3D, but it continues to have\nNeumann conditions in 2D.",
            "transformed_points"_a, "indices"_a, "min_robin_coeff_values"_a, "max_robin_coeff_values"_a,
            "transformed_min_robin_coeff_values"_a, "transformed_max_robin_coeff_values"_a);
}

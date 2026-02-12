cmake_minimum_required(VERSION 3.15...3.27)
project(zombie)

option(ZOMBIE_BUILD_DEMO "Build demo" ON)
option(ZOMBIE_BUILD_BINDINGS "Build bindings" OFF)

################################################################################
if(ZOMBIE_BUILD_BINDINGS)
    if (CMAKE_VERSION VERSION_LESS 3.18)
        set(DEV_MODULE Development)
    else()
        set(DEV_MODULE Development.Module)
    endif()
    find_package(Python 3.8
        REQUIRED COMPONENTS Interpreter ${DEV_MODULE}
        OPTIONAL_COMPONENTS Development.SABIModule)
endif()

# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    message(STATUS "Setting build type to 'Release' as none was specified.")
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
    add_definitions(-DNDEBUG)
endif()

################################################################################
# add dependencies
get_directory_property(hasParent PARENT_DIRECTORY)

set(TBB_TEST          OFF CACHE BOOL " " FORCE)
set(TBB_INSTALL       OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)

if(hasParent)
    set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include PARENT_SCOPE)
    set(ZOMBIE_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include PARENT_SCOPE)
else()
    set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
    set(ZOMBIE_NANOFLANN_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/nanoflann/include)
endif()

set(FCPW_USE_ENOKI OFF CACHE BOOL "Enable CPU vectorization" FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fcpw)

list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/pcg32)
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_TBB_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_NANOFLANN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_EIGEN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_ENOKI_INCLUDES})

################################################################################
# build library
if(MSVC)
    file(GLOB_RECURSE ZOMBIE_HEADER_FILES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/include/zombie/*.h)
    add_library(${PROJECT_NAME} INTERFACE ${ZOMBIE_HEADER_FILES})
    source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR}/include/zombie FILES ${ZOMBIE_HEADER_FILES})
else()
    add_library(${PROJECT_NAME} INTERFACE)
endif()

target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include> ${ZOMBIE_DEPS_INCLUDES})
target_link_libraries(${PROJECT_NAME} INTERFACE fcpw TBB::tbb)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

################################################################################
# build demo
if(ZOMBIE_BUILD_DEMO)
    add_subdirectory(demo)
endif()

# build Python bindings
if(ZOMBIE_BUILD_BINDINGS)
    add_subdirectory(deps/nanobind)
    add_subdirectory(python)
endif()

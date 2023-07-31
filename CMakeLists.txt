cmake_minimum_required(VERSION 3.1.0)
project(zombie)

option(ZOMBIE_BUILD_DEMO "Build zombie demo" ON)

################################################################################
# submodule check
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
	"MinSizeRel" "RelWithDebInfo")
endif()

if(CMAKE_BUILD_TYPE MATCHES RELEASE)
	add_definitions(-DNDEBUG)
endif()

################################################################################
# add dependencies
get_directory_property(hasParent PARENT_DIRECTORY)

set(TBB_BUILD_SHARED          ON  CACHE BOOL " " FORCE)
set(TBB_BUILD_STATIC          OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TESTS           OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC       OFF CACHE BOOL " " FORCE)
set(TBB_BUILD_TBBMALLOC_PROXY OFF CACHE BOOL " " FORCE)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb)
set_property(TARGET tbb tbb_def_files PROPERTY FOLDER "dependencies")
if(hasParent)
	set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include PARENT_SCOPE)
else()
	set(ZOMBIE_TBB_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/tbb/include)
endif()

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/fcpw)

list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/bessel)
list(APPEND ZOMBIE_DEPS_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/deps/pcg32)
list(APPEND ZOMBIE_DEPS_INCLUDES ${ZOMBIE_TBB_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_EIGEN_INCLUDES})
list(APPEND ZOMBIE_DEPS_INCLUDES ${FCPW_ENOKI_INCLUDES})

################################################################################
# build library
add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE $<BUILD_INTERFACE:${${PROJECT_NAME}_SOURCE_DIR}/include> ${ZOMBIE_DEPS_INCLUDES})
target_link_libraries(${PROJECT_NAME} INTERFACE fcpw tbb)
target_compile_features(${PROJECT_NAME} INTERFACE cxx_std_17)

################################################################################
# build demo
if(ZOMBIE_BUILD_DEMO)
	add_subdirectory(demo)
endif()

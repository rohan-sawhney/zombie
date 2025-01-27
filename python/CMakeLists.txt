cmake_minimum_required(VERSION 3.15...3.27)
project(zombie-bindings)

set(ZOMBIE_BINDINGS_SRC_FILES
    "zombie_bindings.cpp"
    "zombie_bindings.h"
)

nanobind_add_module(py NOMINSIZE STABLE_ABI NB_STATIC ${ZOMBIE_BINDINGS_SRC_FILES})
target_link_libraries(py PRIVATE zombie)

install(TARGETS py LIBRARY DESTINATION zombie)

if(MSVC)
    target_compile_options(py PRIVATE /bigobj)
endif()

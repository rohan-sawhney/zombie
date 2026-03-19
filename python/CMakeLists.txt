cmake_minimum_required(VERSION 3.15...3.27)
project(zombie-bindings)

set(ZOMBIE_BINDINGS_SRC_FILES
    "zombie_bindings.cpp"
    "zombie_bindings.h"
)

nanobind_add_module(_zombie NOMINSIZE STABLE_ABI NB_STATIC ${ZOMBIE_BINDINGS_SRC_FILES})
target_link_libraries(_zombie PRIVATE zombie)

install(TARGETS _zombie LIBRARY DESTINATION zombie)

if(MSVC)
    target_compile_options(_zombie PRIVATE /bigobj)
endif()

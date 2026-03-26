cmake_minimum_required(VERSION 3.15...3.27)
project(zombie-bindings)

set(ZOMBIE_BINDINGS_SRC_FILES
    "zombie_bindings.cpp"
    "zombie_bindings.h"
)

nanobind_add_module(_zombie NOMINSIZE STABLE_ABI NB_STATIC ${ZOMBIE_BINDINGS_SRC_FILES})
target_link_libraries(_zombie PRIVATE zombie)

# ensure the extension can find adjacent shared libs at runtime
if(UNIX AND NOT APPLE)
    # use $ORIGIN so the module can load libs placed next to it after install
    set_target_properties(_zombie PROPERTIES BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")
elseif(APPLE)
    # use @loader_path on macOS
    set_target_properties(_zombie PROPERTIES BUILD_RPATH "@loader_path" INSTALL_RPATH "@loader_path")
endif()

install(TARGETS _zombie
    LIBRARY DESTINATION zombie
    RUNTIME DESTINATION zombie
)

if(MSVC)
    target_compile_options(_zombie PRIVATE /bigobj)
endif()

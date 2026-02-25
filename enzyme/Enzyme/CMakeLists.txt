get_target_property(TBL_LINKED_LIBS LLVMSupport INTERFACE_LINK_LIBRARIES)
if (NOT TBL_LINKED_LIBS)
else()
list(REMOVE_ITEM TBL_LINKED_LIBS "ZLIB::ZLIB")
set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES ${TBL_LINKED_LIBS})
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_LINK_COMPONENTS Demangle)

file(GLOB REACTANT_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cpp"
)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (REACTANT_ENABLE_PLUGINS)
        # on windows `PLUGIN_TOOL` doesn't link against LLVM.dll
        if ((WIN32 OR CYGWIN) AND LLVM_LINK_LLVM_DYLIB)
        add_llvm_library( ClangReactant-${LLVM_VERSION_MAJOR}
                    ${REACTANT_SRC} Clang/EnzymeClang.cpp
                    Clang/EnzymePassLoader.cpp
                    PARTIAL_SOURCES_INTENDED
                    MODULE
                    DEPENDS
                    intrinsics_gen
            	LINK_COMPONENTS
            	LLVM
                )
        target_link_libraries(ClangReactant-${LLVM_VERSION_MAJOR}
            PRIVATE
            EnzymeReactantSupport-${LLVM_VERSION_MAJOR}
        )
        target_compile_definitions(ClangReactant-${LLVM_VERSION_MAJOR} PUBLIC ENZYME_RUNPASS)
        else()
        add_llvm_library( ClangReactant-${LLVM_VERSION_MAJOR}
                     ${REACTANT_SRC} Clang/EnzymeClang.cpp
                     Clang/EnzymePassLoader.cpp
                     PARTIAL_SOURCES_INTENDED
                     MODULE
                     DEPENDS
                     intrinsics_gen
                     PLUGIN_TOOL
                     clang
                 )
        target_link_libraries(ClangReactant-${LLVM_VERSION_MAJOR}
            PRIVATE
            EnzymeReactantSupport-${LLVM_VERSION_MAJOR}
        )
        target_compile_definitions(ClangReactant-${LLVM_VERSION_MAJOR} PUBLIC ENZYME_RUNPASS)
        endif()
endif()

if (TARGET ClangReactant-${LLVM_VERSION_MAJOR})
    set_target_properties(ClangReactant-${LLVM_VERSION_MAJOR} PROPERTIES
        PREFIX "lib")
endif()

if (APPLE AND TARGET ClangReactant-${LLVM_VERSION_MAJOR})
    set_target_properties(ClangReactant-${LLVM_VERSION_MAJOR} PROPERTIES
        LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

add_library(ClangReactantFlags INTERFACE)
target_compile_options(ClangReactantFlags INTERFACE "-fplugin=$<TARGET_FILE:ClangReactant-${LLVM_VERSION_MAJOR}>")

# this custom target exists to prevent CMake from incorrectly assuming that
# targets that link depend on ClangReactant-XX can be built at the same time or
# before ClangReactant-XX has finished.
add_custom_target(ClangReactantDummy "" DEPENDS ClangReactant-${LLVM_VERSION_MAJOR})
add_dependencies(ClangReactantFlags ClangReactantDummy)

install(TARGETS ClangReactantFlags EXPORT ReactantTargets)

install(TARGETS ClangReactant-${LLVM_VERSION_MAJOR}
    EXPORT ReactantTargets
    LIBRARY DESTINATION lib COMPONENT shlib
    COMPONENT dev)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/enzyme" DESTINATION "include") 

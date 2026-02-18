#if(WIN32 OR CYGWIN)
#  set(LLVM_LINK_COMPONENTS Core Support)
#endif()

if ("${Enzyme_TABLEGEN_EXE}" STREQUAL "")
  set(Enzyme_TABLEGEN_EXE enzyme-tblgen)
endif()

get_target_property(TBL_LINKED_LIBS LLVMSupport INTERFACE_LINK_LIBRARIES)
if (NOT TBL_LINKED_LIBS)
else()
list(REMOVE_ITEM TBL_LINKED_LIBS "ZLIB::ZLIB")
set_property(TARGET LLVMSupport PROPERTY INTERFACE_LINK_LIBRARIES ${TBL_LINKED_LIBS})
endif()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

set(LLVM_LINK_COMPONENTS Demangle)

file(GLOB ENZYME_SRC CONFIGURE_DEPENDS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    "*.cpp"
)
list(REMOVE_ITEM ENZYME_SRC "api.cpp")
list(REMOVE_ITEM ENZYME_SRC "eopt.cpp")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (ENZYME_ENABLE_PLUGINS)
        # on windows `PLUGIN_TOOL` doesn't link against LLVM.dll
        if ((WIN32 OR CYGWIN) AND LLVM_LINK_LLVM_DYLIB)
                add_llvm_library( ClangEnzyme-${LLVM_VERSION_MAJOR}
                    ${ENZYME_SRC} Clang/EnzymeClang.cpp
                    Clang/EnzymePassLoader.cpp
                    PARTIAL_SOURCES_INTENDED
                    MODULE
                    DEPENDS
                    intrinsics_gen
            	LINK_COMPONENTS
            	LLVM
                )
                    target_compile_definitions(ClangEnzyme-${LLVM_VERSION_MAJOR} PUBLIC ENZYME_RUNPASS)
        else()
                 add_llvm_library( ClangEnzyme-${LLVM_VERSION_MAJOR}
                     ${ENZYME_SRC} Clang/EnzymeClang.cpp
                     Clang/EnzymePassLoader.cpp
                     PARTIAL_SOURCES_INTENDED
                     MODULE
                     DEPENDS
                     intrinsics_gen
                     PLUGIN_TOOL
                     clang
                 )
             target_compile_definitions(ClangEnzyme-${LLVM_VERSION_MAJOR} PUBLIC ENZYME_RUNPASS)
        endif()
endif()

if (${ENZYME_STATIC_LIB})
    add_llvm_library( EnzymeStatic-${LLVM_VERSION_MAJOR}
        ${ENZYME_SRC}
        PARTIAL_SOURCES_INTENDED
        STATIC
        DEPENDS
        intrinsics_gen
    )
endif()

if (${ENZYME_EXTERNAL_SHARED_LIB})
    add_library( Enzyme-${LLVM_VERSION_MAJOR}
        SHARED
        ${ENZYME_SRC}
    )
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} intrinsics_gen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} BinopDerivativesIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} IntrinsicDerivativesIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} CallDerivativesIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} InstructionDerivativesIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} BlasDerivativesIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} BlasDeclarationsIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} BlasTAIncGen)
    add_dependencies(Enzyme-${LLVM_VERSION_MAJOR} BlasDiffUseIncGen)
    target_link_libraries(Enzyme-${LLVM_VERSION_MAJOR} LLVM)
    install(TARGETS Enzyme-${LLVM_VERSION_MAJOR}
        EXPORT EnzymeTargets
        LIBRARY DESTINATION lib COMPONENT shlib
        PUBLIC_HEADER DESTINATION "${INSTALL_INCLUDE_DIR}/Enzyme"
        COMPONENT dev)
endif()

if (APPLE)
set_target_properties(ClangEnzyme-${LLVM_VERSION_MAJOR} PROPERTIES
        LINK_FLAGS "-Wl,-flat_namespace -Wl,-undefined -Wl,suppress")
endif()

add_library(ClangEnzymeFlags INTERFACE)
target_compile_options(ClangEnzymeFlags INTERFACE "-fplugin=$<TARGET_FILE:ClangEnzyme-${LLVM_VERSION_MAJOR}>")

# this custom target exists to prevent CMake from incorrectly assuming that
# targets that link depend on ClangEnzyme-XX can be built at the same time or
# before ClangEnzyme-XX has finished.
add_custom_target(ClangEnzymeDummy "" DEPENDS ClangEnzyme-${LLVM_VERSION_MAJOR})
add_dependencies(ClangEnzymeFlags ClangEnzymeDummy)

install(TARGETS ClangEnzymeFlags EXPORT EnzymeTargets)

install(TARGETS ClangEnzyme-${LLVM_VERSION_MAJOR}
    EXPORT EnzymeTargets
    LIBRARY DESTINATION lib COMPONENT shlib
    COMPONENT dev)
    install(DIRECTORY "${PROJECT_SOURCE_DIR}/include/enzyme" DESTINATION "include")

    

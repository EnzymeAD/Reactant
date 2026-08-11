cmake_minimum_required(VERSION 3.13)
project(Reactant)

include(CMakePackageConfigHelpers)
include(CheckIncludeFile)
include(CheckIncludeFileCXX)
include(FetchContent)

set(ENZYME_MAJOR_VERSION 0)
set(ENZYME_MINOR_VERSION 0)
set(ENZYME_PATCH_VERSION 79)
set(ENZYME_VERSION
  ${ENZYME_MAJOR_VERSION}.${ENZYME_MINOR_VERSION}.${ENZYME_PATCH_VERSION})

add_definitions(-DENZYME_VERSION_MAJOR=${ENZYME_MAJOR_VERSION})
add_definitions(-DENZYME_VERSION_MINOR=${ENZYME_MINOR_VERSION})
add_definitions(-DENZYME_VERSION_PATCH=${ENZYME_PATCH_VERSION})

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

if (NOT DEFINED REACTANT_CONFIGURED_WITH_PRESETS)
  set(CMAKE_CXX_FLAGS "-Wall -fno-rtti ${CMAKE_CXX_FLAGS} -Werror=unused-variable -Werror=dangling-else -Werror=unused-but-set-variable -Werror=return-type -Werror=nonnull -Werror=unused-result -Werror=reorder -Werror=switch")
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "-O2 -g -ggdb")
  set(CMAKE_CXX_FLAGS_RELEASE "-O2")
  set(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -ggdb -fno-omit-frame-pointer")
endif()

#SET(CMAKE_CXX_FLAGS_DEBUG  "-O0 -g -fno-omit-frame-pointer -fsanitize=address")
#SET(CMAKE_LINKER_FLAGS_DEBUG "${CMAKE_LINKER_FLAGS_DEBUG} -fno-omit-frame-pointer -fsanitize=address")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(REACTANT_ENABLE_PLUGINS "Enable Clang/LLD/Opt plugins" ON)
option(REACTANT_CLANG "Build reactant clang plugin" ON)
option(REACTANT_WARN_COMPILER "Warn if enzyme detects potentially incompatible compiler" ON)
set(REACTANT_SRC_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(REACTANT_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})
list(APPEND CMAKE_MODULE_PATH "${REACTANT_SRC_DIR}/cmake/modules")

set(LLVM_SHLIBEXT "${CMAKE_SHARED_MODULE_SUFFIX}")
if (APPLE)
    set(LLVM_SHLIBEXT ".dylib")
endif()
message( LLVM_SHLIBEXT = ${LLVM_SHLIBEXT} )

set(LLVM_MAIN_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

get_filename_component(LLVM_ABSOLUTE_DIR
    "${LLVM_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

if (NOT EXISTS "${LLVM_DIR}")
  message("Looking for LLVM_DIR at ${LLVM_DIR}")
  message(SEND_ERROR "The given LLVM_DIR does not exist. Typo?")
endif()

set(LLVM_DIR "${LLVM_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)

if (EXISTS "${LLVM_DIR}/lib/cmake/llvm/LLVMConfig.cmake")
  set(LLVM_DIR "${LLVM_DIR}/lib/cmake/llvm")
endif()

message("LLVM ABS DIR " ${LLVM_DIR})

message("CMAKE_PREFIX_PATH " ${CMAKE_PREFIX_PATH})

find_package(LLVM REQUIRED CONFIG)

if (NOT LLVM_ENABLE_PLUGINS)
  message("LLVM_ENABLE_PLUGINS OFF")
    message("Setting REACTANT_ENABLE_PLUGINS OFF")
    set(REACTANT_ENABLE_PLUGINS OFF)
endif()

if (NOT REACTANT_ENABLE_PLUGINS)
    message("REACTANT_ENABLE_PLUGINS OFF")
    message(SEND_ERROR "You have selected a build without creating the Reactant plugin -- and nothing will be built.")
endif()

if (NOT DEFINED LLVM_EXTERNAL_LIT)
  if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
      message("found llvm match ${CMAKE_MATCH_1} dir ${LLVM_DIR}")
    if (EXISTS ${LLVM_DIR}/../../../bin/llvm-lit)
      set(LLVM_EXTERNAL_LIT ${LLVM_DIR}/../../../bin/llvm-lit)
    else()
      set(LLVM_EXTERNAL_LIT lit)
    endif()
  else()
    if (EXISTS ${LLVM_DIR}/bin/llvm-lit)
      set(LLVM_EXTERNAL_LIT ${LLVM_DIR}/bin/llvm-lit)
    else()
      set(LLVM_EXTERNAL_LIT lit)
    endif()
  endif()
endif()

get_filename_component(LLVM_ABSOLUTE_LIT
                       "${LLVM_EXTERNAL_LIT}"
                       REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")

                   set(LLVM_EXTERNAL_LIT "${LLVM_ABSOLUTE_LIT}" CACHE FILEPATH "a" FORCE)
message("found llvm lit " ${LLVM_EXTERNAL_LIT})

list(INSERT CMAKE_PREFIX_PATH 0 "${LLVM_DIR}")
message("LLVM dir ${LLVM_DIR}")
if (REACTANT_CLANG)
if (DEFINED Clang_DIR)
    get_filename_component(Clang_ABSOLUTE_DIR
      "${Clang_DIR}"
    REALPATH BASE_DIR "${CMAKE_BINARY_DIR}")
set(Clang_DIR "${Clang_ABSOLUTE_DIR}" CACHE FILEPATH "b" FORCE)
  list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
  message("clang dir defined ${Clang_DIR}")
else()
  if(LLVM_DIR MATCHES ".*/cmake/llvm/?$")
    if (EXISTS ${LLVM_DIR}/../clang/../../libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR}/../clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/?$")
    if (EXISTS ${LLVM_DIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR}/lib/cmake/clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  elseif(LLVM_DIR MATCHES ".*/llvm-([0-9]+)/cmake/?$")
      get_filename_component(LLDIR "${LLVM_DIR}" DIRECTORY)
    if (EXISTS ${LLDIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLDIR}/lib/cmake/clang)
     list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  else()
    if (EXISTS ${LLVM_DIR}/lib/libclangBasic.a)
      set(Clang_DIR ${LLVM_DIR})
      list(INSERT CMAKE_PREFIX_PATH 0 "${Clang_DIR}")
    endif()
  endif()
  message("clang dir from llvm ${Clang_DIR}")
endif()
else()
endif()

list(APPEND CMAKE_MODULE_PATH "${LLVM_CMAKE_DIR}")
include(AddLLVM)

message("clang dir ${Clang_DIR}")
message("mlir dir ${MLIR_DIR}")

if (DEFINED Clang_DIR)
    find_package(Clang REQUIRED CONFIG PATHS ${Clang_DIR} NO_DEFAULT_PATH)
if (${Clang_FOUND})
    include_directories(${CLANG_INCLUDE_DIRS})
    message("clang inc dir ${CLANG_INCLUDE_DIRS}")
    add_definitions(${CLANG_DEFINITIONS})
endif()
else()
    set(Clang_FOUND 0)
endif()
message("found Clang ${Clang_FOUND}")

if (DEFINED MLIR_DIR)
    find_package(MLIR REQUIRED CONFIG PATHS ${MLIR_DIR} NO_DEFAULT_PATH)
    if (${MLIR_FOUND})
        include_directories(${MLIR_INCLUDE_DIRS})
        message("MLIR inc dir ${MLIR_INCLUDE_DIRS}")
        add_definitions(${MLIR_DEFINITIONS})
        list(APPEND CMAKE_MODULE_PATH "${MLIR_CMAKE_DIR}")
        include(TableGen)
        include(AddMLIR)
    endif()
else()
    set(MLIR_FOUND 0)
endif()
message("found MLIR ${MLIR_FOUND}")
# include(AddClang)

add_definitions(${LLVM_DEFINITIONS})
include_directories(${LLVM_INCLUDE_DIRS})
message("LLVM_INSTALL_PREFIX: ${LLVM_INSTALL_PREFIX}")
message("LLVM_INCLUDE_DIRS: ${LLVM_INCLUDE_DIRS}")
message("found LLVM definitions " ${LLVM_DEFINITIONS})
message("found LLVM version " ${LLVM_VERSION_MAJOR})

if(LLVM_VERSION_MAJOR LESS 15)
  message(FATAL_ERROR "LLVM version < 15 is not supported")
endif()

# Offer the user the choice of overriding the installation directories
set(INSTALL_INCLUDE_DIR include CACHE PATH "Installation directory for header files")
if(WIN32 AND NOT CYGWIN)
  set(DEF_INSTALL_CMAKE_DIR CMake)
else()
  set(DEF_INSTALL_CMAKE_DIR lib/cmake/Enzyme)
endif()
set(INSTALL_CMAKE_DIR ${DEF_INSTALL_CMAKE_DIR} CACHE PATH
  "Installation directory for CMake files")


list(GET LLVM_INCLUDE_DIRS 0 LLVM_IDIR)
message("first llvm include directory" ${LLVM_IDIR})

file(READ ${LLVM_IDIR}/llvm/Analysis/ScalarEvolution.h INPUT_TEXT)
string(REPLACE private public INPUT_TEXT "${INPUT_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolution.h" "${INPUT_TEXT}")


file(READ ${LLVM_IDIR}/llvm/Analysis/TargetLibraryInfo.h INPUT_TEXT)
string(REPLACE "class TargetLibraryInfo {" "class TargetLibraryInfo {public:" INPUT_TEXT "${INPUT_TEXT}")
file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/TargetLibraryInfo.h" "${INPUT_TEXT}")

file(READ ${LLVM_IDIR}/llvm/Transforms/Utils/ScalarEvolutionExpander.h INPUT_TEXT)

find_library(MPFR_LIB_PATH mpfr)
CHECK_INCLUDE_FILE("mpfr.h" HAS_MPFR_H)
message("MPFR lib: " ${MPFR_LIB_PATH})
message("MPFR header: " ${HAS_MPFR_H})

file(WRITE "${CMAKE_CURRENT_BINARY_DIR}/include/SCEV/ScalarEvolutionExpander.h" "${INPUT_TEXT}")

include_directories("${CMAKE_CURRENT_BINARY_DIR}/include")

set(UPSTREAM_ENZYME_BUILD_DIR "")

# Upstream Enzyme is required for EnzymeReactantSupport.
FetchContent_Declare(
  upstream_enzyme
  GIT_REPOSITORY https://github.com/EnzymeAD/Enzyme.git
  GIT_TAG 0338786b2c26f9aed9d1c9483a241b2284722d74
  SOURCE_SUBDIR enzyme
)

# Build only upstream Reactant support library.
set(ENZYME_ENABLE_REACTANT ON CACHE BOOL "" FORCE)
set(ENZYME_ENABLE_PLUGINS OFF CACHE BOOL "" FORCE)
set(ENZYME_CLANG OFF CACHE BOOL "" FORCE)
set(ENZYME_STATIC_LIB OFF CACHE BOOL "" FORCE)
set(ENZYME_EXTERNAL_SHARED_LIB OFF CACHE BOOL "" FORCE)
set(ENZYME_MLIR OFF CACHE BOOL "" FORCE)
set(ENZYME_ENABLE_BENCHMARKS OFF CACHE BOOL "" FORCE)
set(ENZYME_BC_LOADER OFF CACHE BOOL "" FORCE)

FetchContent_MakeAvailable(upstream_enzyme)
FetchContent_GetProperties(upstream_enzyme)
set(UPSTREAM_ENZYME_BUILD_DIR "${upstream_enzyme_BINARY_DIR}")
message(STATUS "Upstream Enzyme build dir: ${UPSTREAM_ENZYME_BUILD_DIR}")

add_subdirectory(Enzyme)

# Make relative paths absolute (needed later on)
foreach(p LIB BIN INCLUDE CMAKE)
set(var INSTALL_${p}_DIR)
if(NOT IS_ABSOLUTE "${${var}}")
  set(${var} "${CMAKE_INSTALL_PREFIX}/${${var}}")
endif()
endforeach()

if (REACTANT_ENABLE_PLUGINS)
if (${Clang_FOUND})
export(TARGETS ClangReactant-${LLVM_VERSION_MAJOR}
    APPEND FILE "${PROJECT_BINARY_DIR}/ReactantTargets.cmake")
export(TARGETS ClangReactantFlags
    APPEND FILE "${PROJECT_BINARY_DIR}/ReactantTargets.cmake")

file(GLOB_RECURSE REACTANT_INCLUDES CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/include/*")
set(bundled_includes "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/Clang/bundled_includes.h")
list(APPEND REACTANT_INCLUDES ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle-includes.sh)
add_custom_command(OUTPUT ${bundled_includes}
  COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/scripts/bundle-includes.sh
  ${CMAKE_CURRENT_SOURCE_DIR}/include/enzyme/enzyme
  ${bundled_includes}
  DEPENDS ${REACTANT_INCLUDES}
  COMMENT "Bundling includes"
  VERBATIM
)
add_custom_target(bundled_includes_target DEPENDS ${bundled_includes})
add_dependencies(ClangReactant-${LLVM_VERSION_MAJOR} bundled_includes_target)
target_include_directories(ClangReactant-${LLVM_VERSION_MAJOR} PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/Enzyme/Clang/")
endif()
endif()

export(PACKAGE Reactant)

set(CONF_LLVM_VERSION_MAJOR ${LLVM_VERSION_MAJOR})
set(CONF_LLVM_VERSION_MINOR ${LLVM_VERSION_MINOR})
set(CONF_LLVM_VERSION_PATCH ${LLVM_VERSION_PATCH})
set(CONF_LLVM_DIR ${LLVM_DIR})
set(CONF_LLVM_BINARY_DIR ${LLVM_BINARY_DIR})
if (${Clang_FOUND})
    get_property(CONF_CLANG_EXE TARGET clang PROPERTY LOCATION)
else()
    set(CONF_CLANG_EXE "")
endif()

set(CONF_INCLUDE_DIRS lib)
configure_file(cmake/ReactantConfig.cmake.in
     "${PROJECT_BINARY_DIR}/ReactantConfig.cmake" @ONLY)

set(CONF_INCLUDE_DIRS lib)
configure_file(cmake/ReactantConfig.cmake.in
     "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ReactantConfig.cmake" @ONLY)

configure_file(cmake/ReactantConfigVersion.cmake.in
     "${PROJECT_BINARY_DIR}/ReactantConfigVersion.cmake" @ONLY)

install(FILES
     "${PROJECT_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/ReactantConfig.cmake"
     "${PROJECT_BINARY_DIR}/ReactantConfigVersion.cmake"
     DESTINATION "${INSTALL_CMAKE_DIR}" COMPONENT dev)

install(EXPORT ReactantTargets
  DESTINATION "${INSTALL_CMAKE_DIR}"
  COMPONENT   dev)

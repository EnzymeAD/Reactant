load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")
load("@llvm-project//:vars.bzl", "LLVM_VERSION_MAJOR")
load(":clang_resource_dir.bzl", "copy_clang_resource_dir")

CLANG_REACTANT_PLUGIN = "ClangReactant-{}".format(LLVM_VERSION_MAJOR)

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "ReactantStatic",
    srcs = glob(
        [
            "Enzyme/*.cpp",
            "Enzyme/Clang/EnzymeClang.cpp",
        ],
        exclude = ["Enzyme/ropt.cpp"],
    ),
    hdrs = glob([
        "Enzyme/*.h",
    ]) + ["Enzyme/Clang/bundled_includes.h"],
    copts = [
        "-DENZYME_RUNPASS=1",
        "-DREACTANT_USE_LINKED_RAISE=1",
        "-DENZYME_VERSION_MAJOR=0",
        "-DENZYME_VERSION_MINOR=0",
        "-DENZYME_VERSION_PATCH=79",
        "-Wno-unused-variable",
        "-Wno-return-type",
    ],
    data = ["@llvm-project//clang:builtin_headers_gen"],
    includes = [
        "Enzyme/Clang",  # for bundled_includes.h
    ],
    visibility = ["//visibility:public"],
    deps = [
        "@enzyme//:EnzymePreserveNVVM",
        "@enzyme_ad//:RaiseLib",
        ":bundled-includes",
        "@llvm-project//clang:ast",
        "@llvm-project//clang:basic",
        "@llvm-project//clang:driver",
        "@llvm-project//clang:frontend",
        "@llvm-project//clang:frontend_tool",
        "@llvm-project//clang:lex",
        "@llvm-project//clang:sema",
        "@llvm-project//clang:serialization",
        "@llvm-project//llvm:AggressiveInstCombine",
        "@llvm-project//llvm:Analysis",
        "@llvm-project//llvm:CodeGen",
        "@llvm-project//llvm:Core",
        "@llvm-project//llvm:Demangle",
        "@llvm-project//llvm:FrontendOpenMP",
        "@llvm-project//llvm:IPO",
        "@llvm-project//llvm:IRReader",
        "@llvm-project//llvm:InstCombine",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Scalar",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:Target",
        "@llvm-project//llvm:TargetParser",
        "@llvm-project//llvm:TransformUtils",
        "@llvm-project//llvm:config",
    ],
    alwayslink = 1,
)

# Modified from llvm_driver_cc_binary since that can't be used
# outside of llvm tree
expand_template(
    name = "_gen_reactant-clang",
    out = "reactant-clang-driver.cpp",
    substitutions = {"@TOOL_NAME@": "clang", "@INITLLVM_ARGS@": ", /*InstallPipeSignalExitHandler=*/true, /*NeedsPOSIXUtilitySignalHandling=*/true"},
    template = "@llvm-project//llvm:cmake/modules/llvm-driver-template.cpp.in",
)

cc_binary(
    name = "reactant-opt",
    srcs = ["Enzyme/ropt.cpp"],
    deps = [
        ":ReactantStatic",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:opt-driver",
    ],
)

cc_binary(
    name = "reactant-clang",
    srcs = ["reactant-clang-driver.cpp"],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-error=frame-larger-than=",
        "-Denzyme_clang_main=clang_main",
    ],
    deps = [
        ":ReactantStatic",
        "@llvm-project//clang:clang-driver",
        "@llvm-project//llvm:FrontendOpenMP",
        "@llvm-project//llvm:Support",
    ],
)

cc_binary(
    name = CLANG_REACTANT_PLUGIN,
    linkshared = 1,
    linkstatic = 1,
    deps = [
        ":ReactantStatic",
    ],
)

alias(
    name = "ClangReactantPlugin",
    actual = ":{}".format(CLANG_REACTANT_PLUGIN),
)

genrule(
    name = "bundled-includes",
    srcs = glob(["include/**"]) + ["scripts/bundle-includes.sh"],
    outs = ["Enzyme/Clang/bundled_includes.h"],
    cmd = "$(location :scripts/bundle-includes.sh) $(location :include/enzyme/enzyme) $@",
)

genrule(
    name = "gen_reactant-clang++",
    srcs = [":reactant-clang"],
    outs = ["reactant-clang++"],
    cmd = "cp $< $@",
    output_to_bindir = 1,
)

copy_clang_resource_dir(
    name = "reactant-clang-resource",
    srcs = [
        "@llvm-project//clang:builtin_headers_gen",
        "@llvm_openmp_headers//:omp_headers_staged",
    ],
)

exports_files(["run_lit.sh"])

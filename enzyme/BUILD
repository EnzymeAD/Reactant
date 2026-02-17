load("@bazel_skylib//rules:expand_template.bzl", "expand_template")
load("@llvm-project//mlir:tblgen.bzl", "gentbl_cc_library", "td_library")

licenses(["notice"])

package(
    default_applicable_licenses = [],
    default_visibility = ["//visibility:public"],
)

cc_library(
    name = "EnzymeStatic",
    srcs = glob(
        [
            "Enzyme/*.cpp",
            "Enzyme/Clang/EnzymeClang.cpp",
        ],
        exclude = [
            "Enzyme/eopt.cpp",
            "Enzyme/PreserveNVVM.cpp",
            "Enzyme/JLInstSimplify.cpp",
            ],
    ),
    hdrs = glob([
        "Enzyme/*.h",
    ]) + ["Enzyme/Clang/bundled_includes.h"],
    copts = [
        "-DENZYME_RUNPASS=1",
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
        "@enzyme//:EnzymeFrontendSupport",
        "@enzyme//:binop-derivatives",
        "@enzyme//:blas-attributor",
        "@enzyme//:blas-derivatives",
        "@enzyme//:blas-diffuseanalysis",
        "@enzyme//:blas-typeanalysis",
        ":bundled-includes",
        "@enzyme//:call-derivatives",
        "@enzyme//:inst-derivatives",
        "@enzyme//:intr-derivatives",
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
    name = "_gen_enzyme-clang",
    out = "enzyme-clang-driver.cpp",
    substitutions = {"@TOOL_NAME@": "clang"},
    template = "@llvm-project//llvm:cmake/modules/llvm-driver-template.cpp.in",
)

cc_binary(
    name = "enzyme-clang",
    srcs = ["enzyme-clang-driver.cpp"],
    copts = [
        "-Wno-implicit-fallthrough",
        "-Wno-error=frame-larger-than=",
        "-Denzyme_clang_main=clang_main",
    ],
    deps = [
        ":EnzymeStatic",
        "@llvm-project//clang:clang-driver",
        "@llvm-project//llvm:Support",
    ],
)

genrule(
    name = "bundled-includes",
    srcs = glob(["include/**"]) + ["scripts/bundle-includes.sh"],
    outs = ["Enzyme/Clang/bundled_includes.h"],
    cmd = "$(location :scripts/bundle-includes.sh) $(location :include/enzyme/enzyme) $@",
)

genrule(
    name = "gen_enzyme-clang++",
    srcs = [":enzyme-clang"],
    outs = ["enzyme-clang++"],
    cmd = "cp $< $@",
    output_to_bindir = 1,
)

cc_binary(
    name = "enzyme-opt",
    srcs = ["Enzyme/eopt.cpp"],
    deps = [
        ":EnzymeStatic",
        "@llvm-project//llvm:Passes",
        "@llvm-project//llvm:Support",
        "@llvm-project//llvm:opt-driver",
    ],
)


exports_files(["run_lit.sh"])

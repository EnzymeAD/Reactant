# llvm_openmp_repo.bzl
def _llvm_openmp_headers_impl(rctx):
    # Navigate to llvm-raw root via CMakeLists.txt (stable anchor in all LLVM versions)
    llvm_root = rctx.path(Label("@llvm-raw//:CMakeLists.txt")).dirname
    omp_include = str(llvm_root) + "/openmp/runtime/src/include"

    rctx.execute(["mkdir", "-p", "staging/include"])

    # Generate omp.h from the .var template
    result = rctx.execute(["sh", "-c", """
        set -e
        sed \
          -e 's/@LIBOMP_VERSION_MAJOR@/5/g' \
          -e 's/@LIBOMP_VERSION_MINOR@/2/g' \
          -e 's/@LIBOMP_VERSION_BUILD@/20231101/g' \
          -e 's/@LIBOMP_LIB_FILE@/libomp.so/g' \
          -e 's/@LIBOMP_HAVE_QUAD_PRECISION@/0/g' \
          -e 's/@LIBOMP_QUAD_PRECISION_TYPEOF_SPECIFIER@//g' \
          -e 's/@LIBOMP_HAVE_HIDDEN_HELPER_TASK@/1/g' \
          -e 's/@LIBOMP_HAVE_OMPT_SUPPORT@/0/g' \
          -e 's/@LIBOMP_OMPT_OPTIONAL@/0/g' \
          -e 's/@LIBOMP_USE_ITT_NOTIFY@/0/g' \
          '{omp_include}/omp.h.var' > staging/include/omp.h
    """.format(omp_include = omp_include)])

    if result.return_code != 0:
        fail("Failed to generate omp.h: " + result.stderr)

    # Copy direct headers — use || true so missing ones don't fail
    # omp-tools.h: OpenMP Tools Interface (OMPT)
    # ompx.h: OpenMP extensions (LLVM-specific, may not exist in older versions)
    rctx.execute(["sh", "-c", """
        set -e
        for h in omp-tools.h ompx.h omp_lib.h; do
            src='{omp_include}/'"$h"
            test -f "$src" && cp "$src" 'staging/include/'"$h" || true
        done
    """.format(omp_include = omp_include)])

    # Emit the BUILD so the staged files are accessible
    rctx.file("BUILD.bazel", """
filegroup(
    name = "omp_headers_staged",
    srcs = glob(["staging/include/**"]),
    visibility = ["//visibility:public"],
)
""")

llvm_openmp_headers = repository_rule(
    implementation = _llvm_openmp_headers_impl,
    attrs = {},
)
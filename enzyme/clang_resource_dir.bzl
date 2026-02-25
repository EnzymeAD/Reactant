def _copy_clang_resource_dir_impl(ctx):
    outs = []
    marker = "staging/include/"
    for src in ctx.files.srcs:
        short = src.short_path
        idx = short.find(marker)
        if idx < 0:
            fail("unexpected staged header path: {}".format(short))
        rel = short[idx + len(marker):]
        out = ctx.actions.declare_file("{}/include/{}".format(ctx.label.name, rel))
        ctx.actions.run_shell(
            inputs = [src],
            outputs = [out],
            arguments = [out.dirname, src.path, out.path],
            mnemonic = "CopyClangResourceHeader",
            progress_message = "copying clang resource header {}".format(rel),
            command = """
set -euo pipefail
mkdir -p "$1"
cp -f "$2" "$3"
""",
        )
        outs.append(out)

    return [DefaultInfo(files = depset(outs))]

copy_clang_resource_dir = rule(
    implementation = _copy_clang_resource_dir_impl,
    attrs = {
        "srcs": attr.label_list(allow_files = True),
    },
)

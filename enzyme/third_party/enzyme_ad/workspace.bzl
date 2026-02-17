"""Loads Enzyme-JAX."""

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")
load(
    "//:workspace.bzl",
    "ENZYMEXLA_COMMIT",
    "ENZYMEXLA_SHA256",
    "OVERRIDE_ENZYMEXLA_PATH",
)

def _enzyme_ad_local_impl(ctx):
    # Local override uses a copied snapshot instead of native.local_repository so we
    # can apply the same patch-label rewrites as the http_archive path.
    # Workflow:
    # - Normal usage: leave OVERRIDE_ENZYMEXLA_PATH empty (http_archive).
    # - Local debugging: set OVERRIDE_ENZYMEXLA_PATH, edit local Enzyme-JAX, then run
    #   `bazel sync --only=enzyme_ad` to refresh this copied snapshot.
    # - If @enzyme inside Enzyme-JAX points to a local path (OVERRIDE_ENZYME_PATH),
    #   edits in that local Enzyme checkout are visible directly; no enzyme_ad resync
    #   is needed unless Enzyme-JAX workspace metadata itself changed.
    copy = ctx.execute(["/bin/bash", "-c", "cp -a \"$1\"/. .", "--", str(ctx.path(ctx.attr.path))])
    if copy.return_code != 0:
        fail("Failed to copy local Enzyme-JAX checkout: {}".format(copy.stderr))
    patch = ctx.execute([
        "/bin/bash",
        "-c",
        """
set -euo pipefail
sed -i.bak0 "s/\\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/:patches/@enzyme_ad\\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/:patches/g" workspace.bzl
sed -i.bak0 "s,//:patches,@enzyme_ad//:patches,g" third_party/*/workspace.bzl
""",
    ])
    if patch.return_code != 0:
        fail("Failed to rewrite Enzyme-JAX patch labels: {}".format(patch.stderr))

_enzyme_ad_local_repository = repository_rule(
    implementation = _enzyme_ad_local_impl,
    attrs = {
        "path": attr.string(mandatory = True),
    },
)

def repo():
    if len(OVERRIDE_ENZYMEXLA_PATH) != 0:
        _enzyme_ad_local_repository(
            name = "enzyme_ad",
            path = OVERRIDE_ENZYMEXLA_PATH,
        )
    else:
        http_archive(
            name = "enzyme_ad",
            patch_cmds = [
                """
sed -i.bak0 "s/\\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/:patches/@enzyme_ad\\\\\\\\\\\\\\\\\\/\\\\\\\\\\\\\\\\\\/:patches/g" workspace.bzl
sed -i.bak0 "s,//:patches,@enzyme_ad//:patches,g" third_party/*/workspace.bzl
""",
            ],
            sha256 = ENZYMEXLA_SHA256,
            strip_prefix = "Enzyme-JAX-" + ENZYMEXLA_COMMIT,
            urls = ["https://github.com/EnzymeAD/Enzyme-JAX/archive/{commit}.tar.gz".format(commit = ENZYMEXLA_COMMIT)],
        )

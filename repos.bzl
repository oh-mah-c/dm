# repos.bzl — non-Bzlmod repository rules for dm
#
# TensorFlow is built from source via build_all.sh (Phase 1).
# The outputs land in src/core/engine/dist/ which this rule symlinks
# into the Bazel external-repo space as @tf_engine_dist.
#
# This means:
#   - build_all.sh builds TF natively from engine/
#   - dm's Bazel graph cc_imports the built .so + source headers
#   - One script, one build, everything is native source

def _tf_engine_dist_impl(ctx):
    dist_path = ctx.attr.dist_path
    ctx.symlink(dist_path + "/lib", "lib")
    ctx.symlink(ctx.attr.src_path, "src")

    # Auto-detect which .so files are present
    ctx.file("BUILD.bazel", content = """\
package(default_visibility = ["//visibility:public"])

load("@rules_cc//cc:cc_import.bzl", "cc_import")

# libtensorflow_cc — the full TF C++ API shared library
# Built from source by build_all.sh → src/core/engine/dist/lib/
cc_import(
    name = "libtensorflow_cc",
    shared_library = "lib/libtensorflow_cc.so.2",
)

# libtensorflow_framework — TF framework primitives (ops, kernels, etc.)
cc_import(
    name = "libtensorflow_framework",
    shared_library = "lib/libtensorflow_framework.so.2",
)

# C++ headers — taken directly from the engine source tree so they always
# match the compiled binary exactly.
cc_library(
    name = "tensorflow_headers",
    hdrs = glob(
        [
            "src/tensorflow/cc/**/*.h",
            "src/tensorflow/core/framework/*.h",
            "src/tensorflow/core/public/*.h",
            "src/tensorflow/core/platform/*.h",
            "src/tensorflow/core/lib/**/*.h",
            "src/third_party/eigen3/**/*.h",
        ],
        allow_empty = True,
    ),
    strip_include_prefix = "src",
)

# :tensorflow — the single dep that any dm module needs
cc_library(
    name = "tensorflow",
    deps = [
        ":libtensorflow_cc",
        ":libtensorflow_framework",
        ":tensorflow_headers",
    ],
)
""")

tf_engine_dist = repository_rule(
    implementation = _tf_engine_dist_impl,
    attrs = {
        "dist_path": attr.string(mandatory = True),
        "src_path":  attr.string(mandatory = True),
    },
)

#!/bin/bash
#
# Rebuild upstream's MultiScaleDeformableAttention CUDA op for sm_80 (A100) AS WELL AS sm_86,
# OUT OF TREE, into third_party/maskdino_control/ops_build/.
#
# Why this is needed. The .so already installed in the clone's venv was built on an sm_86 node
# and carries sm_86 cubin + sm_86 PTX only. PTX is forward-compatible, so it JITs onto sm_89 but
# NOT onto the OLDER sm_80 of an A100. And upstream's `MSDeformAttn.forward` wraps the CUDA call
# in a bare `except:` that falls back to the pure-pytorch core -- so the wrong arch does not
# crash, it silently costs ~10x throughput. Over 88k steps that is 3 days vs 3 weeks.
# `train_control.py::assert_cuda_msda` fails loudly if this build is missing or wrong.
#
# Why out of tree: the clone must stay pristine so docs/MASKDINO.md §7.6 stays reproducible.
# `train_control.py` puts ops_build/ ahead of site-packages on sys.path; nothing else changes.
# The source is upstream's, unmodified, so §7.6 would produce identical numbers either way.
#
# CPU node is fine (FORCE_CUDA=1). ~5 min. The toolchain gymnastics are install.sh's:
# CUDA 11.3's nvcc needs a pre-GCC11 host compiler, which only stack/2025-06 exposes.
set -euo pipefail

REPO=${MASKDINO_ROOT:-/cluster/home/niacobone/MaskDINO}
OPS="$REPO/maskdino/modeling/pixel_decoder/ops"
OUT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/ops_build"
TMP=$(mktemp -d)

module purge
module load stack/2025-06 gcc/8.5.0
CC85="$(command -v gcc)"; CXX85="$(command -v g++)"
module purge
module load stack/2024-06 gcc/12.2.0 cuda/11.3.1 python/3.9 eth_proxy

mkdir -p "$OUT"
export CC="$CC85" CXX="$CXX85" FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST="8.0;8.6"   # A100 + 3090/A6000; keep BOTH so §7.6's 3090 still works

cd "$OPS"
# -b/-t keep every artefact outside the clone.
"$REPO/myenv/bin/python" setup.py build_ext -b "$OUT" -t "$TMP"
rm -rf "$TMP"

echo "=== built ==="
ls -la "$OUT"
command -v cuobjdump >/dev/null && cuobjdump --list-elf "$OUT"/*.so

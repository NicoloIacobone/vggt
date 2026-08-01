"""Test the chunk-tar contract the 1201-scene build relies on (docs/DATASET.md §5.1).

That build never keeps loose files on scratch — scratch is quota'd on FILE COUNT
(1.0M soft / 1.5M hard) and the tree is ~1.26M files. The tree lives in $TMPDIR and
crosses job boundaries only as a compressed tar, so the *only* thing standing between a
resumable build and silent data loss is that the tar round-trips exactly. The failure
mode is quiet: a dropped `.complete` marker makes a finished scene look unbuilt (wasted
re-download), a dropped `.subset_complete` throws away a `.sens` stream, and a dropped
`_qa/stats.json` fails the packing QA gate for a scene that is actually fine.

Asserts, against a synthetic build tree shaped like the real one:
  - every regular file survives tar -> untar, dotfile markers included (the check the
    snapshot job gates its `rm -rf` on);
  - the png/jpg-only count used by the packing gate matches;
  - re-tarring a restored tree is idempotent (the extend job's checkpoint path);
  - markers land where the resume logic looks for them (raw_data/.complete, depth 3),
    which is what `find -mindepth 3 -maxdepth 3 -name .complete` in the SLURM scripts
    counts.

CPU-only, no network, no cluster. Run:
    python legacy/dataset_build/tests/test_chunk_tar_roundtrip.py
"""
import subprocess
import sys
import tempfile
from pathlib import Path

ZSTD_C = "zstd -1 -T0"
ZSTD_D = "zstd -d"


def make_tree(root: Path, scenes: int = 3, converted: int = 2) -> None:
    """Synthetic build tree: scans/<scene>/raw_data/{subset,masks,masks_instance,_qa}."""
    for i in range(scenes):
        rd = root / "scans" / f"scene{i:04d}_00" / "raw_data"
        (rd / "subset").mkdir(parents=True)
        for f in range(0, 15, 5):
            (rd / "subset" / f"{f:05d}.jpg").write_bytes(b"\xff\xd8jpg")
        (rd / ".subset_complete").touch()          # stage-1 marker: always present
        if i < converted:
            for cls in ("wall", "chair"):
                (rd / "masks" / cls).mkdir(parents=True)
                (rd / "masks_instance" / f"{cls}_0").mkdir(parents=True)
                for f in range(0, 15, 5):
                    (rd / "masks" / cls / f"{f:05d}.png").write_bytes(b"\x89PNG")
                    (rd / "masks_instance" / f"{cls}_0" / f"{f:05d}.png").write_bytes(b"\x89PNG")
            (rd / "_qa").mkdir(parents=True)
            (rd / "_qa" / "stats.json").write_text('{"num_instances": 2}')
            (rd / ".complete").touch()             # stage-2 marker: converted scenes only


def files(root: Path) -> set[str]:
    return {str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()}


def tar_create(src_parent: Path, out: Path) -> None:
    subprocess.run(["tar", f"--use-compress-program={ZSTD_C}", "-C", str(src_parent),
                    "-cf", str(out), "scans"], check=True)


def tar_extract(tar: Path, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run(["tar", f"--use-compress-program={ZSTD_D}", "-C", str(dest),
                    "-xf", str(tar)], check=True)


def tar_list(tar: Path) -> list[str]:
    out = subprocess.run(["tar", f"--use-compress-program={ZSTD_D}", "-tf", str(tar)],
                         check=True, capture_output=True, text=True).stdout
    return [ln for ln in out.splitlines() if ln and not ln.endswith("/")]


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        src, tar = tmp / "src", tmp / "chunk.tar.zst"
        make_tree(src)
        src_files = files(src)

        tar_create(src, tar)

        # 1. The snapshot job deletes a 1.26M-file tree on the strength of this count.
        assert len(tar_list(tar)) == len(src_files), \
            f"archive {len(tar_list(tar))} entries vs {len(src_files)} source files"
        print(f"[1/5] archive entry count matches source ({len(src_files)} files) OK")

        # 2. Round-trip must be exact — markers and stats.json, not just the mask bulk.
        dest = tmp / "dest"
        tar_extract(tar, dest)
        assert files(dest) == src_files, \
            f"round-trip differs: missing={src_files - files(dest)}, extra={files(dest) - src_files}"
        markers = [p for p in dest.rglob(".complete")]
        submarkers = [p for p in dest.rglob(".subset_complete")]
        assert len(markers) == 2 and len(submarkers) == 3, \
            f"markers lost: {len(markers)} .complete, {len(submarkers)} .subset_complete"
        print("[2/5] round-trip exact, dotfile markers preserved OK")

        # 3. Markers must sit at the depth the SLURM resume logic greps for
        #    (find -mindepth 3 -maxdepth 3 -name .complete, relative to scans/).
        for m in markers:
            assert m.relative_to(dest).parts[:1] == ("scans",) and m.parent.name == "raw_data", \
                f"marker at unexpected path: {m.relative_to(dest)}"
            assert len(m.relative_to(dest / "scans").parts) == 3, \
                f"marker depth changed — SLURM find -mindepth 3 would miss it: {m}"
        print("[3/5] marker depth matches the resume find OK")

        # 4. The packing gate counts png/jpg only; it must agree across the archive.
        n_tar_pj = sum(1 for e in tar_list(tar) if e.endswith((".png", ".jpg")))
        n_src_pj = sum(1 for f in src_files if f.endswith((".png", ".jpg")))
        assert n_tar_pj == n_src_pj, f"pack gate: archive {n_tar_pj} vs source {n_src_pj}"
        print(f"[4/5] pack-gate png/jpg count matches ({n_src_pj}) OK")

        # 5. The extend job restores, adds scenes, and re-tars every run. Re-tarring a
        #    restored tree must not drift, or the chunk decays across resubmissions.
        tar2 = tmp / "chunk2.tar.zst"
        tar_create(dest, tar2)
        assert sorted(tar_list(tar2)) == sorted(tar_list(tar)), "re-tar of a restored tree drifted"
        dest2 = tmp / "dest2"
        tar_extract(tar2, dest2)
        assert files(dest2) == src_files, "second round-trip lost files"
        print("[5/5] re-tar round-trip idempotent OK")

    print("\nAll chunk-tar round-trip tests passed.")


if __name__ == "__main__":
    if not subprocess.run(["bash", "-c", "command -v zstd"], capture_output=True).returncode == 0:
        print("zstd not found — skipping (this test needs the same zstd the SLURM jobs use)")
        sys.exit(0)
    main()

#!/usr/bin/env python3
"""Build (or load-cached) the full file manifest for lifuguan/InsScene-15K via the HF tree
API, paginating with the Link header (the API caps page size regardless of the requested
`limit`). Also re-checks whether an Aria/ASE directory has appeared at the repo root since
the 2026-08-07 baseline (only processed_infinigen/processed_re10k/processed_scannetpp_v2
existed then) -- that fact changes how any downstream experiment must be labelled, so it is
recorded prominently rather than assumed unchanged.
"""
import json
import re
import subprocess
import sys
from pathlib import Path

REPO = "lifuguan/InsScene-15K"
API_ROOT = f"https://huggingface.co/api/datasets/{REPO}"
SUBDIRS = ["processed_infinigen", "processed_re10k", "processed_scannetpp_v2"]


def curl_json(url):
    p = subprocess.run(["curl", "-sD", "/dev/stderr", url], capture_output=True, text=True)
    headers = p.stderr
    return json.loads(p.stdout), headers


def paginate(url):
    items = []
    while url:
        data, headers = curl_json(url)
        items.extend(data)
        url = None
        m = re.search(r'<([^>]+)>;\s*rel="next"', headers)
        if m:
            url = m.group(1)
    return items


def main():
    out_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("insscene15k_manifest.json")

    root_items, _ = curl_json(f"{API_ROOT}/tree/main")
    root_dirs = sorted(i["path"] for i in root_items if i["type"] == "directory")
    aria_hit = [d for d in root_dirs if re.search(r"aria|ase", d, re.I)]

    meta, _ = curl_json(API_ROOT)
    readme_note = "still being uploaded" in meta.get("description", "")

    manifest = {"repo": REPO, "root_dirs": root_dirs,
                "aria_or_ase_dir_present": bool(aria_hit),
                "aria_or_ase_dirs": aria_hit,
                "readme_says_still_uploading": readme_note,
                "subsets": {}}

    total_files, total_bytes = 0, 0
    for d in SUBDIRS:
        if d not in root_dirs:
            manifest["subsets"][d] = {"present": False}
            continue
        url = f"{API_ROOT}/tree/main/{d}?recursive=true&expand=true"
        items = paginate(url)
        files = [i for i in items if i["type"] == "file"]
        entry = {
            "present": True,
            "n_files": len(files),
            "total_bytes": sum(f["size"] for f in files),
            "files": [{"path": f["path"], "size": f["size"]} for f in files],
        }
        manifest["subsets"][d] = entry
        total_files += len(files)
        total_bytes += entry["total_bytes"]
        print(f"[manifest] {d}: {len(files)} files, {entry['total_bytes']/1e9:.2f} GB",
              file=sys.stderr)

    manifest["total_files"] = total_files
    manifest["total_bytes"] = total_bytes
    out_path.write_text(json.dumps(manifest, indent=2))
    print(f"[manifest] wrote {out_path} : {total_files} files, {total_bytes/1e9:.2f} GB total",
          file=sys.stderr)
    print(f"[manifest] root_dirs={root_dirs} aria_or_ase_present={bool(aria_hit)}",
          file=sys.stderr)


if __name__ == "__main__":
    main()

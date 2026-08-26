"""
The multi-dataset 2D loader (docs/todo.md 6f): scenes stored as per-frame INSTANCE-ID MAPS.

`data/scannet_overfit.py` reads ScanNet's one-folder-per-instance mask tree, which encodes the
class in the folder name and therefore only exists for a 19-class taxonomy. Everything built by
`slurm/build_insscene2d.py` — ScanNet++ and Infinigen, out of the InsScene-15K mirror — instead
stores one uint16 id map per frame:

    <scene>/color/<stem>.jpg        518x518 RGB
    <scene>/instance/<stem>.png     uint16, 0 = background, ids CONSISTENT ACROSS THE SCENE
    <scene>/manifest.json           frames, id table, provenance

This module reads that layout and returns **the same sample dict** as
`ScanNetSingleSceneDataset`, so the trainer, the feature cache, `build_frame_targets` and both
evaluators consume the two interchangeably. `build_scene_dataset` dispatches per directory, which
is what makes a mixed ScanNet + ScanNet++ + Infinigen + RE10K scene list a single flat list of
paths — a source added to the build needs no change here.

**Every instance is class 1 here.** These datasets' taxonomies are not ScanNet's, so a scene from
them is only meaningful under `--class_agnostic` (docs/todo.md 6e), where the head has one class
and `build_frame_targets` collapses every label onto it. Training class-aware on such a scene
would supervise every object as ScanNet class 1; the trainer refuses that combination rather than
silently doing it.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

AGNOSTIC_CLASS = 1          # the single class index these datasets can honestly claim


def is_instance_map_scene(scene_dir) -> bool:
    """True when `scene_dir` is one of `slurm/build_insscene2d.py`'s scenes."""
    path = Path(scene_dir)
    return (path / "instance").is_dir() and (path / "color").is_dir()


class InstanceMapSceneDataset(Dataset):
    """
    One scene stored as per-frame instance-id maps. Yields exactly one sample, like its ScanNet
    sibling, so `frame_sampling='random'` is what produces a different bundle per draw.

    Args:
        scene_dir: directory holding color/, instance/ and manifest.json
        num_frames: frames per bundle
        img_size: side of the square the images and maps are resized to
        frame_sampling: 'even' (deterministic, spans the scene) or 'random'
        **_: `instance_level` and friends are accepted and ignored — an id map has no
            per-class variant, so the distinction does not exist here.
    """

    def __init__(self, scene_dir: str, num_frames: int = 8, img_size: int = 518,
                 frame_sampling: str = "even", **_):
        super().__init__()
        self.scene_dir = Path(scene_dir)
        self.num_frames = num_frames
        self.img_size = img_size
        if frame_sampling not in ("even", "random"):
            raise ValueError(f"frame_sampling must be 'even' or 'random', got {frame_sampling!r}")
        self.frame_sampling = frame_sampling

        self.color_dir = self.scene_dir / "color"
        self.instance_dir = self.scene_dir / "instance"
        if not self.color_dir.is_dir() or not self.instance_dir.is_dir():
            raise ValueError(f"not an instance-map scene: {self.scene_dir}")

        self.frames = sorted(p.stem for p in self.color_dir.iterdir()
                             if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
        self.frames = [s for s in self.frames if (self.instance_dir / f"{s}.png").exists()]
        if not self.frames:
            raise ValueError(f"no frame has both a colour image and an id map: {self.scene_dir}")

        manifest_path = self.scene_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}

    def __len__(self) -> int:
        return 1

    def _sample_frames(self) -> List[str]:
        k = min(self.num_frames, len(self.frames))
        if self.frame_sampling == "even":
            idx = np.unique(np.linspace(0, len(self.frames) - 1, k).round().astype(int)).tolist()
        else:
            idx = sorted(random.sample(range(len(self.frames)), k))
        return [self.frames[i] for i in idx]

    def _load_frame(self, stem: str) -> Tuple[torch.Tensor, np.ndarray]:
        image = Image.open(self.color_dir / f"{stem}.jpg").convert("RGB") \
            if (self.color_dir / f"{stem}.jpg").exists() \
            else Image.open(self.color_dir / f"{stem}.png").convert("RGB")
        if image.size != (self.img_size, self.img_size):
            image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        tensor = torch.from_numpy(
            np.array(image, dtype=np.float32) / 255.0).permute(2, 0, 1)

        ids = Image.open(self.instance_dir / f"{stem}.png")
        if ids.size != (self.img_size, self.img_size):
            ids = ids.resize((self.img_size, self.img_size), Image.NEAREST)
        return tensor, np.array(ids, dtype=np.int32)

    @staticmethod
    def _centroid(mask: np.ndarray) -> Tuple[float, float]:
        if not mask.any():
            return (0.5, 0.5)
        rows, cols = np.nonzero(mask)
        h, w = mask.shape
        return (float(cols.mean() / max(w - 1, 1)), float(rows.mean() / max(h - 1, 1)))

    def __getitem__(self, idx) -> Dict:
        stems = self._sample_frames()
        images, id_maps = [], []
        for stem in stems:
            image, ids = self._load_frame(stem)
            images.append(image)
            id_maps.append(ids)

        # Renumber the ids PRESENT in this bundle onto 1..G, keeping one id per object across
        # every frame it appears in — the cross-view identity the bundle GT is re-linked by.
        present = sorted({int(v) for ids in id_maps for v in np.unique(ids) if v > 0})
        remap = {src: new for new, src in enumerate(present, start=1)}

        masks = np.zeros((len(stems), self.img_size, self.img_size), dtype=np.int32)
        best = {new: (-1, -1, (0.5, 0.5)) for new in remap.values()}   # area, frame, centroid
        for frame_idx, ids in enumerate(id_maps):
            for src, new in remap.items():
                pixels = ids == src
                area = int(pixels.sum())
                if area == 0:
                    continue
                masks[frame_idx][pixels] = new
                if area > best[new][0]:
                    best[new] = (area, frame_idx, self._centroid(pixels))

        kept = [new for new in sorted(best) if best[new][0] > 0]
        classes = torch.full((len(kept),), AGNOSTIC_CLASS, dtype=torch.long)
        coordinates = torch.tensor([best[n][2] for n in kept], dtype=torch.float32) \
            if kept else torch.zeros((0, 2), dtype=torch.float32)
        frame_ids = torch.tensor([best[n][1] for n in kept], dtype=torch.long) \
            if kept else torch.zeros(0, dtype=torch.long)
        instance_ids = torch.tensor(kept, dtype=torch.long) if kept \
            else torch.zeros(0, dtype=torch.long)

        return {
            "images": torch.stack(images, dim=0),
            "masks": torch.from_numpy(masks),
            "classes": classes,
            "coordinates": coordinates,
            "frame_ids": frame_ids,
            "instance_ids": instance_ids,
            "frame_names": stems,
            "num_instances": len(kept),
        }


class MixedMultiSceneDataset(Dataset):
    """
    One item per scene, each read by whichever loader its directory layout calls for.

    This is the only place that knows a scene list may span datasets; everything downstream sees
    one flat sequence of samples with identical keys.
    """

    def __init__(self, scene_dirs: List[str], **kwargs):
        super().__init__()
        if not scene_dirs:
            raise ValueError("scene_dirs must contain at least one scene directory")
        from data.scannet_overfit import ScanNetSingleSceneDataset   # local: avoids a cycle

        self.scenes, self.scene_names, self.sources = [], [], []
        for scene_dir in scene_dirs:
            path = Path(scene_dir)
            if is_instance_map_scene(path):
                self.scenes.append(InstanceMapSceneDataset(str(path), **kwargs))
                self.sources.append(path.parent.name)      # scannetpp | infinigen
            else:
                self.scenes.append(ScanNetSingleSceneDataset(str(path), **kwargs))
                self.sources.append("scannet")
            self.scene_names.append(path.parent.name if path.name == "raw_data" else path.name)

    def __len__(self) -> int:
        return len(self.scenes)

    def __getitem__(self, idx) -> Dict:
        sample = self.scenes[idx][0]
        sample["scene_name"] = self.scene_names[idx]
        sample["scene_idx"] = idx
        sample["source"] = self.sources[idx]
        return sample

    def counts_by_source(self) -> Dict[str, int]:
        out: Dict[str, int] = {}
        for source in self.sources:
            out[source] = out.get(source, 0) + 1
        return out


def build_scene_dataset(scene_dirs: List[str], **kwargs) -> Dataset:
    """
    The dispatcher `train/maskdino_data.py::prepare_scenes` calls.

    Pure ScanNet lists keep using `ScanNetMultiSceneDataset`, so no existing run changes shape;
    anything else goes through the mixed wrapper.
    """
    from data.scannet_overfit import ScanNetMultiSceneDataset

    if any(is_instance_map_scene(d) for d in scene_dirs):
        return MixedMultiSceneDataset(scene_dirs, **kwargs)
    return ScanNetMultiSceneDataset(scene_dirs, **kwargs)

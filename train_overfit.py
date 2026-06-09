#!/usr/bin/env python3
"""
Phase 6: Minimal Overfit Training Loop for D4RT Instance Segmentation

This script combines all components (VGGT backbone, QueryGenerator, InstanceDecoder, Loss)
into an end-to-end training pipeline. It validates that gradients flow correctly from
SAM3 pseudo-labels through the entire system.

Usage:
    python train_overfit.py --num_epochs 500 --scene_dir /path/to/scannet/scene
"""

import argparse
import random
import sys
from pathlib import Path
from typing import Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from vggt.models.vggt import VGGT
from models.d4rt_decoder import D4RTInstanceSegmentationHead
from train.loss import D4RTLoss
from train.eval_metrics import compute_instance_segmentation_metrics
from data.scannet_overfit import ScanNetSingleSceneDataset


class D4RTModel(nn.Module):
    """
    Complete D4RT instance segmentation model combining VGGT backbone and decoder.

    Args:
        freeze_backbone (bool): If True, freeze VGGT backbone weights
        num_views (int): Maximum number of views
        decoder_hidden_dim (int): Decoder hidden dimension
        mask_embed_dim (int): Mask embedding dimension
    """

    def __init__(
        self,
        freeze_backbone: bool = True,
        num_views: int = 10,
        decoder_hidden_dim: int = 256,
        mask_embed_dim: int = 256,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.freeze_backbone = freeze_backbone

        # Load VGGT backbone
        print("Loading VGGT backbone...")
        try:
            self.backbone = VGGT.from_pretrained("facebook/VGGT-1B")
            print("✓ Loaded pretrained VGGT-1B")
        except Exception as e:
            print(f"⚠ Could not load pretrained VGGT: {e}")
            print("  Creating VGGT with random initialization...")
            self.backbone = VGGT()

        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("✓ VGGT backbone frozen")
        else:
            print("✓ VGGT backbone trainable")

        # D4RT decoder head
        self.decoder_head = D4RTInstanceSegmentationHead(
            num_views=num_views,
            hidden_dim=decoder_hidden_dim,
            num_classes=20,
            num_decoder_layers=4,
            patch_size=9,
            mask_embed_dim=mask_embed_dim,
            memory_dim=2048,  # 2 * embed_dim from VGGT
            dropout=dropout,
        )

    def forward(
        self,
        images: torch.Tensor,
        coordinates: torch.Tensor,
        view_ids: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through VGGT backbone and D4RT decoder.

        Args:
            images (torch.Tensor): [B, S, 3, H, W] input images
            coordinates (torch.Tensor): [B, N, 2] query coordinates
            view_ids (torch.Tensor): [B, N] view indices

        Returns:
            Dict with class_logits and mask_embeddings
        """
        # VGGT forward pass
        with torch.no_grad() if self.freeze_backbone else torch.enable_grad():
            aggregated_tokens_list, patch_start_idx = self.backbone.aggregator(images)

        # Extract global features from final cached layer
        global_features = aggregated_tokens_list[-1]  # [B, S, P, 2048]

        # D4RT decoder forward pass
        class_logits, mask_embeddings, pred_masks = self.decoder_head(
            coordinates, view_ids, images, global_features, patch_start_idx
        )

        return {
            "class_logits": class_logits,
            "mask_embeddings": mask_embeddings,
            "pred_masks": pred_masks,
        }


def create_dataloader(
    scene_dir: Optional[str] = None,
    num_frames: int = 8,
    batch_size: int = 1,
    num_workers: int = 0,
) -> DataLoader:
    """Create a dataloader for the scene."""
    if scene_dir is None:
        scene_dir = "/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data"

    scene_dir = Path(scene_dir)
    if not scene_dir.exists():
        raise ValueError(f"Scene directory not found: {scene_dir}")

    print(f"Loading dataset from: {scene_dir}")
    dataset = ScanNetSingleSceneDataset(
        scene_dir=str(scene_dir),
        num_frames=num_frames,
        img_size=518,  # Must be divisible by VGGT patch_size (14)
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    return dataloader


def create_synthetic_scene(scene_dir: str, num_frames: int = 4, img_size: int = 256):
    """Create a minimal synthetic ScanNet-like scene for testing."""
    import numpy as np
    from PIL import Image

    scene_path = Path(scene_dir)
    images_dir = scene_path / "images"
    masks_dir = scene_path / "masks"

    images_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    # Create sample images
    print(f"Creating {num_frames} synthetic images...")
    for i in range(num_frames):
        # Random RGB image
        img_array = np.random.randint(0, 256, (img_size, img_size, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img.save(images_dir / f"frame_{i:05d}.jpg")

    # Create masks for 3 classes
    test_classes = ["wall", "floor", "chair"]
    print(f"Creating masks for classes: {test_classes}")

    for class_name in test_classes:
        class_mask_dir = masks_dir / class_name
        class_mask_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_frames):
            # Create a simple binary mask (random rectangular region)
            mask_array = np.zeros((img_size, img_size), dtype=np.uint8)

            # Add a random rectangular foreground region
            y_start = np.random.randint(0, img_size // 2)
            y_end = np.random.randint(img_size // 2, img_size)
            x_start = np.random.randint(0, img_size // 2)
            x_end = np.random.randint(img_size // 2, img_size)
            mask_array[y_start:y_end, x_start:x_end] = 255

            mask = Image.fromarray(mask_array)
            mask.save(class_mask_dir / f"frame_{i:05d}.png")

    print(f"Synthetic scene created at {scene_path}")
    return scene_path


def generate_query_points(
    batch: Dict[str, torch.Tensor],
    num_queries: int = 16,
    device: str = "cpu",
) -> tuple:
    """
    Generate query points from instance mask centroid and background points.

    Args:
        batch: Batch from dataloader containing 'coordinates' and 'num_instances'
        num_queries: Total number of queries per batch
        device: Device to create tensors on

    Returns:
        Tuple of (coordinates, view_ids)
    """
    B = batch["images"].shape[0]
    S = batch["images"].shape[1]

    coordinates_list = []
    view_ids_list = []

    for b in range(B):
        # num_instances may be an int or a (collated) 1-element tensor; coerce to int
        num_instances = int(batch["num_instances"]) if not isinstance(batch["num_instances"], int) else batch["num_instances"]
        num_bg_points = max(1, num_queries - num_instances)

        # Instance points from dataset
        # batch["coordinates"] might be [num_instances, 2], move to device
        instance_coords = batch["coordinates"].squeeze(0) if batch["coordinates"].dim() > 2 else batch["coordinates"]
        instance_coords = instance_coords.to(device)  # [num_instances, 2]

        # Each instance query takes the view of the frame it was labeled in, so its view
        # embedding AND its local RGB patch are sampled from the correct frame (instances
        # from different frames may share near-identical centroids).
        if "frame_ids" in batch:
            instance_view_ids = batch["frame_ids"].squeeze(0) if batch["frame_ids"].dim() > 1 else batch["frame_ids"]
            instance_view_ids = instance_view_ids.to(device).long()
        else:
            instance_view_ids = torch.zeros(len(instance_coords), dtype=torch.long, device=device)

        # Generate random background points
        bg_coords = torch.rand(num_bg_points, 2, device=device)
        bg_view_ids = torch.randint(0, S, (num_bg_points,), device=device)

        # Combine
        all_coords = torch.cat([instance_coords, bg_coords], dim=0)  # [num_queries, 2]
        all_view_ids = torch.cat([instance_view_ids, bg_view_ids], dim=0)  # [num_queries]

        coordinates_list.append(all_coords)
        view_ids_list.append(all_view_ids)

    # Stack into batch
    coordinates = torch.stack(coordinates_list, dim=0)  # [B, num_queries, 2]
    view_ids = torch.stack(view_ids_list, dim=0)  # [B, num_queries]

    return coordinates, view_ids


def _squeeze_batch_dim(t: torch.Tensor) -> torch.Tensor:
    """Drop the leading (size-1) batch dim added by the default DataLoader collate."""
    return t.squeeze(0) if t.dim() > 0 and t.shape[0] == 1 and t.dim() > 1 else t


@torch.no_grad()
def build_gt_targets(
    batch: Dict[str, torch.Tensor],
    patch_start_idx: int,
    num_patch_tokens: int,
    device: str,
) -> Dict[str, torch.Tensor]:
    """
    Build dense ground-truth targets for multi-view instance segmentation (item 8.4).

    Every target is derived directly from the SAM3 pseudo-labels:

      - classes      : the real ScanNet class label of each cross-view instance.
      - coordinates  : the real (u, v) centroid of each instance (representative frame).
      - masks        : per-instance DENSE binary masks at the VGGT patch-grid resolution,
                       shape [Ng, S, h, w]. The instance's binary mask in each frame is
                       downsampled (area interpolation) to the patch grid; the peak patch is
                       always kept so visible-but-small instances are not erased. These are the
                       targets for the Dice + BCE mask loss and for the mask-aware matcher.

    Because the dataset assigns a single global ID per instance across frames (item 8.3), an
    instance's mask spans every frame it is visible in.

    Args:
        batch: One (collated, batch-size-1) sample from ScanNetSingleSceneDataset.
        patch_start_idx: index where patch tokens begin (special tokens precede them).
        num_patch_tokens: number of patch tokens P - patch_start_idx (=> h = w = sqrt of it).
        device: target device.

    Returns:
        Dict with classes [Ng], coordinates [Ng, 2], masks [Ng, S, h, w].
    """
    classes = _squeeze_batch_dim(batch["classes"]).to(device)            # [Ng]
    coordinates = _squeeze_batch_dim(batch["coordinates"]).to(device)    # [Ng, 2]
    masks = _squeeze_batch_dim(batch["masks"]).to(device)                # [S, H, W] global-id map

    S = masks.shape[0]
    h = w = int(round(num_patch_tokens ** 0.5))
    assert h * w == num_patch_tokens, "patch tokens do not form a square grid"

    num_instances = int(classes.shape[0])
    if num_instances == 0:
        # Degenerate scene: a single empty dummy target so the loss/metrics stay defined.
        return {
            "classes": torch.zeros(1, dtype=torch.long, device=device),
            "coordinates": torch.full((1, 2), 0.5, device=device),
            "masks": torch.zeros(1, S, h, w, device=device),
        }

    gt_masks = torch.zeros(num_instances, S, h, w, device=device)
    for i in range(num_instances):
        inst_id = i + 1  # i-th instance has global id (i+1) in the mask map (all frames)
        for f in range(S):
            bin_mask = (masks[f] == inst_id).float()  # [H, W]
            if bin_mask.sum() == 0:
                continue
            occ = F.interpolate(bin_mask[None, None], size=(h, w), mode="area")[0, 0]  # [h, w]
            # Keep the peak patch even for small instances; exclude non-overlapping patches.
            thr = min(0.5, float(occ.max().item()))
            gt_masks[i, f] = ((occ >= thr) & (occ > 0)).float()

    return {
        "classes": classes,
        "coordinates": coordinates,
        "masks": gt_masks,
    }


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    images: torch.Tensor,
    coordinates: torch.Tensor,
    view_ids: torch.Tensor,
    gt: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    """
    Run one forward pass in eval mode and compute instance-segmentation metrics (item 8.4).

    Assumes batch size 1 (single overfit scene). Returns the metric dict from
    `compute_instance_segmentation_metrics` (mIoU / AP50 / AP75 / mAP / class_acc).
    """
    was_training = model.training
    model.eval()
    outputs = model(images, coordinates, view_ids)
    metrics = compute_instance_segmentation_metrics(
        pred_masks=outputs["pred_masks"][0],     # [N, S, h, w]
        class_logits=outputs["class_logits"][0],  # [N, C]
        gt_masks=gt["masks"],                      # [Ng, S, h, w]
        gt_classes=gt["classes"],                  # [Ng]
        background_class=0,
    )
    if was_training:
        model.train()
    return metrics


def _format_metrics(m: Dict[str, float]) -> str:
    return (
        f"mIoU={m['mIoU']:.3f}  AP50={m['AP50']:.3f}  AP75={m['AP75']:.3f}  "
        f"mAP={m['mAP']:.3f}  class_acc={m['class_acc']:.3f}  "
        f"(pred={m['num_pred']}, gt={m['num_gt']})"
    )


def main():
    parser = argparse.ArgumentParser(description="D4RT Overfit Training Loop")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--num_frames", type=int, default=4, help="Number of frames for overfitting")
    parser.add_argument("--num_queries", type=int, default=16, help="Number of query points per image")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument(
        "--scene_dir",
        type=str,
        default="/cluster/work/igp_psr/niacobone/distillation/dataset/scannet/scans/scene0000_00/raw_data",
        help="Path to ScanNet scene directory",
    )
    parser.add_argument("--freeze_backbone", action="store_true", default=True, help="Freeze VGGT backbone")
    parser.add_argument("--unfreeze_backbone", action="store_true", help="Unfreeze VGGT backbone")
    parser.add_argument("--dropout", type=float, default=0.0,
                        help="Decoder dropout (0 for a clean overfit; >0 for regularized training)")
    parser.add_argument("--log_interval", type=int, default=10, help="Log every N iterations")
    parser.add_argument("--save_checkpoint", type=str, default=None, help="Path to save checkpoint")

    args = parser.parse_args()

    # Seed everything for reproducibility of the overfit test
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)

    # Override freeze_backbone if unfreeze_backbone is set
    if args.unfreeze_backbone:
        args.freeze_backbone = False

    # Setup device
    if args.device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        device = "cpu"
    else:
        device = args.device

    print(f"Using device: {device}")
    print("=" * 70)

    # Create model
    print("\n=== Initializing Model ===")
    model = D4RTModel(
        freeze_backbone=args.freeze_backbone,
        num_views=10,
        decoder_hidden_dim=256,
        mask_embed_dim=256,
        dropout=args.dropout,
    )
    model = model.to(device)

    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Create dataloader
    print("\n=== Loading Dataset ===")
    try:
        dataloader = create_dataloader(
            scene_dir=args.scene_dir,
            num_frames=args.num_frames,
            batch_size=args.batch_size,
        )
        print(f"✓ Loaded dataset with {len(dataloader)} batches")
    except Exception as e:
        print(f"✗ Failed to load dataset: {e}")
        print("  Creating synthetic dataset for testing...")
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            scene_dir = create_synthetic_scene(tmpdir, num_frames=args.num_frames, img_size=280)
            dataloader = create_dataloader(
                scene_dir=str(scene_dir),
                num_frames=args.num_frames,
                batch_size=args.batch_size,
            )

    # Create loss function
    print("\n=== Initializing Loss Function ===")
    # NOTE: the model now predicts DENSE masks (item 8.4), so the mask objective is the real
    # Dice + BCE loss (`mask_loss_weight=1.0`) and the matcher is mask-aware. The old
    # mask-embedding regression proxy is disabled (`mask_embed_loss_weight=0.0`); mask
    # embeddings are now trained purely as the per-query mask kernels via the dense mask loss.
    # coord_loss_weight is 0 because there is no coordinate-regression head — the "predicted"
    # coordinates are the fixed input query coordinates, so a coord loss has no gradient path;
    # coordinates are still used inside the matcher.
    loss_fn = D4RTLoss(
        num_classes=20,
        focal_alpha=0.25,
        focal_gamma=2.0,
        class_loss_weight=1.0,
        mask_embed_loss_weight=0.0,
        coord_loss_weight=0.0,
        mask_loss_weight=1.0,
    )
    loss_fn = loss_fn.to(device)
    print("✓ Loss function initialized")

    # Create optimizer
    print("\n=== Initializing Optimizer ===")
    optimizer = AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=1e-4,
    )
    print(f"✓ AdamW optimizer with lr={args.learning_rate}")

    # Prepare a SINGLE fixed batch + fixed targets for a genuine overfit test.
    # A real overfit requires constant inputs and constant targets so the model has
    # something stable to memorize. (Regenerating random frames/queries/targets every
    # iteration makes the loss a moving target that can never decrease, regardless of
    # whether gradients flow.)
    print("\n=== Preparing Fixed Overfit Batch ===")
    fixed_batch = next(iter(dataloader))
    images = fixed_batch["images"].to(device)  # [B, S, 3, H, W]
    coordinates, view_ids = generate_query_points(fixed_batch, args.num_queries, device)

    # Run the frozen backbone ONCE to obtain the patch layout, then build dense GT targets
    # (classes / centroids / per-instance binary masks) from the SAM3 masks (item 8.4). The GT
    # masks live at the VGGT patch-grid resolution so they line up with the predicted masks.
    with torch.no_grad():
        agg_list, patch_start_idx = model.backbone.aggregator(images)
    global_features = agg_list[-1]  # [B, S, P, C_feat]
    num_patch_tokens = global_features.shape[2] - patch_start_idx

    gt = build_gt_targets(fixed_batch, patch_start_idx, num_patch_tokens, device)
    print(
        f"✓ Fixed batch: images={tuple(images.shape)}, "
        f"queries={coordinates.shape[1]}, gt_instances={gt['classes'].shape[0]}, "
        f"gt_masks={tuple(gt['masks'].shape)} (real classes/centroids/dense masks)"
    )

    # Baseline metrics before any training (sanity reference for the final numbers).
    init_metrics = evaluate_model(model, images, coordinates, view_ids, gt)
    print(f"  Initial metrics: {_format_metrics(init_metrics)}")

    # Training loop (overfit a single fixed batch)
    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)

    losses_history = []

    for epoch in range(args.num_epochs):
        model.train()
        optimizer.zero_grad()

        outputs = model(images, coordinates, view_ids)
        total_loss_val, loss_components = loss_fn(
            outputs["class_logits"],
            outputs["mask_embeddings"],
            coordinates,
            gt["classes"],
            gt_mask_embeddings=None,           # descriptor proxy disabled (item 8.4)
            gt_coordinates=gt["coordinates"],
            gt_masks=gt["masks"],
            pred_masks=outputs["pred_masks"],
        )

        total_loss_val.backward()
        # Clip gradients: the dense mask dot-product can produce large, oscillating gradients.
        torch.nn.utils.clip_grad_norm_(
            filter(lambda p: p.requires_grad, model.parameters()), max_norm=1.0
        )
        optimizer.step()

        epoch_losses = {
            "total_loss": total_loss_val.item(),
            "class_loss": loss_components["class_loss"].item(),
            "mask_embed_loss": loss_components["mask_embed_loss"].item(),
            "coord_loss": loss_components["coord_loss"].item(),
            "mask_loss": loss_components["mask_loss"].item()
            if isinstance(loss_components["mask_loss"], torch.Tensor)
            else float(loss_components["mask_loss"]),
        }
        losses_history.append(epoch_losses)

        # Log every log_interval epochs (and the first/last)
        if epoch == 0 or (epoch + 1) % args.log_interval == 0 or epoch == args.num_epochs - 1:
            print(
                f"[Epoch {epoch + 1:4d}/{args.num_epochs}] "
                f"Total: {epoch_losses['total_loss']:8.4f} "
                f"(Class: {epoch_losses['class_loss']:6.4f}, "
                f"Mask(Dice+BCE): {epoch_losses['mask_loss']:7.4f}) "
                f"Matches: {loss_components['num_matches']}"
            )

        # Check for divergence
        if epoch_losses["total_loss"] > 1e6 or np.isnan(epoch_losses["total_loss"]):
            print("⚠ Warning: Loss is diverging or NaN!")
            break

    # Summary
    print("\n" + "=" * 70)
    print("TRAINING COMPLETE")
    print("=" * 70)

    initial_loss = losses_history[0]["total_loss"]
    final_loss = losses_history[-1]["total_loss"]
    loss_reduction = ((initial_loss - final_loss) / initial_loss * 100) if initial_loss > 0 else 0

    print(f"\nInitial loss: {initial_loss:.4f}")
    print(f"Final loss:   {final_loss:.4f}")
    print(f"Reduction:    {loss_reduction:.2f}%")

    # Final instance-segmentation metrics (item 8.4 eval metric).
    final_metrics = evaluate_model(model, images, coordinates, view_ids, gt)
    print("\n--- Instance Segmentation Metrics ---")
    print(f"  Before training: {_format_metrics(init_metrics)}")
    print(f"  After training:  {_format_metrics(final_metrics)}")

    if loss_reduction > 50:
        print("\n✅ SUCCESS: Loss decreased significantly!")
        print("   Gradients flow correctly through entire pipeline.")
    elif loss_reduction > 0:
        print("\n⚠ PARTIAL SUCCESS: Loss decreased but not dramatically.")
        print("   Check if overfitting parameters are appropriate.")
    else:
        print("\n❌ FAILURE: Loss did not decrease.")
        print("   There may be issues with gradient flow.")

    # Save checkpoint
    if args.save_checkpoint:
        ckpt_path = Path(args.save_checkpoint)
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving checkpoint to {ckpt_path}")
        # Self-contained checkpoint: only the trainable decoder head (the ~1.26B frozen VGGT
        # backbone is reloaded from `facebook/VGGT-1B`), plus the exact fixed overfit batch
        # (images, queries, dense GT, patch layout) so visualization/eval reproduces precisely
        # what was trained — no dependence on re-deriving RNG state.
        torch.save(
            {
                "decoder_head_state_dict": model.decoder_head.state_dict(),
                "images": images.cpu(),
                "coordinates": coordinates.cpu(),
                "view_ids": view_ids.cpu(),
                "gt": {k: v.cpu() for k, v in gt.items()},
                "patch_start_idx": int(patch_start_idx),
                "num_patch_tokens": int(num_patch_tokens),
                "frame_names": fixed_batch.get("frame_names", None),
                "args": vars(args),
                "init_metrics": init_metrics,
                "final_metrics": final_metrics,
            },
            ckpt_path,
        )
        print(f"✓ Saved decoder head + fixed batch ({ckpt_path.stat().st_size / 1e6:.1f} MB)")

    return 0 if loss_reduction > 50 else 1


if __name__ == "__main__":
    sys.exit(main())

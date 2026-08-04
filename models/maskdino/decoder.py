"""
MaskDINO transformer decoder, ported to run on VGGT features (docs/MASKDINO.md).

Faithful port of `maskdino/modeling/transformer_decoder/maskdino_decoder.py` with the
detectron2 machinery removed (see §2 of the trial doc). The generic DAB/DINO decoder stack it
drives lives in decoder_layers.py. Everything that makes MaskDINO a DINO-family detector is
kept:

  - DAB-style anchor boxes as query positional embeddings, **refined at every layer**;
  - two-stage query selection from the encoder's own class/box predictions;
  - mask-enhanced anchor initialisation (`initialize_box_type`);
  - denoising (DN) queries built from noised GT labels+boxes, isolated by an attention mask;
  - deep supervision (initial prediction + every layer + the encoder's interm output).
"""

from typing import List, Optional

import torch
from torch import Tensor, nn

from . import box_ops
from .anchor3d import ANCHOR_LOG_R0, pyramid_token_xyz
from .decoder_layers import DeformableTransformerDecoderLayer, TransformerDecoder
from .utils import MLP, gen_encoder_output_proposals, inverse_sigmoid


class MaskDINODecoder(nn.Module):
    """
    Args mirror `cfg.MODEL.MaskDINO.*` of the COCO instance-segmentation config
    (`maskdino_R50_bs16_50ep_3s.yaml`): 300 queries, 9 layers, two-stage, DN "seg",
    `initialize_box_type='bitmask'`.

    Args:
        in_channels: channels of the multi-scale memory maps (the pixel decoder's conv_dim).
        num_classes: number of foreground classes; there is NO background column — "no object"
            is represented by all sigmoid logits being low (DINO convention).
        mask_dim: channels of `mask_features`.
        dn: "no" | "seg" — "seg" adds mask losses on the denoising queries too.
        initialize_box_type: "no" | "bitmask" — seed the decoder's anchor boxes from the
            initial predicted masks instead of learned/encoder boxes. Upstream also offers
            "mask2box"; this port implements only the tight-bounding-box ("bitmask") variant,
            so that name is not accepted rather than silently aliased.
    """

    def __init__(
        self,
        in_channels: int = 256,
        num_classes: int = 19,
        hidden_dim: int = 256,
        num_queries: int = 300,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        dec_layers: int = 9,
        mask_dim: int = 256,
        enforce_input_project: bool = False,
        two_stage: bool = True,
        dn: str = "seg",
        noise_scale: float = 0.4,
        dn_num: int = 100,
        initialize_box_type: str = "bitmask",
        initial_pred: bool = True,
        learn_tgt: bool = False,
        total_num_feature_levels: int = 3,
        dropout: float = 0.0,
        activation: str = "relu",
        dec_n_points: int = 4,
        query_dim: int = 4,
        cross_frame_attn: bool = False,
        anchor_3d: bool = False,
    ):
        super().__init__()
        self.num_feature_levels = total_num_feature_levels
        self.total_num_feature_levels = total_num_feature_levels
        self.initial_pred = initial_pred
        self.dn = dn
        self.learn_tgt = learn_tgt
        self.noise_scale = noise_scale
        self.dn_num = dn_num
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.two_stage = two_stage
        if initialize_box_type not in ("no", "bitmask"):
            raise ValueError(
                f"initialize_box_type={initialize_box_type!r} is not implemented by this port. "
                "Upstream MaskDINO also offers 'mask2box'; only 'no' and 'bitmask' exist here.")
        self.initialize_box_type = initialize_box_type
        self.num_queries = num_queries
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.anchor_3d = anchor_3d

        if (not two_stage) or self.learn_tgt:
            self.query_feat = nn.Embedding(num_queries, hidden_dim)
        if (not two_stage) and initialize_box_type == "no":
            self.query_embed = nn.Embedding(num_queries, 4)
        if two_stage:
            self.enc_output = nn.Linear(hidden_dim, hidden_dim)
            self.enc_output_norm = nn.LayerNorm(hidden_dim)

        self.input_proj = nn.ModuleList()
        for _ in range(self.num_feature_levels):
            if in_channels != hidden_dim or enforce_input_project:
                conv = nn.Conv2d(in_channels, hidden_dim, kernel_size=1)
                nn.init.xavier_uniform_(conv.weight, gain=1)
                nn.init.constant_(conv.bias, 0)
                self.input_proj.append(conv)
            else:
                self.input_proj.append(nn.Sequential())

        self.class_embed = nn.Linear(hidden_dim, num_classes)
        self.label_enc = nn.Embedding(num_classes, hidden_dim)
        self.mask_embed = MLP(hidden_dim, hidden_dim, mask_dim, 3)

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        decoder_layer = DeformableTransformerDecoderLayer(
            hidden_dim, dim_feedforward, dropout, activation,
            self.num_feature_levels, nheads, dec_n_points)
        self.decoder = TransformerDecoder(
            decoder_layer, self.num_layers, self.decoder_norm, d_model=hidden_dim,
            query_dim=query_dim, num_feature_levels=self.num_feature_levels,
            nheads=nheads, dropout=dropout, cross_frame_attn=cross_frame_attn,
        )

        self._bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
        nn.init.constant_(self._bbox_embed.layers[-1].weight.data, 0)
        nn.init.constant_(self._bbox_embed.layers[-1].bias.data, 0)
        # one shared box head applied at every layer (MaskDINO shares, DINO can un-share)
        self.bbox_embed = nn.ModuleList([self._bbox_embed for _ in range(self.num_layers)])
        self.decoder.bbox_embed = self.bbox_embed

        if anchor_3d:
            # The 3D twin of `_bbox_embed`: Delta(x, y, z, log r), shared across layers and
            # zero-initialised the same way, so layer 0 starts exactly at the gathered anchor.
            self._anchor_embed = MLP(hidden_dim, hidden_dim, 4, 3)
            nn.init.constant_(self._anchor_embed.layers[-1].weight.data, 0)
            nn.init.constant_(self._anchor_embed.layers[-1].bias.data, 0)
            self.anchor_embed = nn.ModuleList(
                [self._anchor_embed for _ in range(self.num_layers)])
            self.decoder.anchor_embed = self.anchor_embed
            if not two_stage:
                # No encoder proposals to gather an anchor from — learn one per query instead.
                self.anchor_query_embed = nn.Embedding(num_queries, 4)
                nn.init.normal_(self.anchor_query_embed.weight, std=0.5)
                nn.init.constant_(self.anchor_query_embed.weight.data[:, 3], ANCHOR_LOG_R0)

    # ---- denoising ---------------------------------------------------------------------

    def prepare_for_dn(self, targets, tgt, refpoint_emb, batch_size, device):
        """
        Build the denoising query group from noised GT labels+boxes (DN-DETR / DINO).

        Returns (input_query_label, input_query_bbox, attn_mask, mask_dict); all None outside
        training or when there is no GT to noise.
        """
        if not self.training:
            if refpoint_emb is not None:
                return tgt.repeat(batch_size, 1, 1), refpoint_emb.repeat(batch_size, 1, 1), None, None
            return None, None, None, None

        scalar, noise_scale = self.dn_num, self.noise_scale

        known = [torch.ones_like(t["labels"], device=device) for t in targets]
        know_idx = [torch.nonzero(t) for t in known]
        known_num = [int(k.sum()) for k in known]

        # fixed DN budget: `dn_num` queries total → `scalar` noised copies of the GT set
        scalar = scalar // int(max(known_num)) if max(known_num) > 0 else 0
        if scalar == 0:
            return None, None, None, None

        unmask_bbox = unmask_label = torch.cat(known)
        labels = torch.cat([t["labels"] for t in targets])
        boxes = torch.cat([t["boxes"] for t in targets])
        batch_idx = torch.cat([torch.full_like(t["labels"].long(), i) for i, t in enumerate(targets)])

        known_indice = torch.nonzero(unmask_label + unmask_bbox).view(-1)
        known_indice = known_indice.repeat(scalar, 1).view(-1)
        known_labels = labels.repeat(scalar, 1).view(-1)
        known_bid = batch_idx.repeat(scalar, 1).view(-1)
        known_bboxs = boxes.repeat(scalar, 1)
        known_labels_expaned = known_labels.clone()
        known_bbox_expand = known_bboxs.clone()

        if noise_scale > 0:
            # flip a fraction of the labels ...
            p = torch.rand_like(known_labels_expaned.float())
            chosen_indice = torch.nonzero(p < (noise_scale * 0.5)).view(-1)
            new_label = torch.randint_like(chosen_indice, 0, self.num_classes)
            known_labels_expaned.scatter_(0, chosen_indice, new_label)
            # ... and jitter the boxes (centre by w/2, size by w)
            diff = torch.zeros_like(known_bbox_expand)
            diff[:, :2] = known_bbox_expand[:, 2:] / 2
            diff[:, 2:] = known_bbox_expand[:, 2:]
            known_bbox_expand = known_bbox_expand + torch.mul(
                (torch.rand_like(known_bbox_expand) * 2 - 1.0), diff) * noise_scale
            known_bbox_expand = known_bbox_expand.clamp(min=0.0, max=1.0)

        input_label_embed = self.label_enc(known_labels_expaned.long())
        input_bbox_embed = inverse_sigmoid(known_bbox_expand)
        single_pad = int(max(known_num))
        pad_size = int(single_pad * scalar)

        padding_label = torch.zeros(pad_size, self.hidden_dim, device=device)
        padding_bbox = torch.zeros(pad_size, 4, device=device)
        if refpoint_emb is not None:
            input_query_label = torch.cat([padding_label, tgt], dim=0).repeat(batch_size, 1, 1)
            input_query_bbox = torch.cat([padding_bbox, refpoint_emb], dim=0).repeat(batch_size, 1, 1)
        else:
            input_query_label = padding_label.repeat(batch_size, 1, 1)
            input_query_bbox = padding_bbox.repeat(batch_size, 1, 1)

        map_known_indice = torch.tensor([], device=device)
        if len(known_num):
            map_known_indice = torch.cat([torch.arange(num) for num in known_num]).to(device)
            map_known_indice = torch.cat(
                [map_known_indice + single_pad * i for i in range(scalar)]).long()
        if len(known_bid):
            input_query_label[(known_bid.long(), map_known_indice)] = input_label_embed
            input_query_bbox[(known_bid.long(), map_known_indice)] = input_bbox_embed

        tgt_size = pad_size + self.num_queries
        attn_mask = torch.zeros(tgt_size, tgt_size, device=device, dtype=torch.bool)
        # matching queries cannot see the reconstruction group ...
        attn_mask[pad_size:, :pad_size] = True
        # ... and the DN groups cannot see each other.
        for i in range(scalar):
            if i == 0:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
            if i == scalar - 1:
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True
            else:
                attn_mask[single_pad * i:single_pad * (i + 1), single_pad * (i + 1):pad_size] = True
                attn_mask[single_pad * i:single_pad * (i + 1), :single_pad * i] = True

        mask_dict = {
            "known_indice": known_indice.long(),
            "batch_idx": batch_idx.long(),
            "map_known_indice": map_known_indice.long(),
            "known_lbs_bboxes": (known_labels, known_bboxs),
            "know_idx": know_idx,
            "pad_size": pad_size,
            "scalar": scalar,
        }
        return input_query_label, input_query_bbox, attn_mask, mask_dict

    def dn_post_process(self, outputs_class, outputs_coord, mask_dict, outputs_mask):
        """Split the DN part off the front of every prediction and stash it in `mask_dict`."""
        assert mask_dict["pad_size"] > 0
        pad = mask_dict["pad_size"]
        output_known_class = outputs_class[:, :, :pad, :]
        outputs_class = outputs_class[:, :, pad:, :]
        output_known_coord = outputs_coord[:, :, :pad, :]
        outputs_coord = outputs_coord[:, :, pad:, :]
        output_known_mask = None
        if outputs_mask is not None:
            output_known_mask = outputs_mask[:, :, :pad, :]
            outputs_mask = outputs_mask[:, :, pad:, :]
        out = {"pred_logits": output_known_class[-1], "pred_boxes": output_known_coord[-1],
               "pred_masks": output_known_mask[-1]}
        out["aux_outputs"] = self._set_aux_loss(output_known_class, output_known_mask,
                                                output_known_coord)
        mask_dict["output_known_lbs_bboxes"] = out
        return outputs_class, outputs_coord, outputs_mask

    # ---- prediction heads --------------------------------------------------------------

    def pred_box(self, reference, hs, ref0=None):
        outputs_coord_list = [] if ref0 is None else [ref0]
        for layer_ref_sig, layer_bbox_embed, layer_hs in zip(reference[:-1], self.bbox_embed, hs):
            layer_delta_unsig = layer_bbox_embed(layer_hs)
            outputs_coord_list.append((layer_delta_unsig + inverse_sigmoid(layer_ref_sig)).sigmoid())
        return torch.stack(outputs_coord_list)

    def forward_prediction_heads(self, output, mask_features, pred_mask=True):
        decoder_output = self.decoder_norm(output).transpose(0, 1)
        outputs_class = self.class_embed(decoder_output)
        outputs_mask = None
        if pred_mask:
            mask_embed = self.mask_embed(decoder_output)
            outputs_mask = torch.einsum("bqc,bchw->bqhw", mask_embed, mask_features)
        return outputs_class, outputs_mask

    @torch.jit.unused
    def _set_aux_loss(self, outputs_class, outputs_seg_masks, out_boxes=None):
        if out_boxes is None:
            return [{"pred_logits": a, "pred_masks": b}
                    for a, b in zip(outputs_class[:-1], outputs_seg_masks[:-1])]
        return [{"pred_logits": a, "pred_masks": b, "pred_boxes": c}
                for a, b, c in zip(outputs_class[:-1], outputs_seg_masks[:-1], out_boxes[:-1])]

    # ---- forward -------------------------------------------------------------------------

    def forward(self, x: List[Tensor], mask_features: Tensor, targets: Optional[List[dict]] = None,
                frames_per_sample: int = 1, token_xyz: Optional[Tensor] = None):
        """
        Args:
            x: multi-scale memory maps [B, C, H_l, W_l], HIGH→LOW resolution (as produced by
               `VGGTPixelDecoder`). Upstream MaskDINO reverses its own list internally; here the
               order is fixed by the pixel decoder and used as given.
            mask_features: [B, mask_dim, h, w] — the map every mask is decoded against.
            targets: per-sample dicts with "labels"/"boxes" (training only, for DN).
            frames_per_sample: S > 1 switches on the multi-frame mode (docs/MASKDINO.md §8):
               the batch is B bundles of S frames, contiguous in the batch dimension, and the
               query set is SHARED across the S frames of a bundle — selected once per bundle,
               broadcast to every frame, then refined per frame (each view keeps its own anchor
               box) and tied together by the decoder's cross-frame blocks.
            token_xyz: [B*S, h*w, 3] per-patch 3D positions from VGGT's frozen point head,
               normalised per bundle. Required by (and only used with) `anchor_3d`
               (docs/MASKDINO.md §8.3).
        Returns:
            (out, mask_dict) — `out` has pred_logits / pred_masks / pred_boxes (+ aux_outputs,
            interm_outputs); `mask_dict` carries the DN predictions for the criterion.
        """
        assert len(x) == self.num_feature_levels
        device = x[0].device
        bs = x[0].shape[0]
        s = int(frames_per_sample)
        assert s >= 1 and bs % s == 0, f"batch {bs} is not a multiple of frames_per_sample {s}"
        if self.anchor_3d and token_xyz is None:
            raise ValueError(
                "anchor_3d=True needs `token_xyz` (the cached per-patch 3D positions of VGGT's "
                "point head); see docs/MASKDINO.md §8.3 and train/maskdino_data.py.")
        anchor = None

        src_flatten, spatial_shapes = [], []
        for lvl, src in enumerate(x):
            spatial_shapes.append(src.shape[-2:])
            src_flatten.append(self.input_proj[lvl](src).flatten(2).transpose(1, 2))
        src_flatten = torch.cat(src_flatten, 1)                       # [bs, sum(hw), c]
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=device)
        level_start_index = torch.cat(
            (spatial_shapes.new_zeros((1,)), spatial_shapes.prod(1).cumsum(0)[:-1]))
        mask_flatten = torch.zeros(src_flatten.shape[:2], dtype=torch.bool, device=device)
        valid_ratios = torch.ones(bs, self.num_feature_levels, 2, device=device)

        predictions_class, predictions_mask = [], []
        interm_outputs = None

        if self.two_stage:
            output_memory, output_proposals = gen_encoder_output_proposals(
                src_flatten, mask_flatten, spatial_shapes)
            output_memory = self.enc_output_norm(self.enc_output(output_memory))
            enc_outputs_class_unselected = self.class_embed(output_memory)
            enc_outputs_coord_unselected = self._bbox_embed(output_memory) + output_proposals

            if s > 1:
                # Multi-frame: select ONE query set per bundle from the union of the S frames'
                # encoder proposals, then broadcast it to every frame. Query q is therefore the
                # same object hypothesis in all views; the anchor it starts from comes from the
                # frame that proposed it and is immediately re-derived per frame by the
                # mask-enhanced box init below (and refined per frame afterwards).
                hw = output_memory.shape[1]
                scores = enc_outputs_class_unselected.max(-1)[0].reshape(bs // s, s * hw)
                topk_proposals = torch.topk(scores, self.num_queries, dim=1)[1]     # [B, Q]
                mem_b = output_memory.reshape(bs // s, s * hw, self.hidden_dim)
                coord_b = enc_outputs_coord_unselected.reshape(bs // s, s * hw, 4)
                refpoint_embed_undetach = torch.gather(
                    coord_b, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4)
                ).repeat_interleave(s, dim=0)
                tgt_undetach = torch.gather(
                    mem_b, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim)
                ).repeat_interleave(s, dim=0)
            else:
                topk_proposals = torch.topk(
                    enc_outputs_class_unselected.max(-1)[0], self.num_queries, dim=1)[1]
                refpoint_embed_undetach = torch.gather(
                    enc_outputs_coord_unselected, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 4))
                tgt_undetach = torch.gather(
                    output_memory, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, self.hidden_dim))
            refpoint_embed = refpoint_embed_undetach.detach()

            if self.anchor_3d:
                # 3D two-stage costs a gather: every memory token already has a cached 3D
                # position, so the selected proposals' positions ARE the anchors. The radius
                # starts at the same constant for every query and is refined per layer.
                xyz_mem = pyramid_token_xyz(token_xyz, spatial_shapes)   # [bs, sum_hw, 3]
                xyz_b = xyz_mem.reshape(bs // s, s * xyz_mem.shape[1], 3)
                anchor_xyz = torch.gather(
                    xyz_b, 1, topk_proposals.unsqueeze(-1).repeat(1, 1, 3)).detach()
                anchor = torch.cat(
                    [anchor_xyz, anchor_xyz.new_full((*anchor_xyz.shape[:2], 1), ANCHOR_LOG_R0)],
                    dim=-1)                                              # [B, Q, 4]

            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt_undetach.transpose(0, 1), mask_features)
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1) if self.learn_tgt \
                else tgt_undetach.detach()
            interm_outputs = {"pred_logits": outputs_class,
                              "pred_boxes": refpoint_embed_undetach.sigmoid(),
                              "pred_masks": outputs_mask}

            if self.initialize_box_type != "no":
                # mask-enhanced anchor init: the initial masks give much better boxes than the
                # encoder's coarse proposals (MaskDINO's 'bitmask' / 'mask2box').
                assert self.initial_pred
                flaten_mask = outputs_mask.detach().flatten(0, 1)
                h, w = outputs_mask.shape[-2:]
                refpoint_embed = box_ops.masks_to_boxes(flaten_mask > 0).to(device)
                refpoint_embed = box_ops.box_xyxy_to_cxcywh(refpoint_embed) / torch.as_tensor(
                    [w, h, w, h], dtype=torch.float, device=device)
                refpoint_embed = refpoint_embed.reshape(outputs_mask.shape[0],
                                                        outputs_mask.shape[1], 4)
                refpoint_embed = inverse_sigmoid(refpoint_embed)
        else:
            tgt = self.query_feat.weight[None].repeat(bs, 1, 1)
            refpoint_embed = self.query_embed.weight[None].repeat(bs, 1, 1)
            if self.anchor_3d:
                anchor = self.anchor_query_embed.weight[None].repeat(bs // s, 1, 1)

        tgt_mask = None
        mask_dict = None
        if self.dn != "no" and self.training:
            assert targets is not None
            input_query_label, input_query_bbox, tgt_mask, mask_dict = self.prepare_for_dn(
                targets, None, None, bs, device)
            if mask_dict is not None:
                tgt = torch.cat([input_query_label, tgt], dim=1)

        # prediction from the raw (matching + DN) queries, before any decoder layer
        if self.initial_pred:
            outputs_class, outputs_mask = self.forward_prediction_heads(
                tgt.transpose(0, 1), mask_features, self.training)
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)
        if self.dn != "no" and self.training and mask_dict is not None:
            refpoint_embed = torch.cat([input_query_bbox, refpoint_embed], dim=1)

        hs, references = self.decoder(
            tgt=tgt.transpose(0, 1),
            memory=src_flatten.transpose(0, 1),
            memory_key_padding_mask=mask_flatten,
            refpoints_unsigmoid=refpoint_embed.transpose(0, 1),
            level_start_index=level_start_index,
            spatial_shapes=spatial_shapes,
            valid_ratios=valid_ratios,
            tgt_mask=tgt_mask,
            frames_per_sample=s,
            num_shared_queries=self.num_queries,
            anchor=anchor,
            token_xyz=token_xyz,
        )
        for i, output in enumerate(hs):
            outputs_class, outputs_mask = self.forward_prediction_heads(
                output.transpose(0, 1), mask_features, self.training or (i == len(hs) - 1))
            predictions_class.append(outputs_class)
            predictions_mask.append(outputs_mask)

        if self.initial_pred:
            out_boxes = self.pred_box(references, hs, refpoint_embed.sigmoid())
            assert len(predictions_class) == self.num_layers + 1
        else:
            out_boxes = self.pred_box(references, hs)

        if mask_dict is not None:
            predictions_class, out_boxes, predictions_mask = self.dn_post_process(
                torch.stack(predictions_class), out_boxes, mask_dict, torch.stack(predictions_mask))
            predictions_class, predictions_mask = list(predictions_class), list(predictions_mask)
        elif self.training:
            # keep label_enc in the autograd graph even without DN (upstream does the same)
            predictions_class[-1] = predictions_class[-1] + 0.0 * self.label_enc.weight.sum()

        out = {
            "pred_logits": predictions_class[-1],
            "pred_masks": predictions_mask[-1],
            "pred_boxes": out_boxes[-1],
            "aux_outputs": self._set_aux_loss(predictions_class, predictions_mask, out_boxes),
        }
        if interm_outputs is not None:
            out["interm_outputs"] = interm_outputs
        return out, mask_dict

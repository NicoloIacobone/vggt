"""
The ScanNet200 instance taxonomy (docs/todo.md 6c/6d).

ScanNet200 is the SAME 1513 scans and the SAME annotations as ScanNet v2 — only the label
set differs: 200 raw ScanNet categories are evaluated instead of the 18 nyu40 classes of the
v2 benchmark. So the ScanNet200 column needs **no new data at all**: the existing
`scannet_3d_gt_val312.tar.zst` (mesh + superpoints + aggregation) plus this id list and the
`id` column of `scannetv2-labels.combined.tsv` are enough.

`VALID_CLASS_IDS_200` is the official list from ScanNet's `scannet200_constants.py`
(`VALID_CLASS_IDS_200`), in the official order. It is the same list SegVGGT evaluates
against (`SegVGGT/eval/scannet_utils.py::get_seg_label_mapping_scannet200`), which is where
this copy was taken from — the two agree element for element, and every id resolves in the
labels TSV (`tests/test_scannet200_taxonomy.py`).

Note the ids are **raw ScanNet label ids** (the TSV's `id` column, 1..1191), NOT nyu40 ids.
The two taxonomies overlap numerically in the low range and must never be mixed: `5` is
`chair` here and `chair` in nyu40 only by coincidence.

Our 19-class head cannot be class-aware on 200 classes, so every ScanNet200 number this
project reports is CLASS-AGNOSTIC (`docs/RESULTS.md` §7) — the setting FAST3DIS and IGGT
report in. What the ScanNet200 column actually varies against ScanNetv2 is therefore the
**GT instance set**: 200 categories admit far more instances per scene than 18, so recall
has a larger denominator. Unlike the v2 benchmark, ScanNet200 DOES include `wall` and
`floor` as valid classes, which is why the prediction side stops dropping them for this
dataset (`train/datasets3d.py`).
"""

# ruff: noqa: E501
VALID_CLASS_IDS_200 = (
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23,
    24, 26, 27, 28, 29, 31, 32, 33, 34, 35, 36, 38, 39, 40, 41, 42, 44, 45, 46,
    47, 48, 49, 50, 51, 52, 54, 55, 56, 57, 58, 59, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 82, 84, 86, 87, 88, 89, 90,
    93, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 110, 112,
    115, 116, 118, 120, 121, 122, 125, 128, 130, 131, 132, 134, 136, 138, 139,
    140, 141, 145, 148, 154, 155, 156, 157, 159, 161, 163, 165, 166, 168, 169,
    170, 177, 180, 185, 188, 191, 193, 195, 202, 208, 213, 214, 221, 229, 230,
    232, 233, 242, 250, 261, 264, 276, 283, 286, 300, 304, 312, 323, 325, 331,
    342, 356, 370, 392, 395, 399, 408, 417, 488, 540, 562, 570, 572, 581, 609,
    748, 776, 1156, 1163, 1164, 1165, 1166, 1167, 1168, 1169, 1170, 1171, 1172,
    1173, 1174, 1175, 1176, 1178, 1179, 1180, 1181, 1182, 1183, 1184, 1185,
    1186, 1187, 1188, 1189, 1190, 1191,
)

VALID_CLASS_IDS_200_SET = frozenset(VALID_CLASS_IDS_200)

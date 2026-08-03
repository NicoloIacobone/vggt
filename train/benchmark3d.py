"""
The official ScanNet 3D semantic-instance evaluator, vendored (docs/MASKDINO.md §9).

A faithful Python-3 port of `ScanNet/BenchmarkScripts/3d_evaluation/
evaluate_semantic_instance.py` + the pieces of `util_3d.py` it uses (fetched from
ScanNet@master 2026-08-01; upstream is Python 2). This is the code whose numbers SegVGGT
and FAST3DIS report against — porting it, rather than re-deriving AP, is what makes our 3D
numbers comparable. Changes are strictly interface-level:

  - operates on in-memory arrays instead of the benchmark's txt-file tree: a prediction is
    `{"mask": bool [V], "label_id": nyu40 int, "confidence": float}`, GT is the per-vertex
    id array in the `1000 * label + instance` encoding (`train/scannet3d.py::build_gt_ids`);
  - `pred_visited` is keyed by (scene, prediction index) instead of the mask filename;
  - `np.float`/`np.bool`/print statements modernised; evaluation parameters can be
    overridden ONLY for unit tests — defaults are the official ones and the eval entry
    point never touches them.

Everything score-relevant — overlap thresholds (0.50:0.05:0.95 + 0.25), the 100-vertex
minimum region, void/ignore handling, greedy confidence matching, the duplicate-detection
false-positive rule, and the exact precision-recall integration (including the artificial
first curve point and the convolution step widths) — is line-for-line the official logic.

The 18 evaluated classes live in `train/scannet3d.py` (BENCHMARK_CLASS_IDS/NAMES).
"""

from copy import deepcopy
from typing import Dict, List

import numpy as np

from train.scannet3d import BENCHMARK_CLASS_IDS, BENCHMARK_CLASS_NAMES

ID_TO_LABEL = dict(zip(BENCHMARK_CLASS_IDS, BENCHMARK_CLASS_NAMES))

# ---------- Evaluation params (official values; overridable only in tests) ---------- #
OVERLAPS = np.append(np.arange(0.5, 0.95, 0.05), 0.25)
MIN_REGION_SIZE = 100          # vertices
DISTANCE_THRESH = float("inf")
DISTANCE_CONF = -float("inf")


def get_instances(ids: np.ndarray) -> Dict[str, List[dict]]:
    """util_3d.get_instances: GT instances of the valid classes, keyed by class name."""
    instances: Dict[str, List[dict]] = {label: [] for label in BENCHMARK_CLASS_NAMES}
    for instance_id in np.unique(ids):
        if instance_id == 0:
            continue
        label_id = int(instance_id // 1000)
        if label_id in ID_TO_LABEL:
            instances[ID_TO_LABEL[label_id]].append({
                "instance_id": int(instance_id),
                "label_id": label_id,
                "vert_count": int((ids == instance_id).sum()),
                "med_dist": -1,
                "dist_conf": 0.0,
            })
    return instances


def assign_instances_for_scan(scene: str, preds: List[dict], gt_ids: np.ndarray,
                              min_region_size: int):
    """Original assign_instances_for_scan, with in-memory preds instead of mask files."""
    gt2pred = deepcopy(get_instances(gt_ids))
    for label in gt2pred:
        for gt in gt2pred[label]:
            gt["matched_pred"] = []
    pred2gt: Dict[str, List[dict]] = {label: [] for label in BENCHMARK_CLASS_NAMES}
    num_pred_instances = 0
    # mask of void labels in the groundtruth
    bool_void = np.logical_not(np.isin(gt_ids // 1000, BENCHMARK_CLASS_IDS))
    for pred in preds:
        label_id = int(pred["label_id"])
        if label_id not in ID_TO_LABEL:
            continue
        label_name = ID_TO_LABEL[label_id]
        pred_mask = np.asarray(pred["mask"], dtype=bool)
        if len(pred_mask) != len(gt_ids):
            raise ValueError(f"{scene}: pred mask has {len(pred_mask)} verts, GT has "
                             f"{len(gt_ids)}")
        num = int(np.count_nonzero(pred_mask))
        if num < min_region_size:
            continue  # skip if empty

        pred_instance = {
            "filename": (scene, num_pred_instances),   # pred_visited key
            "pred_id": num_pred_instances,
            "label_id": label_id,
            "vert_count": num,
            "confidence": float(pred["confidence"]),
            "void_intersection": int(np.count_nonzero(np.logical_and(bool_void, pred_mask))),
        }
        matched_gt = []
        for gt_num, gt_inst in enumerate(gt2pred[label_name]):
            intersection = int(np.count_nonzero(
                np.logical_and(gt_ids == gt_inst["instance_id"], pred_mask)))
            if intersection > 0:
                gt_copy = gt_inst.copy()
                pred_copy = pred_instance.copy()
                gt_copy["intersection"] = intersection
                pred_copy["intersection"] = intersection
                matched_gt.append(gt_copy)
                gt2pred[label_name][gt_num]["matched_pred"].append(pred_copy)
        pred_instance["matched_gt"] = matched_gt
        num_pred_instances += 1
        pred2gt[label_name].append(pred_instance)

    return gt2pred, pred2gt


def evaluate_matches(matches: dict, overlaps: np.ndarray, min_region_size: int
                     ) -> np.ndarray:
    """Original evaluate_matches: AP per (class, overlap threshold)."""
    ap = np.zeros((len(BENCHMARK_CLASS_NAMES), len(overlaps)), float)
    for oi, overlap_th in enumerate(overlaps):
        pred_visited = {}
        for m in matches:
            for label_name in BENCHMARK_CLASS_NAMES:
                for p in matches[m]["pred"][label_name]:
                    pred_visited[p["filename"]] = False
        for li, label_name in enumerate(BENCHMARK_CLASS_NAMES):
            y_true = np.empty(0)
            y_score = np.empty(0)
            hard_false_negatives = 0
            has_gt = False
            has_pred = False
            for m in matches:
                pred_instances = matches[m]["pred"][label_name]
                gt_instances = matches[m]["gt"][label_name]
                # filter groups in ground truth
                gt_instances = [gt for gt in gt_instances
                                if gt["instance_id"] >= 1000
                                and gt["vert_count"] >= min_region_size
                                and gt["med_dist"] <= DISTANCE_THRESH
                                and gt["dist_conf"] >= DISTANCE_CONF]
                if gt_instances:
                    has_gt = True
                if pred_instances:
                    has_pred = True

                cur_true = np.ones(len(gt_instances))
                cur_score = np.ones(len(gt_instances)) * (-float("inf"))
                cur_match = np.zeros(len(gt_instances), dtype=bool)
                # collect matches
                for gti, gt in enumerate(gt_instances):
                    found_match = False
                    for pred in gt["matched_pred"]:
                        # greedy assignments
                        if pred_visited[pred["filename"]]:
                            continue
                        overlap = float(pred["intersection"]) / (
                            gt["vert_count"] + pred["vert_count"] - pred["intersection"])
                        if overlap > overlap_th:
                            confidence = pred["confidence"]
                            # if already have a prediction for this gt,
                            # the lower-score one is automatically a false positive
                            if cur_match[gti]:
                                max_score = max(cur_score[gti], confidence)
                                min_score = min(cur_score[gti], confidence)
                                cur_score[gti] = max_score
                                # append false positive
                                cur_true = np.append(cur_true, 0)
                                cur_score = np.append(cur_score, min_score)
                                cur_match = np.append(cur_match, True)
                            else:
                                found_match = True
                                cur_match[gti] = True
                                cur_score[gti] = confidence
                                pred_visited[pred["filename"]] = True
                    if not found_match:
                        hard_false_negatives += 1
                # remove non-matched ground truth instances
                cur_true = cur_true[cur_match]
                cur_score = cur_score[cur_match]

                # collect non-matched predictions as false positive
                for pred in pred_instances:
                    found_gt = False
                    for gt in pred["matched_gt"]:
                        overlap = float(gt["intersection"]) / (
                            gt["vert_count"] + pred["vert_count"] - gt["intersection"])
                        if overlap > overlap_th:
                            found_gt = True
                            break
                    if not found_gt:
                        num_ignore = pred["void_intersection"]
                        for gt in pred["matched_gt"]:
                            # group?
                            if gt["instance_id"] < 1000:
                                num_ignore += gt["intersection"]
                            # small ground truth instances
                            if (gt["vert_count"] < min_region_size
                                    or gt["med_dist"] > DISTANCE_THRESH
                                    or gt["dist_conf"] < DISTANCE_CONF):
                                num_ignore += gt["intersection"]
                        proportion_ignore = float(num_ignore) / pred["vert_count"]
                        # if not ignored append false positive
                        if proportion_ignore <= overlap_th:
                            cur_true = np.append(cur_true, 0)
                            cur_score = np.append(cur_score, pred["confidence"])

                # append to overall results
                y_true = np.append(y_true, cur_true)
                y_score = np.append(y_score, cur_score)

            # compute average precision
            if has_gt and has_pred:
                # sorting and cumsum
                score_arg_sort = np.argsort(y_score)
                y_score_sorted = y_score[score_arg_sort]
                y_true_sorted = y_true[score_arg_sort]
                y_true_sorted_cumsum = np.cumsum(y_true_sorted)

                # unique thresholds
                (thresholds, unique_indices) = np.unique(y_score_sorted, return_index=True)
                num_prec_recall = len(unique_indices) + 1

                # prepare precision recall
                num_examples = len(y_score_sorted)
                num_true_examples = y_true_sorted_cumsum[-1] if num_examples > 0 else 0
                precision = np.zeros(num_prec_recall)
                recall = np.zeros(num_prec_recall)

                # deal with the first point
                y_true_sorted_cumsum = np.append(y_true_sorted_cumsum, 0)
                # deal with remaining
                for idx_res, idx_scores in enumerate(unique_indices):
                    cumsum = y_true_sorted_cumsum[idx_scores - 1]
                    tp = num_true_examples - cumsum
                    fp = num_examples - idx_scores - tp
                    fn = cumsum + hard_false_negatives
                    p = float(tp) / (tp + fp)
                    r = float(tp) / (tp + fn)
                    precision[idx_res] = p
                    recall[idx_res] = r

                # first point in curve is artificial
                precision[-1] = 1.0
                recall[-1] = 0.0

                # compute average of precision-recall curve
                recall_for_conv = np.copy(recall)
                recall_for_conv = np.append(recall_for_conv[0], recall_for_conv)
                recall_for_conv = np.append(recall_for_conv, 0.0)
                stepWidths = np.convolve(recall_for_conv, [-0.5, 0, 0.5], "valid")
                # integrate is now simply a dot product
                ap_current = np.dot(precision, stepWidths)
            elif has_gt:
                ap_current = 0.0
            else:
                ap_current = float("nan")
            ap[li, oi] = ap_current
    return ap


def compute_averages(aps: np.ndarray, overlaps: np.ndarray) -> dict:
    """Original compute_averages: AP (0.5:0.95), AP50, AP25, overall and per class."""
    o50 = np.where(np.isclose(overlaps, 0.5))
    o25 = np.where(np.isclose(overlaps, 0.25))
    o_all_but25 = np.where(np.logical_not(np.isclose(overlaps, 0.25)))
    avg_dict = {
        "all_ap": np.nanmean(aps[:, o_all_but25]),
        "all_ap_50%": np.nanmean(aps[:, o50]),
        "all_ap_25%": np.nanmean(aps[:, o25]),
        "classes": {},
    }
    for li, label_name in enumerate(BENCHMARK_CLASS_NAMES):
        avg_dict["classes"][label_name] = {
            "ap": np.average(aps[li, o_all_but25]),
            "ap50%": np.average(aps[li, o50]),
            "ap25%": np.average(aps[li, o25]),
        }
    return avg_dict


def evaluate(preds_by_scene: Dict[str, List[dict]], gt_by_scene: Dict[str, np.ndarray],
             overlaps: np.ndarray = OVERLAPS, min_region_size: int = MIN_REGION_SIZE
             ) -> dict:
    """
    The official evaluation over a set of scenes.

    preds_by_scene: scene -> list of {"mask": bool [V], "label_id": nyu40, "confidence"}.
    gt_by_scene:    scene -> per-vertex id [V], `1000 * nyu40_label + instance`, 0 = none.

    Returns {"all_ap", "all_ap_50%", "all_ap_25%", "classes": {name: {"ap", "ap50%",
    "ap25%"}}} — per-class values are NaN where the class has no GT in any scene.
    `overlaps` / `min_region_size` exist for unit tests only; results quoted anywhere must
    use the defaults.
    """
    matches = {}
    for scene, gt_ids in gt_by_scene.items():
        gt2pred, pred2gt = assign_instances_for_scan(
            scene, preds_by_scene.get(scene, []), np.asarray(gt_ids), min_region_size)
        matches[scene] = {"gt": gt2pred, "pred": pred2gt}
    aps = evaluate_matches(matches, overlaps, min_region_size)
    return compute_averages(aps, overlaps)


def format_results(avgs: dict) -> str:
    """The official per-class results table, as a string."""
    lines = ["#" * 64,
             "{:<18}{:>15}{:>15}{:>15}".format("what", "AP", "AP_50%", "AP_25%"),
             "#" * 64]
    for label_name in BENCHMARK_CLASS_NAMES:
        c = avgs["classes"][label_name]
        lines.append("{:<18}{:>15.3f}{:>15.3f}{:>15.3f}".format(
            label_name, c["ap"], c["ap50%"], c["ap25%"]))
    lines.append("-" * 64)
    lines.append("{:<18}{:>15.3f}{:>15.3f}{:>15.3f}".format(
        "average", avgs["all_ap"], avgs["all_ap_50%"], avgs["all_ap_25%"]))
    return "\n".join(lines)

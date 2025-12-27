"""
Evaluate long-form repaired motions stored in `sample.fix_globsmpl_long_smart` or
similar scripts that output `motion_fix` lists covering entire clips.
"""

import os
import numpy as np
import torch
from argparse import ArgumentParser
from tqdm import tqdm

from data_loaders.amasstools.extract_joints import extract_joints_smpldata
from data_loaders.amasstools.smplh_layer import SMPLH
from eval.eval_motion import (
    compute_foot_sliding_wrapper,
    compute_jitter_wrapper,
    compute_label_metrics,
    compute_skating_ratio_wrapper,
    compute_gpenetration_wrapper,
    compute_rte_wrapper,
    compute_jpe_wrapper,
    compute_error_accel_wrapper,
    tmr_m2m_score_wrapper,
)


def parse_args():
    parser = ArgumentParser(description="Evaluate long-form motion réparations.")
    parser.add_argument("--data_path", type=str, required=True, help="Predictions .npy path.")
    parser.add_argument(
        "--gt_data_path",
        type=str,
        default=None,
        help="Optional GT .npy (from collect_long_ground_truth.py) for MPJPE/RTE.",
    )
    parser.add_argument(
        "--motiontypes",
        nargs="+",
        default=["motion_fix"],
        help="Keys inside predictions to evaluate.",
    )
    parser.add_argument(
        "--force_redo",
        action="store_true",
        help="Recompute joints even if cached.",
    )
    parser.add_argument(
        "--fps",
        default=20,
        type=int,
        help="Frames per second used when extracting joints.",
    )
    parser.add_argument(
        "--smplh_folder",
        default="data_loaders/amasstools/deps/smplh",
        type=str,
        help="Path to the SMPLH layer sources.",
    )
    parser.add_argument(
        "--device",
        default="cuda",
        type=str,
        help="Device to allocate SMPLH (falls back to CPU if CUDA unavailable).",
    )
    return parser.parse_args()


def build_joints_list(
    *,
    motions,
    lengths,
    smplh,
    root_offset,
    device,
    fps,
    cache_path=None,
    desc="motions",
    force_redo=False,
):
    if cache_path and os.path.exists(cache_path) and not force_redo:
        return np.load(cache_path, allow_pickle=True)["joints"]

    joints = []
    for idx, smpldata in tqdm(enumerate(motions), desc=desc, total=len(motions)):
        output = extract_joints_smpldata(
            smpldata=smpldata,
            fps=fps,
            value_from="smpl",
            smpl_layer=smplh,
            root_offset=root_offset,
            device=device,
        )
        joints.append(output["joints"][:, :22])

    if cache_path:
        np.savez(cache_path, joints=joints)
    return joints


def main():
    args = parse_args()
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA unavailable, falling back to CPU.")
        device = "cpu"

    datas = np.load(args.data_path, allow_pickle=True).item()
    lengths = datas["lengths"]

    root_offset = np.array([-0.00179506, -0.22333382, 0.02821918])
    smplh = SMPLH(
        path=args.smplh_folder,
        jointstype="both",
        input_pose_rep="axisangle",
        gender="neutral",
    )
    smplh.to(device)

    def _normalize_label_list(raw_list):
        """
        将 label 列表规范化为 [num_seq][T] 的列表，兼容 batch/单序列两种形状。
        """
        out = []
        for item in raw_list:
            arr = np.array(item)
            if arr.ndim == 1:
                out.append(arr)
            elif arr.ndim == 2:
                for row in arr:
                    out.append(np.array(row))
            else:
                raise ValueError(f"Unsupported label shape {arr.shape}")
        return out

    if "gt_labels" in datas and "label" in datas:
        assert len(datas["gt_labels"]) == len(datas["label"])
        pred_labels = _normalize_label_list(datas["label"])
        gt_labels = _normalize_label_list(datas["gt_labels"])
        assert len(pred_labels) == len(gt_labels) == len(lengths), "label 与 lengths 数量不一致"
        acc, precision, recall, f1 = compute_label_metrics(
            preds=pred_labels,
            gts=gt_labels,
            lengths=lengths,
        )
        print(
            f"Label prediction: Accuracy: {acc:.5f}; Precision: {precision:.5f}, "
            f"Recall: {recall:.5f}, f1-score: {f1:.5f}"
        )

    gt_joints = None
    if args.gt_data_path:
        gt_datas = np.load(args.gt_data_path, allow_pickle=True).item()
        assert len(gt_datas["lengths"]) == len(lengths)
        gt_cache = args.gt_data_path.replace(".npy", "_motionjoints.npz")
        gt_joints = build_joints_list(
            motions=gt_datas["motion"],
            lengths=gt_datas["lengths"],
            smplh=smplh,
            root_offset=root_offset,
            device=device,
            fps=args.fps,
            cache_path=gt_cache,
            desc="gt motions",
            force_redo=args.force_redo,
        )

    for motiontype in args.motiontypes:
        if motiontype not in datas:
            print(f"Motion type {motiontype} does not exist in results file")
            continue

        joints_cache_path = args.data_path.replace(".npy", f"_{motiontype}joints.npz")
        smpldatas = datas[motiontype]
        assert len(lengths) == len(smpldatas)

        joints = build_joints_list(
            motions=smpldatas,
            lengths=lengths,
            smplh=smplh,
            root_offset=root_offset,
            device=device,
            fps=args.fps,
            cache_path=joints_cache_path,
            desc=f"{motiontype} joints",
            force_redo=args.force_redo,
        )

        motion_slide_dis = compute_foot_sliding_wrapper(joints, lengths, upaxis=2, ankle_h=0.1)
        motion_jitter = compute_jitter_wrapper(joints, lengths)
        motion_skating_ratio = compute_skating_ratio_wrapper(joints, lengths)
        motion_gp_ratio, motion_gp_dist = compute_gpenetration_wrapper(joints, lengths)

        print(
            f"Motion from {motiontype}"
            f"\n Foot Sliding Dist (mm): {motion_slide_dis:.5f}"
            f"\n Skating Ratio (%): {motion_skating_ratio * 100:.5f}"
            f"\n Ground Penetration Ratio (%): {motion_gp_ratio * 100:.5f}"
            f"\n Ground Penetration Dist (mm): {motion_gp_dist:.5f}"
            f"\n Jittering measurement (x10 m/s^3): {motion_jitter.mean():.5f}"
        )

        if gt_joints is not None:
            mpjpe = compute_jpe_wrapper(gt_joints, joints, lengths)
            gb_mpjpe = compute_jpe_wrapper(gt_joints, joints, lengths, local=False)
            rte = compute_rte_wrapper(gt_joints, joints, lengths)
            raw_rte = compute_rte_wrapper(gt_joints, joints, lengths, align=False)
            error_accel = compute_error_accel_wrapper(gt_joints, joints, lengths)
            m2m_score, m2m_r1, m2m_r3 = tmr_m2m_score_wrapper(
                joints, gt_joints, lengths, y_is_z_axis=False, device="cpu"
            )
            print(
                f"Motion from {motiontype}"
                f"\n MPJPE (cm): {mpjpe:.5f}"
                f"\n GBMPJPE (cm): {gb_mpjpe:.5f}"
                f"\n RTE (m): {rte:.5f}"
                f"\n Unaligned RTE (m): {raw_rte:.5f}"
                f"\n Accel Error (m/s^2): {error_accel:.5f}"
                f"\n m2m_score: {m2m_score:.5f}"
                f"\n m2m_r1: {m2m_r1 * 100:.5f}"
                f"\n m2m_r3: {m2m_r3 * 100:.5f}"
            )


if __name__ == "__main__":
    main()


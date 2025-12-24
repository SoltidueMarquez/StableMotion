"""
Segment-wise repair pipeline that cuts each motion into fixed-length chunks
and applies the detect/fix loop from `sample/fix_globsmpl.py` to every chunk.
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch

from ema_pytorch import EMA
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata
from data_loaders.get_data import get_dataset_loader
from sample.fix_globsmpl import detect_labels, fix_motion
from sample.utils import build_output_dir, prepare_cond_fn
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import add_base_options, add_sampling_options, parse_and_load_from_model
from utils import dist_util


def cut_args():
    """
    构造并解析命令行参数，结合基础/采样选项以及分段修复所需的控制参数。
    解析结束后通过 `parse_and_load_from_model` 会再读取 checkpoint 的 args.json，
    以确保 dataset/model/diffusion 的配置与训练时一致。
    """
    parser = ArgumentParser(description="Fix motions by splitting them into fixed-length chunks.")
    add_base_options(parser)
    add_sampling_options(parser)
    parser.add_argument(
        "--segment_length",
        type=int,
        default=100,
        help="每个待 fix 的 segment 长度（帧数量），必须为正；越小运行越多次检测/修复。",
    )
    parser.add_argument(
        "--segment_stride",
        type=int,
        default=0,
        help="Segment 切片的步长（<= segment_length）；默认等于 segment_length，可设小于以制造重叠。",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="",
        help="可选：指定 motion 文件夹（支持 .npz/.npy），会优先遍历该目录而非 split。",
    )
    parser.add_argument(
        "--use_precomputed_cleanup",
        action="store_true",
        help="是否将在检测分支输出的 label/re_sample 特征传给 ensemble clean-up 逻辑。",
    )
    return parse_and_load_from_model(parser)


def compute_segment_starts(seq_len: int, segment_length: int, stride: int):
    """
    生成按顺序切分的 segment 起点，确保尾部也会被覆盖。
    - 以 stride 为步长从起点 0 依次切片，直到整个序列被覆盖；
    - 若最后一个窗口右侧不足 segment_length，则再补一个起点使其右对齐末尾；
    - 返回 sorted(set()) 保证没有重复起点。
    """
    if seq_len <= 0:
        return []
    starts = []
    cursor = 0
    while cursor < seq_len:
        starts.append(cursor)
        cursor += stride
    if starts:
        last = starts[-1]
        if last + segment_length < seq_len:
            starts.append(max(seq_len - segment_length, 0))
    else:
        starts = [0]
    return sorted(set(starts))


@torch.no_grad()
def main():
    args = cut_args()
    if args.segment_length <= 0:
        raise ValueError("segment_length 必须为正整数")
    segment_stride = args.segment_stride if args.segment_stride > 0 else args.segment_length
    segment_stride = max(1, min(segment_stride, args.segment_length))
    folder_path = args.folder_path or None

    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    out_path = build_output_dir(args)

    # loader configured 为 sequential，辅助逐个 sample 按顺序切分；folder_path 不为空时会直接遍历给定目录

    # 构造按顺序 batch 的 loader：folder_path 提供时遍历该目录，否则基于 test_benchmark split
    data = get_dataset_loader(
        name="globsmpl",
        batch_size=args.batch_size,
        split="test_benchmark",
        data_dir=args.testdata_dir,
        normalizer_dir=args.normalizer_dir,
        shuffle=False,
        sample_mode="sequential",
        sequence_window_size=None,
        sequence_stride=None,
        folder_path=folder_path,
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer

    # 打印采样器信息并构建模型

    print("创建 model and diffusion...")
    print("使用采样器: ", args.ts_respace)
    _model, diffusion = create_model_and_diffusion(args)
    model = EMA(_model, include_online_model=False) if args.use_ema else _model

    print(f"加载 checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()

    # Optional 的 classifier guidance：cond_fn 为 None 时不启用
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)
    if cond_fn is not None:
        cond_fn.keywords["model"] = model

    all_motions = []
    all_lengths = []
    all_input_motions_vec = []
    gt_labels_buf = []
    all_motions_fix = []
    all_fix_motions_vec = []
    labels = []
    segment_meta = []

    total_segments = 0
    total_samples = 0
    stop = False

    # 逐 batch 遍历 loader，包含 collect_dataset 与 fix 模式

    for i, input_batch in enumerate(data):
        input_batch = {
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_batch.items()
        }
        input_motions = input_batch["x"]  # [B, C, N]
        attention_mask = input_batch["mask"].squeeze().bool()
        if attention_mask.dim() == 1:
            attention_mask = attention_mask.unsqueeze(0)
        length = input_batch["length"]

        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())
        gt_labels = (temp_sample[..., -1] > 0.5).numpy()

        if args.collect_dataset:
            for b in range(input_motions.shape[0]):
                seq_len = int(length[b].item())
                if seq_len == 0:
                    continue
                # 计算当前样本所有 segment 起点，用原始 GT 直接拼接 SMPL 结构
                for start in compute_segment_starts(seq_len, args.segment_length, segment_stride):
                    end = min(start + args.segment_length, seq_len)
                    if end <= start:
                        continue
                    segment_feats = temp_sample[b, start:end, :-1]
                    all_motions.append(globsmplrifkefeats_to_smpldata(segment_feats))
                    all_lengths.append(end - start)
                    total_segments += 1
                    print(f"Collected {total_segments} segments (length {end-start})")
                    if args.num_samples and total_segments >= args.num_samples:
                        stop = True
                        break
                if stop:
                    break
            if stop:
                break
            continue

        for b in range(input_motions.shape[0]):
            seq_len = int(length[b].item())
            if seq_len == 0:
                continue
            seq_feats = input_motions[b, :, :seq_len]
            seq_mask = attention_mask[b : b + 1, :seq_len]
            sample_gt_labels = gt_labels[b, :seq_len]

            segment_starts = compute_segment_starts(seq_len, args.segment_length, segment_stride)
            for seg_id, start in enumerate(segment_starts):
                end = min(start + args.segment_length, seq_len)
                if end <= start:
                    continue
                seg_len = end - start
                seg_input = seq_feats[:, start:end].unsqueeze(0)
                seg_mask = seq_mask[:, start:end]
                length_tensor = torch.tensor([seg_len], device=device, dtype=length.dtype)

                # 检测分支只修复标签通道，同时保留上下文中的现有 motion
                det_out = detect_labels(
                    model=model,
                    diffusion=diffusion,
                    args=args,
                    input_motions=seg_input,
                    length=length_tensor,
                    attention_mask=seg_mask,
                    motion_normalizer=motion_normalizer,
                )
                label = det_out["label"].to(device)
                fix_kwargs = dict(
                    model=model,
                    diffusion=diffusion,
                    args=args,
                    input_motions=seg_input,
                    length=length_tensor,
                    attention_mask=seg_mask,
                    motion_normalizer=motion_normalizer,
                    label=label,
                    re_sample_det_feats=det_out["re_sample_det_feats"],
                    cond_fn=cond_fn,
                )
                if args.use_precomputed_cleanup:
                    fix_kwargs.update(
                        label_for_cleanup=label,
                        re_sample_det_feats_for_cleanup=det_out["re_sample_det_feats"],
                    )
                fix_out = fix_motion(**fix_kwargs)

                # 生成当前 segment 的修复结果，去掉 label 通道并转为 SMPL 表示
                sample_fix_feats = fix_out["sample_fix_feats"][0]
                det_feats = det_out["re_sample_det_feats"][0]

                det_motion_denorm = motion_normalizer.inverse(det_feats.transpose(0, 1).cpu())
                fix_motion_denorm = motion_normalizer.inverse(sample_fix_feats.transpose(0, 1).cpu())

                all_motions.append(globsmplrifkefeats_to_smpldata(det_motion_denorm[..., :-1]))
                all_motions_fix.append(globsmplrifkefeats_to_smpldata(fix_motion_denorm[..., :-1]))
                labels.append(label[0, :seg_len].cpu().numpy())
                gt_labels_buf.append(sample_gt_labels[start:end])
                all_lengths.append(seg_len)
                all_input_motions_vec.append(seg_input[0].transpose(0, 1).cpu().numpy())
                all_fix_motions_vec.append(sample_fix_feats.transpose(0, 1).cpu().numpy())
                segment_meta.append(
                    {
                        "sample_index": total_samples,
                        "segment_index": seg_id,
                        "start_frame": start,
                        "end_frame": end,
                    }
                )

                total_segments += 1
                print(
                    f"Processed segment {total_segments} "
                    f"(sample {total_samples}, start {start}, len {seg_len})"
                )

                if args.num_samples and total_segments >= args.num_samples:
                    stop = True
                    break
            if stop:
                break
            total_samples += 1
        if stop:
            break

    os.makedirs(out_path, exist_ok=True)
    # 若开启 collect_dataset，则只保存截取的 GT Motion，跳过修复集成结果
    if args.collect_dataset:
        npy_path = os.path.join(out_path, "results_collected_cut.npy")
        print(f"saving collected motion file to [{npy_path}]")
        np.save(npy_path, {"motion": all_motions, "lengths": np.array(all_lengths, dtype=np.int32)})
        with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
            fw.write("\n".join([str(l) for l in all_lengths]))
        return

    npy_path = os.path.join(out_path, "results_cut.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "motion_fix": all_motions_fix,
            "label": labels,
            "gt_labels": gt_labels_buf,
            "lengths": np.array(all_lengths, dtype=np.int32),
            "segment_meta": segment_meta,
            "all_fix_motions_vec": all_fix_motions_vec,
            "all_input_motions_vec": all_input_motions_vec,
        },
    )
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:
        fw.write("\n".join([str(l) for l in all_lengths]))

    # 调用现有评估脚本产生 metrics/logs
    os.system(f"python -m eval.eval_scripts  --data_path {npy_path} --force_redo")


if __name__ == "__main__":
    main()


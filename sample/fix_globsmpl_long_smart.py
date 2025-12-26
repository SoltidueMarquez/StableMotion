"""
Smart detection pipeline for sliding-window-long globsmpl motions.

与 `sample/fix_globsmpl_long.py` 保持类似的滑窗思路：使用“history + target + future”的窗口结构，
但每次只真正写入 target 部分的标签，history/future 只是提供上下文，不会被标注或覆盖。
由于每次滑窗的步长正好等于 target 的长度，所有帧只会被写入一次，因此无需在多个窗口间取平均。
"""

import os
from argparse import ArgumentParser

import math

import numpy as np
import torch

from ema_pytorch import EMA

from sample.fix_globsmpl import detect_labels, fix_motion
from sample.utils import build_output_dir, prepare_cond_fn
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata
from data_loaders.get_data import get_dataset_loader
from utils.fixseed import fixseed
from utils.model_util import create_model_and_diffusion
from utils.parser_util import add_base_options, add_sampling_options, parse_and_load_from_model
from utils import dist_util


def smart_args():
    """
    解析当前 smart detection 脚本所需的命令行参数。
    """
    parser = ArgumentParser(description="Smart long-form detection using sliding windows.")
    add_base_options(parser)
    add_sampling_options(parser)
    parser.add_argument(
        "--target_window",
        type=int,
        default=100,
        help="兼容旧参数：默认的 target 窗口大小，若未指定 detect/fix 单独值即使用该值。",
    )
    parser.add_argument(
        "--history_frames",
        dest="history_frames",
        type=int,
        default=20,
        help="兼容旧参数：默认的 history/future 帧数，若未指定 detect/fix 单独值即使用该值。",
    )
    parser.add_argument(
        "--detect_target_window",
        type=int,
        default=None,
        help="检测分支独立的 target 窗口大小（覆盖 --target_window）。",
    )
    parser.add_argument(
        "--detect_history_frames",
        type=int,
        default=None,
        help="检测分支独立的 history/future 帧数（覆盖 --history_frames）。",
    )
    parser.add_argument(
        "--fix_target_window",
        type=int,
        default=None,
        help="修复分支独立的 target 窗口大小（覆盖 --target_window）。",
    )
    parser.add_argument(
        "--fix_history_frames",
        type=int,
        default=None,
        help="修复分支独立的 history/future 帧数（覆盖 --history_frames）。",
    )
    parser.add_argument(
        "--folder_path",
        type=str,
        default="",
        help="可选：直接遍历某个文件夹，跳过 split 列表。",
    )
    parser.add_argument(
        "--debug_print_intervals",
        action="store_true",
        help="调试用：打印每条序列的坏帧闭区间。",
    )
    parser.add_argument(
        "--no-reverse-pass-after-forward",
        dest="reverse_pass_after_forward",
        action="store_false",
        help="跳过正向修复后的反向修复。",
    )
    parser.add_argument(
        "--use-window-detect",
        dest="use_window_detect",
        action="store_true",
        help="使用滑窗检测法，默认切片检测",
    )
    parser.set_defaults(reverse_pass_after_forward=True)
    parser.add_argument(
        "--use_precomputed_cleanup",
        action="store_true",
        help="在 run_section 中复用 detection 结果作为进一步上下文（默认关闭）。",
    )
    args = parse_and_load_from_model(parser)
    args.detect_target_window = (
        args.detect_target_window if args.detect_target_window is not None else args.target_window
    )
    args.detect_history_frames = (
        args.detect_history_frames if args.detect_history_frames is not None else args.history_frames
    )
    args.fix_target_window = (
        args.fix_target_window if args.fix_target_window is not None else args.target_window
    )
    args.fix_history_frames = (
        args.fix_history_frames if args.fix_history_frames is not None else args.history_frames
    )
    return args


def compute_target_starts(seq_len: int, target_size: int):
    """
    依据 target_size 生成覆盖整条序列的起点列表（stride=target_size）。
    """
    if seq_len <= 0:
        return []
    starts = list(range(0, seq_len, target_size))
    if not starts:
        starts = [0]
    return starts


def compute_segment_starts(seq_len: int, segment_length: int, stride: int):
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

def build_corrupt_intervals(mask):
    """
    将 bool 掩码转换成 [start,end] 闭区间列表（1-based）。
    """
    intervals = []
    start = None
    for idx, flag in enumerate(mask):
        if flag:
            if start is None:
                start = idx
        elif start is not None:
            intervals.append((start + 1, idx))
            start = None
    if start is not None:
        intervals.append((start + 1, len(mask)))
    return intervals


def build_corrupt_intervals_zero(mask):
    """
    把 bool 掩码转换成 [start,end) 的闭区间列表，便于按区间修复。
    """
    intervals = []
    start = None
    for idx, flag in enumerate(mask):
        if flag:
            if start is None:
                start = idx
        elif start is not None:
            intervals.append((start, idx))
            start = None
    if start is not None:
        intervals.append((start, len(mask)))
    return intervals

def format_intervals(intervals):
    """
    把 [start,end] 区间列表拼成易读的字符串，比如 "[[1,5],[10,12]]"。
    """
    if not intervals:
        return "[]"
    return "[" + ",".join(f"[{s},{e}]" for s, e in intervals) + "]"


@torch.no_grad()
def detect_sequence_quality_labels_window(
    *,
    model,
    diffusion,
    args,
    motion_normalizer,
    input_sequence,
    length,
    device,
    cond_fn=None,
):
    """
    使用 history/target/future 窗口结构遍历整条序列，仅对 target 部分执行 detect_labels。
    """
    #region 预处理：提取序列长度、通道数，处理空序列
    seq_len = int(length.item())  # 将长度 tensor 提取成 python int，方便后续切片
    nfeats = input_sequence.shape[0]  # 每帧通道数（包含标签）
    if seq_len == 0: return torch.zeros((0,), dtype=torch.bool, device=device)  # 空序列直接返回空标签

    seq_data = input_sequence[:, :seq_len]  # 丢弃 padding，只保留真实有效帧
    target_size = args.target_window  # target 区域的帧数（同时也是 stride）
    history_frames = args.history_frames  # history/future 各自的帧数
    future_frames = history_frames  # 强制未来帧与 history 保持对称
    #endregion

    #region 初始化：准备标签缓冲区和窗口起点列表
    label_buffer = torch.zeros((seq_len,), dtype=torch.bool, device=device)  # 最终标签缓冲区
    window_starts = compute_target_starts(seq_len, target_size)  # 计算 target 区域覆盖整条序列的起点列表
    #endregion

    # 遍历每个 target 区域，逐帧执行 detect_labels
    for start in window_starts:
        target_end = min(seq_len, start + target_size)  # 计算 target 区域结束位置（不超过序列长度）
        target_len = target_end - start
        if target_len <= 0:  # target 区域为空，跳过
            continue

        #region 计算可用的 history 和 future 长度：从当前 target 起点往前推 history 帧，往后推 future 帧
        history_start = max(0, start - history_frames)  # history 起始位置（不超过 0）
        history_len = start - history_start  # history 帧数（可能小于 history_frames）
        missing_history = max(0, history_frames - history_len)
        future_end = min(seq_len, target_end + future_frames + missing_history)  # future 结束位置（不超过序列长度）
        future_len = future_end - target_end    # future 帧数（可能小于 future_frames + missing_history）

        # history缺失的部分需要用future填补
        window_future = future_frames + missing_history
        window_frames = history_frames + target_size + window_future  # 窗口总帧数（history + target + future）
        window_input = torch.zeros((1, nfeats, window_frames), device=device)  # 输入张量：1 batch, nfeats 通道, window_frames 帧
        window_attn = torch.zeros((1, window_frames), dtype=torch.bool, device=device)  # attention mask：全 False，表示所有帧都有效
        #endregion

        #region 设置attentionMask
        # 将 history 对齐到 history 区间的右侧，attention_mask 只标记有效帧
        if history_len > 0:
            history_offset = history_frames - history_len
            window_input[:, :, history_offset:history_frames] = seq_data[
                :, history_start:start
            ].unsqueeze(0)
            window_attn[:, history_offset:history_frames] = True

        # target 部分直接占据中间区域
        target_offset = history_frames
        window_input[:, :, target_offset : target_offset + target_len] = seq_data[
            :, start:target_end
        ].unsqueeze(0)
        window_attn[:, target_offset : target_offset + target_len] = True

        # future 也只填入实际帧数，超出部分留 False
        future_offset = target_offset + target_len
        if future_len > 0:
            window_input[:, :, future_offset : future_offset + future_len] = seq_data[
                :, target_end:future_end
            ].unsqueeze(0)
            window_attn[:, future_offset : future_offset + future_len] = True
        #endregion

        length_tensor = torch.tensor(
            [history_len + target_len + future_len], device=device, dtype=length.dtype
        )
        #region 执行检测 直接将（history+target+future）作为输入，但是只取target的输出
        det_out = detect_labels(
            model=model,
            diffusion=diffusion,
            args=args,
            input_motions=window_input,
            length=length_tensor,
            attention_mask=window_attn,
            motion_normalizer=motion_normalizer,
        )

        label = det_out["label"][0].to(device)

        # 设置label_buffer只更新target部分的输出，无视history和future
        target_label_slice = label[target_offset : target_offset + target_len]
        label_buffer[start:target_end] = target_label_slice

        if args.debug_print_intervals:
            corrupt_intervals = format_intervals(
                build_corrupt_intervals(target_label_slice.cpu().numpy().astype(bool))
            )
            print(
                f"检测窗口: start={start}, end={target_end}, history_len={history_len}, "
                f"future_len={future_len}, detect_len={target_len}, corrupt_intervals={corrupt_intervals}"
            )
        #endregion

    return label_buffer

@torch.no_grad()
def detect_sequence_quality_labels_segment(
    *,
    model,
    diffusion,
    args,
    motion_normalizer,
    input_sequence,
    length,
    device,
    cond_fn=None,
):
    """
    直接切分整条序列，执行 detect_labels。
    """
    #region 预处理：提取序列长度、通道数，处理空序列
    seq_len = int(length.item())  # 将长度 tensor 提取成 python int，方便后续切片
    nfeats = input_sequence.shape[0]  # 每帧通道数（包含标签）
    if seq_len == 0: return torch.zeros((0,), dtype=torch.bool, device=device)  # 空序列直接返回空标签

    seq_data = input_sequence[:, :seq_len]  # 丢弃 padding，只保留真实有效帧
    target_size = args.detect_target_window + args.detect_history_frames*2  # target+history+future 区域的帧数（同时也是 stride）
    #endregion

    #region 初始化：准备标签缓冲区和 segment 起点列表
    label_buffer = torch.zeros((seq_len,), dtype=torch.bool, device=device)
    segment_length = target_size
    segment_stride = segment_length
    segment_starts = compute_segment_starts(seq_len, segment_length, segment_stride)
    #endregion

    for seg_id, start in enumerate(segment_starts):
        end = min(start + segment_length, seq_len)
        if end <= start:
            continue
        seg_len = end - start
        seg_input = seq_data[:, start:end].unsqueeze(0)
        length_tensor = torch.tensor([seg_len], device=device, dtype=length.dtype)
        seg_mask = torch.ones((1, seg_len), dtype=torch.bool, device=device)

        det_out = detect_labels(
            model=model,
            diffusion=diffusion,
            args=args,
            input_motions=seg_input,
            length=length_tensor,
            attention_mask=seg_mask,
            motion_normalizer=motion_normalizer,
        )
        label = det_out["label"][:, :seg_len].to(device)
        label_buffer[start:end] = label[0]

        if args.debug_print_intervals:
            flat_labels = label[0].cpu().numpy().astype(bool)
            corrupt_intervals = format_intervals(
                build_corrupt_intervals(flat_labels)
            )
            print(
                f"检测 segment {seg_id}: start={start}, end={end}, len={seg_len}, "
                f"corrupt_intervals={corrupt_intervals}"
            )

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
        # if args.use_precomputed_cleanup:
        #     fix_kwargs.update(
        #         label_for_cleanup=label,
        #         re_sample_det_feats_for_cleanup=det_out["re_sample_det_feats"],
        #     )
        fix_motion(**fix_kwargs)

    return label_buffer


def gather_context(
    *,
    fix_out,
    window_input,
    window_attn,
    history_frames,
    history_start,
    history_len,
    future_start,
    future_len,
    window_target_len,
):
    """
    填充 history/future 段数据，并用 fix_out 的标签通道屏蔽仍旧损坏的帧。
    """
    device = window_attn.device
    fix_labels = fix_out[-1]

    if history_len > 0:
        history_offset = history_frames - history_len
        history_slice = fix_out[:, history_start : history_start + history_len].unsqueeze(0)
        window_input[:, :, history_offset:history_frames] = history_slice
        history_valid = (~fix_labels[history_start : history_start + history_len].to(torch.bool)).to(device)
        window_attn[:, history_offset:history_frames] = history_valid.unsqueeze(0)

    future_offset = history_frames + window_target_len
    if future_len > 0:
        future_slice = fix_out[:, future_start : future_start + future_len].unsqueeze(0)
        window_input[:, :, future_offset : future_offset + future_len] = future_slice
        future_valid = (~fix_labels[future_start : future_start + future_len].to(torch.bool)).to(device)
        window_attn[:, future_offset : future_offset + future_len] = future_valid.unsqueeze(0)


@torch.no_grad()
def fix_sequence_with_labels(
    *,
    model,
    diffusion,
    args,
    motion_normalizer,
    input_sequence,
    length,
    device,
    label_buffer,
    cond_fn,
    direction="ltr",
):
    """
    基于 label_buffer 的坏帧，按滑窗策略逐段运行 fix_motion。
    """
    seq_len = int(length.item()) # 将长度 tensor 提取成 python int，方便后续切片
    nfeats = input_sequence.shape[0] # 每帧通道数（包含标签）
    if seq_len == 0:
        # 空序列没有帧需要修复，立即返回一个空的修复输出
        return {"fixed_feats": input_sequence[:, :0].clone()}

    # region 初始化
    seq_data = input_sequence[:, :seq_len] # 获取输入数据
    label_mask = label_buffer[:seq_len].to(device) # 获取标签
    target_window = max(1, args.fix_target_window) # 确保 target_window 至少为 1
    history_frames = args.fix_history_frames # 历史帧数
    future_frames = history_frames # 未来帧数
    intervals = build_corrupt_intervals_zero(label_mask.cpu().numpy()) # 构建坏帧区间列表
    if direction == "rtl":
        intervals = list(reversed(intervals)) # 反向遍历坏帧区间
    if not intervals:
        return {"fixed_feats": seq_data.clone()} # 没有坏帧需要修复，直接返回原始序列
    
    # 初始化修复输出，每次修复都会更新
    final_fix = seq_data.clone() # 默认先用原始序列填充
    fix_out = seq_data.clone()
    fix_out[-1, :seq_len] = label_mask.to(fix_out.dtype)
    #endregion

    #region 如果run_section使用model前向，则可能导致修复上下文，因此需要累加结果，否则直接使用原始序列
    use_aggregation = not args.use_precomputed_cleanup
    if use_aggregation:
        result_accum = torch.zeros((nfeats, seq_len), device=device)
        result_counts = torch.zeros((seq_len,), device=device)
    #endregion

    # 修复逻辑是：先获取坏帧区间，根据坏帧区间获取目标窗口，
    # 根据目标窗口获取历史窗口和未来窗口，和目标窗口，
    # 然后根据输入窗口获取输出窗口获取修复结果
    for interval_start, interval_end in intervals: # 遍历每个坏帧区间
        interval_len = interval_end - interval_start
        if interval_len <= 0:
            continue

        #region 计算目标窗口，如果坏帧区间比 target_window 短，窗口就是 target_window；如果坏帧区间更长，就把窗口拉长到相同长度且
        window_target_len = max(target_window, interval_len) # 确保目标窗口长度至少为 target_window
        center = (interval_start + interval_end) / 2 # 计算目标窗口中心点，对齐中心
        raw_start = int(math.floor(center - window_target_len / 2 + 0.5)) # 计算目标窗口起始点
        target_start = max(0, min(raw_start, seq_len - window_target_len)) # 确保目标窗口起始点不越界
        target_end = target_start + window_target_len # 确保目标窗口结束点不越界
        if target_start > interval_start:
            target_start = interval_start # 确保目标窗口起始点不越界
            target_end = target_start + window_target_len # 确保目标窗口结束点不越界
        if target_end < interval_end:
            target_end = interval_end # 确保目标窗口结束点不越界    
            target_start = max(0, target_end - window_target_len) # 确保目标窗口起始点不越界
            target_end = target_start + window_target_len # 确保目标窗口结束点不越界
        window_target_len = target_end - target_start # 确保目标窗口长度不小于0
        if window_target_len <= 0:
            continue
        #endregion

        #region 计算历史窗口和未来窗口
        history_start = max(0, target_start - history_frames) # 确保历史窗口起始点不越界
        history_len = target_start - history_start # 确保历史窗口长度不小于0
        future_start = target_end # 确保未来窗口起始点不越界
        future_end = min(seq_len, future_start + future_frames) # 确保未来窗口结束点不越界
        future_len = future_end - future_start # 确保未来窗口长度不小于0
        #endregion

        #region 初始化输入窗口和注意力掩码
        window_frames = history_frames + window_target_len + future_frames # 确保窗口总长度为上文+目标+下文
        window_input = torch.zeros((1, nfeats, window_frames), device=device) # 初始化输入窗口
        window_attn = torch.zeros((1, window_frames), dtype=torch.bool, device=device) # 初始化注意力掩码
        #endregion

        #region 填充历史窗口和未来窗口
        gather_context(
            fix_out=fix_out,
            window_input=window_input,
            window_attn=window_attn,
            history_frames=history_frames,
            history_start=history_start,
            history_len=history_len,
            future_start=future_start,
            future_len=future_len,
            window_target_len=window_target_len,
        )
        #endregion

        #region 填充目标窗口
        target_slice_start = history_frames
        target_slice_end = target_slice_start + window_target_len
        window_input[:, :, target_slice_start:target_slice_end] = fix_out[
            :, target_start:target_end
        ].unsqueeze(0)
        window_attn[:, target_slice_start:target_slice_end] = True
        #endregion

        #region 初始化标签掩码
        label_mask = torch.zeros((1, window_frames), dtype=torch.bool, device=device)
        label_slice_start = history_frames
        label_slice_end = history_frames + window_target_len
        label_mask[:, label_slice_start:label_slice_end] = fix_out[
            -1, target_start:target_end
        ].to(torch.bool).unsqueeze(0)
        #endregion

        #region 初始化长度张量
        length_tensor = torch.tensor(
            [history_len + window_target_len + future_len], device=device, dtype=length.dtype
        )
        #endregion

        #region 执行检测
        det_out = detect_labels(
            model=model,
            diffusion=diffusion,
            args=args,
            input_motions=window_input,
            length=length_tensor,
            attention_mask=window_attn,
            motion_normalizer=motion_normalizer,
        )
        #endregion

        #region 执行修复
        fix_kwargs = dict(
            model=model,
            diffusion=diffusion,
            args=args,
            input_motions=window_input,
            length=length_tensor,
            attention_mask=window_attn,
            motion_normalizer=motion_normalizer,
            label=label_mask,
            re_sample_det_feats=det_out["re_sample_det_feats"],
            cond_fn=cond_fn,
        )
        if args.use_precomputed_cleanup:
            fix_kwargs.update(
                label_for_cleanup=label_mask,
                re_sample_det_feats_for_cleanup=det_out["re_sample_det_feats"],
            )

        fix_result = fix_motion(**fix_kwargs)
        fix_feats = fix_result["sample_fix_feats"][0]
        #endregion

        #region 更新修复输出，这里只更新了当前正在修复的损坏区间的部分
        interval_slice = slice(label_slice_start, label_slice_end)
        fixed_target_segment = fix_feats[:, interval_slice]
        target_relative_start = interval_start - target_start
        target_relative_end = target_relative_start + interval_len
        fixed_segment = fixed_target_segment[
            :, target_relative_start:target_relative_end
        ]
        # 如果允许 连着上下文一起修复，则在累加器里累加当前段的修复结果
        if use_aggregation:
            result_accum[:, interval_start:interval_end] += fixed_segment
            result_counts[interval_start:interval_end] += 1
            counts_slice = result_counts[interval_start:interval_end].clamp(min=1.0)
            averaged = result_accum[:, interval_start:interval_end] / counts_slice.unsqueeze(0)
            # 把平均结果写回修复缓冲，用于后续窗口的上下文
            fix_out[:, interval_start:interval_end] = averaged
        else:
            # 不聚合时直接把结果写入最终输出，也同步更新 fix_out 以便 context 复用
            final_fix[:, interval_start:interval_end] = fixed_segment
            fix_out[:, interval_start:interval_end] = fixed_segment

        # 修复后目标段不再被视作坏帧（在窗口范围内）
        local_interval_start = history_frames + max(0, interval_start - target_start)
        local_interval_end = local_interval_start + interval_len
        label_mask[0, local_interval_start:local_interval_end] = False
        # 把标签通道同步写回 fix_out，保持通道与实际好帧状态一致
        fix_out[-1, interval_start:interval_end] = label_mask[
            0, local_interval_start:local_interval_end
        ].to(fix_out.dtype)
        #endregion

    #region 处理累积结果
    if use_aggregation:
        missing_mask = result_counts == 0
        if missing_mask.any():
            result_accum[:, missing_mask] = seq_data[:, missing_mask]
            result_counts[missing_mask] = 1
        final_fix = result_accum / result_counts.clamp(min=1.0).unsqueeze(0)
    #endregion

    return {"fixed_feats": final_fix}


@torch.no_grad()
def detect_and_fix_sequence(
    *,
    model,
    diffusion,
    args,
    motion_normalizer,
    input_sequence,
    length,
    device,
    cond_fn,
    direction="ltr",
    sequence_index=None,
):
    """
    先跑一次 detect_sequence_quality_labels 再调用 fix_sequence_with_labels。
    """
    if(args.use_window_detect):
        label_buffer = detect_sequence_quality_labels_window(
            model=model,
            diffusion=diffusion,
            args=args,
            motion_normalizer=motion_normalizer,
            input_sequence=input_sequence,
            length=length,
            device=device,
            cond_fn=cond_fn,
        )
    else:
        label_buffer = detect_sequence_quality_labels_segment(
            model=model,
            diffusion=diffusion,
            args=args,
            motion_normalizer=motion_normalizer,
            input_sequence=input_sequence,
            length=length,
            device=device,
            cond_fn=cond_fn,
        )

    if args.debug_print_intervals:
        flat_labels = label_buffer.cpu().numpy().astype(bool)
        intervals = format_intervals(build_corrupt_intervals(flat_labels))
        if sequence_index is None:
            prefix = "当前"
        else:
            prefix = f"序列 {sequence_index}"
        print(f"{prefix} 的坏帧区间: {intervals}")

    fix_out = fix_sequence_with_labels(
        model=model,
        diffusion=diffusion,
        args=args,
        motion_normalizer=motion_normalizer,
        input_sequence=input_sequence,
        length=length,
        device=device,
        label_buffer=label_buffer,
        cond_fn=cond_fn,
        direction=direction,
    )
    return fix_out["fixed_feats"], label_buffer


def main():
    #region 解析参数并固定随机
    args = smart_args()
    fixseed(args.seed)
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    out_path = build_output_dir(args)
    #endregion

    #region 加载数据集
    print(f"加载 dataset from {args.testdata_dir}...")
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
        folder_path=args.folder_path or None,
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer
    #endregion

    #region 创建模型和扩散调度器
    print("创建 model and diffusion...")
    print("使用采样器: ", args.ts_respace)
    _model, diffusion = create_model_and_diffusion(args)
    model = EMA(_model, include_online_model=False) if args.use_ema else _model
    #endregion

    #region 加载模型权重
    print(f"加载 checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")
    model.load_state_dict(state_dict, strict=False)
    model.to(device)
    model.eval()
    #endregion

    #region 准备条件函数，用于分类器引导
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)
    if cond_fn is not None:
        cond_fn.keywords["model"] = model
    #endregion

    #region 缓冲区：存储处理过程中的各种数据     
    all_lengths = []            # 存储每个样本的长度列表
    all_input_motions_vec = []  # 存储修复前的原始特征
    labels = []                 # 存储最终的坏帧标签
    gt_labels_buf = []          # 存储 GT 帧级标签
    all_motions_fix = []        # 存储修复后的 SMPL 字典
    all_fix_motions_vec = []    # 存储修复后的特征
    total_samples = 0           # 总样本计数器
    #endregion

    # 遍历每个 batch，按序逐帧滑窗处理
    for i, input_batch in enumerate(data):
        #region 将 batch 中的张量搬到目标设备
        input_batch = {
            k: (v.to(device) if torch.is_tensor(v) else v)
            for k, v in input_batch.items()
        }
        input_motions = input_batch["x"]
        length = input_batch["length"]
        #endregion

        #region 将输入反归一化到原尺度以便读取 GT
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())
        gt_labels = (temp_sample[..., -1] > 0.5).numpy()
        gt_labels_buf.extend(gt_labels)
        #endregion

        # 每个 batch 内逐个样本，按其真实长度处理
        for b in range(input_motions.shape[0]):
            seq_len = int(length[b].item())
            if seq_len == 0:
                continue

            seq_feats = input_motions[b, :, :seq_len]
            seq_length = torch.tensor([seq_len], device=device, dtype=length.dtype)

            #region 正向修复
            fixed_feats, forward_label_buffer = detect_and_fix_sequence(
                model=model,
                diffusion=diffusion,
                args=args,
                motion_normalizer=motion_normalizer,
                input_sequence=seq_feats,
                length=seq_length,
                device=device,
                cond_fn=cond_fn,
                direction="ltr",
                sequence_index=total_samples + 1,
            )
            #endregion

            combined_label_buffer = forward_label_buffer.clone()

            #region 如果开启反向修复，则再跑一次反向修复
            if args.reverse_pass_after_forward:
                print(f"执行反向修复")
                fixed_feats, reverse_label_buffer = detect_and_fix_sequence(
                    model=model,
                    diffusion=diffusion,
                    args=args,
                    motion_normalizer=motion_normalizer,
                    input_sequence=fixed_feats,
                    length=seq_length,
                    device=device,
                    cond_fn=cond_fn,
                    direction="rtl",
                    sequence_index=total_samples + 1,
                )
                combined_label_buffer |= reverse_label_buffer
            #endregion

            #region 最终数据记录（反向修复后替代正向结果）
            labels.append(combined_label_buffer.cpu().numpy())
            all_lengths.append(seq_len)
            all_input_motions_vec.append(seq_feats.transpose(0, 1).cpu().numpy())

            fixed_motion_denorm = motion_normalizer.inverse(
                fixed_feats.transpose(0, 1).cpu()
            )
            all_motions_fix.append(
                globsmplrifkefeats_to_smpldata(fixed_motion_denorm[..., :-1])
            )
            all_fix_motions_vec.append(fixed_feats.transpose(0, 1).cpu().numpy())
            #endregion

            total_samples += 1
            print(f"已处理 {total_samples} 条序列（当前长度 {seq_len} 帧）\n")

            if args.num_samples and total_samples >= args.num_samples:
                break
        if args.num_samples and total_samples >= args.num_samples:
            break

    #region 保存结果（动作、修复、标签等）
    os.makedirs(out_path, exist_ok=True)
    result_path = os.path.join(out_path, "results_long_smart.npy")
    print(f"saving results to [{result_path}]")
    np.save(
        result_path,
        {
            "motion_fix": all_motions_fix,
            "label": labels,
            "gt_labels": gt_labels_buf,
            "lengths": np.array(all_lengths, dtype=np.int32),
            "all_fix_motions_vec": all_fix_motions_vec,
            "all_input_motions_vec": all_input_motions_vec,
        },
    )
    with open(result_path.replace(".npy", "_len.txt"), "w", encoding="utf-8") as fw:
        fw.write("\n".join(str(l) for l in all_lengths))
    #endregion


if __name__ == "__main__":
    main()


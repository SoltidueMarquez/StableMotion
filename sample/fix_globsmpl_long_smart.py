"""
Smart detection pipeline for sliding-window-long globsmpl motions.

与 `sample/fix_globsmpl_long.py` 保持类似的滑窗思路：使用“history + target + future”的窗口结构，
但每次只真正写入 target 部分的标签，history/future 只是提供上下文，不会被标注或覆盖。
由于每次滑窗的步长正好等于 target 的长度，所有帧只会被写入一次，因此无需在多个窗口间取平均。
"""

import os
from argparse import ArgumentParser

import numpy as np
import torch

from ema_pytorch import EMA

from sample.fix_globsmpl import detect_labels
from sample.utils import build_output_dir, prepare_cond_fn
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
        help="每个滑窗的 target 部分帧数，同时决定 stride (等于 target_window)。",
    )
    parser.add_argument(
        "--context_frames",
        type=int,
        default=20,
        help="history 和 future 的帧数（对称），只作为上下文不会直接被标注。",
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
    return parse_and_load_from_model(parser)


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

def format_intervals(intervals):
    if not intervals:
        return "[]"
    return "[" + ",".join(f"[{s},{e}]" for s, e in intervals) + "]"


@torch.no_grad()
def detect_sequence_quality_labels(
    *,
    model,
    diffusion,
    args,
    motion_normalizer,
    input_sequence,
    length,
    device,
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
    history_frames = args.context_frames  # history/future 各自的帧数
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
        future_end = min(seq_len, target_end + future_frames)  # future 结束位置（不超过序列长度）
        future_len = future_end - target_end    # future 帧数（可能小于 future_frames）

        window_frames = history_frames + target_size + future_frames  # 窗口总帧数（history + target + future）
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
        label[:history_frames] = False
        label[history_frames + target_len :] = False

        # 设置label_buffer只更新target部分的输出，无视history和future
        label_buffer[start:target_end] = label[target_offset : target_offset + target_len]
        #endregion

    return label_buffer


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

            #region 执行检测 直接将（history+target+future）作为输入，但是只取target的输出
            label_buffer = detect_sequence_quality_labels(
                model=model,
                diffusion=diffusion,
                args=args,
                motion_normalizer=motion_normalizer,
                input_sequence=seq_feats,
                length=seq_length,
                device=device,
            )
            #endregion

            labels.append(label_buffer.cpu().numpy())
            all_lengths.append(seq_len)
            all_input_motions_vec.append(seq_feats.transpose(0, 1).cpu().numpy())
            total_samples += 1
            print(f"已处理 {total_samples} 条序列（当前长度 {seq_len} 帧）")

            #region 打印坏帧区间
            if args.debug_print_intervals:
                flat_labels = label_buffer.cpu().numpy().astype(bool)
                intervals = format_intervals(build_corrupt_intervals(flat_labels))
                print(f"序列 {total_samples} 的坏帧区间: {intervals}")
            #endregion

            # TODO: 在得到整条序列的 label 之后再设计用这些标签去进行修复的逻辑。

            if args.num_samples and total_samples >= args.num_samples:
                break
        if args.num_samples and total_samples >= args.num_samples:
            break

    #region 保存结果
    # os.makedirs(out_path, exist_ok=True)
    # result_path = os.path.join(out_path, "results_long_smart.npy")
    # print(f"saving detection results to [{result_path}]")
    # np.save(
    #     result_path,
    #     {
    #         "label": labels,
    #         "gt_labels": gt_labels_buf,
    #         "lengths": np.array(all_lengths, dtype=np.int32),
    #         "all_input_motions_vec": all_input_motions_vec,
    #     },
    # )
    # with open(result_path.replace(".npy", "_len.txt"), "w") as fw:
    #     fw.write("\n".join(str(l) for l in all_lengths))
    #endregion


if __name__ == "__main__":
    main()


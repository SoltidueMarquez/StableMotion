"""
Sliding-window repair pipeline for longer AMASS sequences.

基于 `sample/fix_globsmpl.py` 的 detect/fix 逻辑，将每条长序列
切成若干 100 帧窗口、重叠拼接，利用前面已经修复好的帧作为上下文，
避免模型在过长的序列中边界抖动。
"""

import os
from argparse import ArgumentParser

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


def long_args():
    """
    解析用于滑窗修复的命令行参数。
    """
    parser = ArgumentParser(description="Fix long globsmpl motions with sliding windows.")
    add_base_options(parser)  # 加入基础选项（batch_size、seed 等惯用参数）
    add_sampling_options(parser)  # 加入采样/模型路径等采样专用参数
    parser.add_argument(
        "--window_size",
        type=int,
        default=100,
        help="Sliding window size (frames) passed to the diffusion model.",
    )  # 每次送入模型的帧数上限（不含 context）
    parser.add_argument(
        "--window_stride",
        type=int,
        default=80,
        help="Stride between windows; controls overlap.",
    )  # 窗口的步长，用于控制重叠长度（window_size - window_stride）
    parser.add_argument(
        "--context_frames",
        type=int,
        default=20,
        help="Number of previously repaired frames that are passed as context for continuity.",
    )  # 每个窗口前端保留的已修复帧数，提供边界上下文
    parser.add_argument(
        "--future_frames",
        type=int,
        default=0,
        help="可选：每个窗口后向延伸多少帧作为已知上下文（用于边界平滑）。",
    ) # 每个窗口后向延伸多少帧作为已知上下文（用于边界平滑）。
    parser.add_argument(
        "--folder_path",
        type=str,
        default="",
        help="可选：直接从某个文件夹遍历 motion，避免用 split 文件。",
    ) # 可选：直接从某个文件夹遍历 motion，避免用 split 文件。
    return parse_and_load_from_model(parser)


def compute_window_starts(seq_len: int, window_size: int, stride: int):
    """
    生成滑窗起点，确保每帧都能被某个窗口覆盖，同时控制 overlap 大小。
    Args:
        seq_len: 序列长度
        window_size: 每次送入模型的帧数上限（不含 context）
        stride: 窗口的步长，用于控制重叠长度（window_size - window_stride）
    Returns:
        list: 滑窗起点列表
    """
    # 序列长度 non-positive 时直接返回空列表
    if seq_len <= 0:
        return []
    # stride 必须正数，否则滑窗无法朝前推进
    if stride <= 0:
        raise ValueError("滑窗之间的步长必须正数")
    # 若序列比窗口短，则只需要一个起点 0
    if seq_len <= window_size:
        return [0]

    # 从 0 开始每次以 stride 前进，直到窗口末尾能覆盖到序列末尾
    starts = list(range(0, seq_len - window_size + 1, stride))
    # 最后一个窗口若不能覆盖尾部，则再补一个以末尾对齐的起点
    if starts[-1] + window_size < seq_len:
        starts.append(seq_len - window_size)
    # 排序并去重，防止 stride 恰好覆盖导致重复
    return sorted(set(starts))


def gather_context(result_accum, counts, start, context_frames):
    """
    根据已经累积好的结果，提取前 context_frames 帧作为上下文，防止窗之间突变。
    """
    device = result_accum.device
    # 先创建储存上下文特征的张量，shape=(特征维, 帧数)，默认全部 0
    context_tensor = torch.zeros((result_accum.shape[0], context_frames), device=device)
    # attention mask 表示哪些上下文帧有效（已修复过），默认都无效
    context_mask = torch.zeros(context_frames, dtype=torch.bool, device=device)
    # 如果没有需要的上下文帧或当前窗口从序列起点开始，直接当作空上下文返回
    if context_frames == 0 or start == 0:
        return context_tensor, context_mask, 0

    # 计算要从哪一帧开始取上下文，确保不越界
    context_start = max(0, start - context_frames)
    context_len = start - context_start
    # 遍历每一帧；context_len 可能小于 context_frames，此时会靠右对齐
    for idx in range(context_len):
        global_idx = context_start + idx
        cnt = counts[global_idx]
        if cnt <= 0:
            # 当前帧未被任何窗口覆盖，不能当作有效上下文
            continue
        # 目标位置从右侧对齐：context_frames-context_len 先留空
        dest_idx = context_frames - context_len + idx
        # result_accum 中保存的是累加值，除以 cnt 得到平均结果
        context_tensor[:, dest_idx] = result_accum[:, global_idx] / cnt
        # 标记该位置为有效上下文帧，attention mask 置 True
        context_mask[dest_idx] = True
    return context_tensor, context_mask, context_len


@torch.no_grad()
def process_long_sequence(
    *,
    model,  # 已加载的扩散模型，用于 detection/fix 分支
    diffusion,  # 对应的 diffusion 采样器，控制生成过程
    args,  # 命令行参数，包含滑窗、上下文等配置
    motion_normalizer,  # 归一化器，保持输入/输出一致尺度
    input_sequence,  # 原始 motion 特征（可能含 padding），按帧排列
    length,  # 实际有效帧数张量，指示序列有效长度
    device,  # 所有张量要放到的设备（CPU/GPU）
    cond_fn,  # 可选的 conditioner，用于条件采样控制
):
    """
    使用滑动窗口在长序列上逐段做 DET + FIX，前后窗口共享 context_frames。
    """
    #region 取出当前序列的真实帧长，之后不会处理 padding 部分
    seq_len = int(length.item())  # 有效帧数，Tensor -> int 方便后续切片
    nfeats = input_sequence.shape[0]  # 每帧特征维度（channel 数）
    if seq_len == 0:
        # 序列为空时直接返回一个空的结果字典，避免后续计算
        return {
            "fixed_feats": torch.zeros((nfeats, 0), device=device),  # 没有帧，宽度为0
            "det_feats": torch.zeros((nfeats, 0), device=device),
            "labels": torch.zeros((0,), dtype=torch.bool, device=device),
        }

    window_size = args.window_size  # 每次送入模型的目标帧数量（不含上下文）
    stride = args.window_stride  # 滑窗移动的步长，用于控制重叠
    context_frames = args.context_frames  # 每个窗口额外携带的历史帧
    future_frames = args.future_frames  # 每个窗口额外携带的未来帧作为条件
    base_window = window_size + context_frames  # 不包括 future 的基础窗口长度
    #endregion

    #region 结果缓冲：累加每帧的修复、检测特征并记录覆盖次数
    # 把每帧从各个窗口修复出的结果加起来，再记录每帧被多少个窗口覆盖，后面会用它们做除法得到融合后的输出（防止重叠区域抖动）。
    result_accum = torch.zeros((nfeats, seq_len), device=device)  # 每帧修复输出累加
    result_counts = torch.zeros((seq_len,), device=device)  # 记录每帧被多少窗口覆盖

    # 同样逻辑但用于检测分支返回的特征，方便后续把多个窗口的检测特征融合成一条连续流。
    det_accum = torch.zeros((nfeats, seq_len), device=device)  # 检测分支特征累加
    det_counts = torch.zeros((seq_len,), device=device)  # 检测特征覆盖次数
    label_buffer = torch.zeros((seq_len,), dtype=torch.bool, device=device)  # 最终标签（有坏帧）

    seq_data = input_sequence[:, :seq_len]  # 只保留有效帧，抛弃 padding
    window_starts = compute_window_starts(seq_len, window_size, stride)
    if len(window_starts) == 0:
        window_starts = [0]
    #endregion

    # 遍历每个滑窗，分段送入检测/修复
    for start in window_starts:
        target_end = min(start + window_size, seq_len)
        target_len = target_end - start
        if target_len <= 0:
            continue

        #region 取出上一窗口中已经修复并平均后的 context_frames 帧，作为本轮输入的前缀
        context_data, context_mask, context_len = gather_context(
            result_accum, result_counts, start, context_frames
        )

        future_len = min(future_frames, seq_len - target_end)
        window_frames = base_window + future_len  # 输入模型帧数：context + target + future
        window_input = torch.zeros((1, nfeats, window_frames), device=device)  # 构造窗口输入张量
        window_attn = torch.zeros((1, window_frames), dtype=torch.bool, device=device)  # attention mask 初始全 False
        length_tensor = torch.tensor([context_len + target_len + future_len], device=device, dtype=length.dtype)
        #endregion

        #region 把 context 部分沿时间轴填入前段，attention 将 context 视为有效帧
        if context_frames > 0:
            window_input[:, :, :context_frames] = context_data.unsqueeze(0)  # 把前一窗口输出作为本窗口输入前缀
            window_attn[:, :context_frames] = context_mask.unsqueeze(0)  # 只能让已有的上下文帧参与 attention
        #endregion
        
        #region 将实际需要检测/修复的当前 window 填入 context 之后的区域
        target_start = context_frames
        target_end_in_window = context_frames + target_len
        window_input[:, :, target_start:target_end_in_window] = seq_data[:, start:target_end].unsqueeze(0)
        window_attn[:, target_start:target_end_in_window] = True
        # endregion

        #region 将未来帧填入 context 之后的区域
        if future_len > 0:
            future_start = target_end_in_window
            future_end = future_start + future_len
            window_input[:, :, future_start:future_end] = seq_data[:, target_end: target_end + future_len].unsqueeze(0)
            window_attn[:, future_start:future_end] = True
        #endregion

        #region ===============================检测=======================================
        # 用检测分支预测当前窗口中的坏帧标签（标签通道），不只取当前窗口做检测是需要用来对未来帧进行attention遮盖处理才能输入修复模型
        # 输入input_motions是 上文的序列帧数 + 当前窗口的帧 + 未来帧
        # 输入attention_mask是 上文的序列帧数 + 当前窗口的帧 + 未来帧
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

        #region 只提取当前目标帧的标签部分，context与future 位置无需label，用attention遮掉损坏的未来帧
        label = det_out["label"].to(device)     # 取出检测分支预测的 label 张量
        label[:, :context_frames] = False       # 上下文部分不应该被标记
        if target_len < window_size:
            label[:, context_frames + target_len:] = False  # 窗口未满时忽略尾部填充
        # 这两行是用于后续可视化的
        target_label = label[0, context_frames:context_frames + target_len]  # 这里直接截取了 label 中从 context_frames 开始、长度为当前窗口目标帧数的切片只保留本窗口目标帧
        label_buffer[start:target_end] |= target_label  # 记录坏帧标签，多个窗口取或保持坏帧
        
        # 处理未来帧的标签，确保损坏的未来帧不被作为上下文污染
        if future_len > 0:
            future_bad = label[0, future_start:future_end]
            if future_bad.any():
                keep_future = ~future_bad.unsqueeze(0)
                # 把检测到的坏未来帧从 attention 中屏蔽，避免作为上下文污染当前窗口
                window_attn[:, future_start:future_end] &= keep_future
                label[:, future_start:future_end] &= keep_future
        #endregion

        #region ===============================修复=======================================
        #运行修复分支，使用检测标签指导生成替换帧
        # 这边的实际输入是 40上文帧+100当前窗口+40下文帧，
        # 但是因为上文和下文都经过了处理，要么不是损坏要么被attentionmask遮住，所以只会修复当前窗口中损坏的帧
        # 这边的输出是包含了没有变化的上文和下文的，最终写回累加的时候只有当前窗口的目标帧会被累加/平均
        fix_out = fix_motion(
            model=model,                    # 传入同一个 diffusion 模型
            diffusion=diffusion,            # 采样器
            args=args,                      # 配置参数
            input_motions=window_input,     # 包含 context + 当前目标帧 + future
            length=length_tensor,           # 有效帧数（context + target + future） [B]，每条序列的有效帧长度
            attention_mask=window_attn,     # [B, N] bool，标记有效帧区域
            motion_normalizer=motion_normalizer,  # 用于特征归一化/反归一化的工具
            label=label,                     # [B, N] bool，检测阶段得到的坏帧标记
            re_sample_det_feats=det_out["re_sample_det_feats"],  # [B, C, N] 检测阶段的采样特征，用于软修复调度
            cond_fn=cond_fn,                 # 可选 classifier guidance 函数
            
            # # 这边强制走构造好的label
            label_for_cleanup=label,
            re_sample_det_feats_for_cleanup=det_out["re_sample_det_feats"]
        )
        #endregion

        #region 累加当前窗口 result，后续通过 counts 计算平均（避免 overlapping 处偏移）
        sample_fix_feats = fix_out["sample_fix_feats"][0]  # 修复分支每帧输出（含 context和future）
        det_feats = det_out["re_sample_det_feats"][0]  # 检测分支的辅助特征，后面也要融合
        target_slice = slice(context_frames, context_frames + target_len)  # 提取当前目标帧（从context_frames开始，长度为target_len）
        result_accum[:, start:target_end] += sample_fix_feats[:, target_slice]  # 累加目标帧特征
        result_counts[start:target_end] += 1  # 记录每帧被累积的次数，用于后续平均
        det_accum[:, start:target_end] += det_feats[:, target_slice]  # 检测特征也同样累加
        det_counts[start:target_end] += 1  # 同步统计检测特征的覆盖次数
        #endregion

    #region 将重叠帧的累积值除以覆盖次数，得到融合后的结果
    # 这边是因为设置了window_stride，
    # 也就是说有可能第一个窗口是0-100，第二个窗口是60-160，中间重叠的40需要取平均
    safe_denom_fix = result_counts.clamp(min=1.0)  # 避免除零，保证每帧至少除以1
    safe_denom_det = det_counts.clamp(min=1.0)  # 同理，检测特征的计数
    final_fix = result_accum / safe_denom_fix.unsqueeze(0)  # 每帧修复结果除以覆盖次数，融合重叠窗口
    final_det = det_accum / safe_denom_det.unsqueeze(0)  # 同样处理检测特征
    #endregion

    return {
        "fixed_feats": final_fix,               # [C, N] 修复后的特征，融合重叠窗口
        "det_feats": final_det,                 # [C, N] 检测特征，融合重叠窗口
        "labels": label_buffer.cpu().numpy(),   # 最终的坏帧标签，多个窗口取或保持坏帧
    }


def main():
    #region 解析参数并固定随机
    args = long_args()
    fixseed(args.seed)
    #endregion

    #region 设备与输出目录
    dist_util.setup_dist(args.device)
    device = dist_util.dev()
    out_path = build_output_dir(args)
    #endregion

    #region 准备数据加载器（不打乱，按顺序处理 test_benchmark）
    print(f"加载 dataset from {args.testdata_dir}...")
    data = get_dataset_loader(
        name="globsmpl",
        batch_size=args.batch_size,
        split="test_benchmark",
        data_dir=args.testdata_dir,
        normalizer_dir=args.normalizer_dir,
        shuffle=False,
        sample_mode="sequential",  # 修改点：让 loader 返回按顺序的整段序列（不再随机裁剪）
        sequence_window_size=args.window_size,
        sequence_stride=args.window_stride,
        folder_path=args.folder_path or None,  # 新增：可直接遍历folder
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer  # 获取特征归一化/反归一化工具
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
    model.load_state_dict(state_dict, strict=False)  # 加载模型权重，忽略不匹配的键
    model.to(device)
    model.eval()  # 设置为评估模式，禁用 dropout 等训练相关行为
    #endregion

    #region 准备条件函数，用于分类器引导
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)  # 准备条件函数，用于分类器引导
    if cond_fn is not None:
        cond_fn.keywords["model"] = model  # 将模型传递给条件函数，用于分类器引导
    #endregion

    #region 缓冲区：存储处理过程中的各种数据
    all_motions = []            # 存储检测阶段解码的 SMPL 字典
    all_lengths = []            # 存储每个样本的长度列表
    all_input_motions_vec = []  # 存储修复前的原始特征
    gt_labels_buf = []          # 存储 GT 帧级标签
    all_motions_fix = []        # 存储修复后的 SMPL 字典
    all_fix_motions_vec = []    # 存储修复后的特征
    labels = []                 # 存储最终的坏帧标签
    #endregion

    total_samples = 0 # 总样本计数器
        
    # 遍历每个 batch，按序逐帧滑窗处理
    for i, input_batch in enumerate(data):
        #region 这段是在每个 batch 里把原始数据搬上 device、恢复成 SMPL 表示，并决定是否只收集 GT
        input_batch = {  # 将 batch 中的张量搬到目标设备
            k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_batch.items()
        }
        input_motions = input_batch["x"]  # [B, C, N] 输入特征序列
        attention_mask = input_batch["mask"].bool()  # 有效帧掩码，去掉多余维度并转为 bool
        length = input_batch["length"]  # 每个样本的有效长度

        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())
        if args.collect_dataset:  # 若只想收集 GT 而不做检测/修复
            for _sample in temp_sample:
                all_motions.append(  # 提取动作部分（去掉标签通道）并保存
                    globsmplrifkefeats_to_smpldata(_sample[..., :-1])
                )
            all_lengths.append(length.cpu().numpy())  # 记录长度
            continue

        gt_labels = (temp_sample[..., -1] > 0.5).numpy()  # 取标签通道作 GT 布尔
        gt_labels_buf.extend(gt_labels)  # 存储 GT 标签
        #endregion

        #region 每个 batch 内逐个样本，按其真实长度处理
        for b in range(input_motions.shape[0]):
            seq_len = int(length[b].item())
            if seq_len == 0: continue # 空序列直接跳过

            seq_feats = input_motions[b, :, :seq_len]   # 取出当前样本的特征序列
            seq_length = torch.tensor([seq_len], device=device, dtype=length.dtype) # 当前样本的有效长度

            window_out = process_long_sequence( # 处理当前样本，得到修复后的特征、检测特征、标签
                model=model,                      # 已加载的扩散模型，用于 detection/fix 分支
                diffusion=diffusion,              # 对应的 diffusion 采样器，控制生成过程
                args=args,                       # 命令行参数，包含滑窗、上下文等配置
                motion_normalizer=motion_normalizer, # 归一化器，保持输入/输出一致尺度
                input_sequence=seq_feats,        # 原始 motion 特征（可能含 padding），按帧排列
                length=seq_length,              # 实际有效帧数张量，指示序列有效长度
                device=device,                  # 所有张量要放到的设备（CPU/GPU）
                cond_fn=cond_fn,                # 可选的 conditioner，用于条件采样控制
            )

            det_motion_denorm = motion_normalizer.inverse( # 将检测特征反归一化到原尺度
                window_out["det_feats"].transpose(0, 1).cpu() # [C, N] -> [N, C]
            )
            fix_motion_denorm = motion_normalizer.inverse( # 将修复特征反归一化到原尺度
                window_out["fixed_feats"].transpose(0, 1).cpu() # [C, N] -> [N, C]
            )

            all_motions.append(globsmplrifkefeats_to_smpldata(det_motion_denorm[..., :-1])) # 将检测特征转换为 SMPL 数据结构
            all_motions_fix.append(globsmplrifkefeats_to_smpldata(fix_motion_denorm[..., :-1])) # 将修复特征转换为 SMPL 数据结构
            labels.append(window_out["labels"]) # 存储当前样本的标签
            all_lengths.append(seq_len) # 记录当前样本的有效长度
            all_input_motions_vec.append(seq_feats.transpose(0, 1).cpu().numpy()) # 存储当前样本的原始特征
            all_fix_motions_vec.append(window_out["fixed_feats"].transpose(0, 1).cpu().numpy()) # 存储当前样本的修复特征
            total_samples += 1 # 更新总样本计数
            print(f"Processed {total_samples} samples (current length {seq_len})") # 打印当前处理进度
            if args.num_samples and total_samples >= args.num_samples: # 如果达到最大样本数，则跳出循环
                break
        if args.num_samples and total_samples >= args.num_samples: # 如果达到最大样本数，则跳出 batch 循环
            break
    #endregion

    #region 保存最终结果（动作、修复、标签等）并触现有评估脚本 
    all_lengths = np.array(all_lengths, dtype=np.int32) # 将长度列表转换为 numpy 数组
    os.makedirs(out_path, exist_ok=True)

    # 保存最终结果（动作、修复、标签等）并触发现有评估
    npy_path = os.path.join(out_path, "results_long.npy")
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,
            "motion_fix": all_motions_fix,
            "label": labels,
            "gt_labels": gt_labels_buf,
            "lengths": all_lengths,
            "all_fix_motions_vec": all_fix_motions_vec,
            "all_input_motions_vec": all_input_motions_vec,
        },
    )
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw: # 另存长度文本，便于快速查看
        fw.write("\n".join([str(l) for l in all_lengths]))

    os.system(f"python -m eval.eval_scripts  --data_path {npy_path} --force_redo") # 触发评估脚本
    #endregion


if __name__ == "__main__":
    main()


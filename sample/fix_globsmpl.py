"""
Detect-and-fix pipeline for AMASS motions using a diffusion model.

- Loads test data (aligned Global SMPL RIFKE feats + labels channel).
- Runs a detection pass to find bad frames.
- Builds an inpainting mask and fixes bad frames.
- Optionally runs an ensemble cleanup path.
- Saves .npy results and triggers evaluation/visualization script.
"""

import os
import numpy as np


import torch
import einops
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True

from utils.fixseed import fixseed
from utils.parser_util import det_args
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata

from ema_pytorch import EMA
from sample.utils import run_cleanup_selection, prepare_cond_fn, choose_sampler, build_output_dir



@torch.no_grad()
def detect_labels(
    *,
    model,
    diffusion,
    args,
    input_motions,           # [B, C, N]
    length,                  # Tensor[int] of shape [B]
    attention_mask,          # [B, N] bool
    motion_normalizer,
):
    """
    Detection pass:
      - inpaint only label channel
      - optional MC averaging (ProbDetNum)
      - return binary label mask, decoded detected motions (list of dicts), re_sample (features)
    """
    device = input_motions.device              # 记录设备（CPU/GPU）
    bs, nfeats, nframes = input_motions.shape  # 解包 batch、通道数、帧数

    # Prepare det-mode kwargs（检测模式的扩散输入/条件）——仅让模型去“预测”标签通道，其他通道保持原样
    inpaint_motion_detmode = input_motions.clone()                  # 复制输入，避免直接改动调用方的张量
    inpaint_motion_detmode[:, -1] = 1.0                             # 将最后一维标签通道全部写成 1.0，作为被“破坏”的占位；模型需重新估计这一通道

    inpainting_mask_detmode = torch.ones_like(input_motions).bool() # 掩码初始全 True，表示默认保持原值不重绘
    inpainting_mask_detmode[:, -1] = False                          # 唯独标签通道设为 False，提示扩散模型：这一通道需要被预测/重绘

    inpaint_cond_detmode = (~inpainting_mask_detmode) & attention_mask.unsqueeze(-2)  # 需要预测的位置：掩码为 False 且帧在有效范围内（attention_mask=True）

    model_kwargs_detmode = {
        "y": {
            "inpainting_mask": inpainting_mask_detmode,             # 传给模型，指示哪些位置保留、哪些位置需要预测
            "inpainted_motion": inpaint_motion_detmode,             # 对应位置的已知值/占位值；在掩码为 False 的位置将作为强制写回或对比的“GT”
        },
        "inpaint_cond": inpaint_cond_detmode,                       # 掩码（True=由模型预测，False=直接用 inpainted_motion）
        "length": length,                                           # 每个样本的有效帧长度，用于后续裁剪或掩码
        "attention_mask": attention_mask,                           # 帧级有效性掩码（形状 [B, N]），确保越界帧不被处理
    }

    # 通过 choose_sampler 选择 diffusion.p_sample_loop（当 ts_respace=False）或 diffusion.ddim_sample_loop（ts_respace=True）。
    sample_fn = choose_sampler(diffusion, args.ts_respace)

    # Single detection pass
    re_sample = sample_fn(
        model,
        (bs, nfeats, nframes),
        clip_denoised=False,
        model_kwargs=model_kwargs_detmode,
        skip_timesteps=0,
        init_image=None,
        progress=True,
        dump_steps=None,
        noise=None,
        const_noise=False,
    )

    # 若开启 ProbDetNum，则额外进行多次采样用于蒙特卡洛平均，降低单次检测波动
    if args.ProbDetNum:
        for _ in range(args.ProbDetNum):
            # 重新采样一次（与第一次独立），累加到 re_sample 中
            re_sample += sample_fn(
                model,                     # 当前检测模型
                (bs, nfeats, nframes),     # 采样的输出形状：批大小、特征维、帧数
                clip_denoised=False,       # 不在扩散末尾裁剪去噪结果
                model_kwargs=model_kwargs_detmode,  # 传入检测模式所需的条件
                skip_timesteps=0,          # 不跳过扩散步数，完整反演
                init_image=None,           # 不提供初始图像，从纯噪声开始
                progress=True,             # 显示进度条
                dump_steps=None,           # 不额外导出中间步
                noise=None,                # 不固定噪声，由采样器内部生成
                const_noise=False,         # 不使用常量噪声，每次重新采样
            )
        # 除以总采样次数（初始一次 + 额外 ProbDetNum 次），得到均值结果
        re_sample = re_sample / (args.ProbDetNum + 1)

    # Read label channel
    # re_sample: [B, C, N]（设备上），先转置为 [B, N, C] 并搬到 CPU，随后通过 normalizer 反归一化得到真实尺度的特征
    sample_det = motion_normalizer.inverse(re_sample.transpose(1, 2).cpu())
    # 最后一维的最后一个通道视作“检测置信度/标签通道”，与阈值 ProbDetTh 比较得到布尔标签 [B, N]
    label = sample_det[..., -1] > args.ProbDetTh
    # print("sample_det.shape:", sample_det.shape)[32,100,233]
    # 除去最后一个标签通道，保留身体运动特征部分 [B, N, C-1]
    sample_body = sample_det[..., :-1]

    # 将每个样本的全局/SMPL/RIFE特征转换为 SMPL 数据结构，便于下游使用
    recon_input_motion = [globsmplrifkefeats_to_smpldata(_s) for _s in sample_body]

    # Pack outputs
    out = {
        "label": label,                              # [B, N] bool (on CPU)
        "recon_input_motion": recon_input_motion,        # list[dict]
        "re_sample_det_feats": re_sample,            # [B, C, N] feats (device)
    }
    return out


@torch.no_grad()
def fix_motion(
    *,
    model,                   # 扩散模型主体（含 inpaint 能力）
    diffusion,               # 扩散调度器/步数配置
    args,                    # 运行配置，包含跳步、集成、软修复等开关
    input_motions,           # [B, C, N] 原始输入动作特征
    length,                  # Tensor[int] shape [B]，每条序列的有效帧长度
    attention_mask,          # [B, N] bool，标记有效帧区域
    motion_normalizer,       # 用于特征归一化/反归一化的工具
    label,                   # [B, N] bool，检测阶段得到的坏帧标记
    re_sample_det_feats,     # [B, C, N] 检测阶段的采样特征，用于软修复调度
    cond_fn=None,            # 可选 classifier guidance 函数
    label_for_cleanup=None,
    re_sample_det_feats_for_cleanup=None,
):
    """
    修复阶段（Fix pass）：
      - 对检测到的坏帧做一次膨胀
      - 构建修复所需的 inpainting 掩码
      - 可选：软修复步数调度（soft-inpaint schedule）
      - 运行扩散采样或集成清理流程
      - 返回采样特征 sample_fix 及解码后的修复动作
    """
    device = input_motions.device
    bs, nfeats, nframes = input_motions.shape

    # 确保 label 与后续计算在同一设备上，检测阶段得到的坏帧标记
    labeSsl = label.to(device)                   # 将标签搬到同一设备

    # 对检测结果做轻微膨胀：左右各扩 1 帧；序列最后一帧强制视为“好”帧
    temp_labels = label.clone()                # 备份原标签
    label[..., 1:] += temp_labels[..., :-1]    # 右移一帧并相加，实现左侧膨胀
    label[..., :-1] += temp_labels[..., 1:]    # 左移一帧并相加，实现右侧膨胀
    for mids, mlen in enumerate(length.cpu().numpy()):  # 遍历每个样本的有效长度
        label[mids, ..., mlen - 1] = 0         # 序列最后一帧强制标记为“好”帧

    # 构建修复掩码：True=保留，False=需要重绘（标签通道始终保留）
    det_good_frames_per_sample = {             # 记录每个样本中需要保留的“好”帧索引
        s_i: np.nonzero(~label.cpu().numpy()[s_i].squeeze())[0].tolist()
        for s_i in range(len(label))
    }

    inpainting_mask_fixmode = torch.zeros_like(input_motions).bool()  # 生成与输入同形状的布尔张量，初始全 False（表示默认都要重绘）
    for s_i in range(len(input_motions)):                             # 逐个样本处理
        inpainting_mask_fixmode[s_i, ..., det_good_frames_per_sample[s_i]] = True  # 将当前样本的“好帧”位置设为 True，表示这些帧直接保留
    inpainting_mask_fixmode[:, -1] = True                              # 最后一个通道（标签通道）整体设为 True，标签不参与重绘

    inpaint_motion_fixmode = input_motions.clone()  # 作为修复起点
    inpaint_motion_fixmode[:, -1] = -1.0            # 标签通道填入占位值
    inpaint_cond_fixmode = (~inpainting_mask_fixmode) & attention_mask.unsqueeze(-2)  # False 处需修复且在有效帧内

    model_kwargs_fix = {
        "y": {
            "inpainting_mask": inpainting_mask_fixmode.clone(),  # 告知哪些位置保持、哪些重绘
            "inpainted_motion": inpaint_motion_fixmode.clone(),  # 修复的起始内容
        },
        "inpaint_cond": inpaint_cond_fixmode.clone(),            # 需要预测的位置掩码
        "length": length,                                       # 每个样本的有效长度
        "attention_mask": attention_mask,                       # 帧级注意力掩码
    }

    # 可选：软修复（soft-inpaint）调度
    if args.enable_sits and args.ProbDetNum:
        # 将检测阶段的标签通道重复到所有通道，用作软修复步数的权重
        soft_inpaint_ts = einops.repeat(re_sample_det_feats[:, [-1]], "b c l -> b (repeat c) l", repeat=nfeats)
        # 将权重映射到 [0,1]，防止越界
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        # 通过正弦调制映射到扩散步数，得到各位置的软起始步
        soft_inpaint_ts = torch.ceil((torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * args.diffusion_steps).long()
    else:
        soft_inpaint_ts = None                                   # 未启用则不使用软修复

    # 选择采样器
    sample_fn = choose_sampler(diffusion, args.ts_respace)      # 根据配置选择采样器

    # 集成（ensemble）清理流程
    if args.ensemble:
        # 使用集成清理：先用 det 模式评估，再结合修复模式结果做选择
        sample_fix = run_cleanup_selection(
            model=model,
            model_kwargs_detmode={
                "y": {
                    # 只重绘标签通道，其余保持原输入
                    "inpainting_mask": torch.ones_like(input_motions).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False),
                    # 标签通道写 1.0 作为待预测占位
                    "inpainted_motion": input_motions.clone().index_fill_(1, torch.tensor([nfeats-1], device=device), 1.0),
                },
                # 标签通道需要预测且在有效帧内
                "inpaint_cond": ((~torch.ones_like(input_motions).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False))
                                 & attention_mask.unsqueeze(-2)),
                "length": length,                   # 有效帧长度
                "attention_mask": attention_mask,   # 注意力掩码
            },
            model_kwargs=model_kwargs_fix,          # 修复模式的条件
            motion_normalizer=motion_normalizer,    # 反归一化工具
            args=args,
            bs=bs,
            nfeats=nfeats,
            nframes=nframes,
            sample_fn=sample_fn,
            cond_fn=cond_fn if args.classifier_scale else None,  # 可选 classifier 指引
            precomputed_label=label_for_cleanup,
            precomputed_re_sample=re_sample_det_feats_for_cleanup,
        )
    else:
        # 常规单次修复采样
        sample_fix = sample_fn(
            model,
            (bs, nfeats, nframes),                   # 采样目标形状
            clip_denoised=False,                     # 不裁剪去噪结果
            model_kwargs=model_kwargs_fix,           # 修复条件
            skip_timesteps=args.skip_timesteps,      # 可选跳过扩散步
            init_image=model_kwargs_fix["y"]["inpainted_motion"],  # 以占位内容作为起点
            progress=True,                           # 显示进度
            dump_steps=None,                         # 不输出中间步
            noise=None,                              # 默认随机噪声
            const_noise=False,                       # 不固定噪声
            soft_inpaint_ts=soft_inpaint_ts,         # 可选软修复步
            cond_fn=cond_fn if args.classifier_scale else None,  # 可选 classifier 指引
        )

    # 解码：将特征反归一化并拆出身体动作部分
    sample_fix_det = motion_normalizer.inverse(sample_fix.transpose(1, 2).cpu())  # 反归一化到物理尺度
    sample_fix_body = sample_fix_det[..., :-1]                                    # 去掉标签通道，保留动作特征
    fixed_motion = [globsmplrifkefeats_to_smpldata(_s) for _s in sample_fix_body] # 转成 SMPL 数据结构

    return {
        "sample_fix_feats": sample_fix,   # [B, C, N] 修复后的特征（设备上）
        "fixed_motion": fixed_motion,     # list[dict] 解码后的 SMPL 数据
    }


# ---------------------------
# Main
# ---------------------------
def main():
    args = det_args()                             # 解析检测/修复所需的命令行参数
    fixseed(args.seed)                            # 固定随机种子，保证可复现

    # Device / dist
    dist_util.setup_dist(args.device)             # 初始化分布式/设备
    device = dist_util.dev()                      # 获取当前设备

    # Flags
    collect_dataset = args.collect_dataset        # 是否仅收集 GT 动作而不做修复

    # Output path
    out_path = build_output_dir(args)             # 构建输出目录

    # Data
    print(f"Loading dataset from {args.testdata_dir}...")            # 打印数据路径
    data = get_dataset_loader(                                       # 加载测试集（对齐的全局 SMPL RIFKE 特征 + 标签通道）
        name="globsmpl",                                             # 数据集名称
        batch_size=args.batch_size,                                  # batch 大小
        split="test_benchmark",                                      # 使用测试拆分
        data_dir=args.testdata_dir,                                  # 数据路径
        normalizer_dir=args.normalizer_dir,                          # 归一化器存放路径
        shuffle=False,                                               # 不打乱顺序
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer # 获取特征归一化/反归一化工具

    # Model + diffusion
    print("Creating model and diffusion...")
    print("USING Sampler: ", args.ts_respace)
    _model, diffusion = create_model_and_diffusion(args)              # 构建模型与扩散调度
    model = EMA(_model, include_online_model=False) if args.use_ema else _model  # 可选 EMA 包裹

    print(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")      # 读取权重
    model.load_state_dict(state_dict, strict=False)                   # 允许部分松匹配
    model.to(device)                                                  # 移到设备
    model.eval()                                                      # 推理模式

    # Optional guidance
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)        # 可选分类器/约束引导
    if cond_fn is not None:
        cond_fn.keywords["model"] = model                             # 注入模型引用

    # Buffers
    all_motions = []            # 检测阶段解码的 SMPL 字典
    all_lengths = []            # 每个样本的长度列表
    all_input_motions_vec = []  # 修复前的原始特征
    gt_labels_buf = []          # GT 帧级标签

    all_motions_fix = []        # 修复后的 SMPL 字典
    all_fix_motions_vec = []    # 修复后的特征
    labels = []                 # 预测出的帧级标签

    # Loop
    for i, input_batch in enumerate(data):                            # 遍历数据
        input_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_batch.items()}  # 将 batch 中的张量搬到目标设备
        input_motions = input_batch["x"]                 # [B, C, N] 输入特征序列

        attention_mask = input_batch["mask"].squeeze().bool().clone() # 有效帧掩码，去掉多余维度并转为 bool
        length = input_batch["length"]                                  # 每个样本的有效长度

        # For collecting GT motions only
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())  # 把输入反归一化到原尺度以便读取 GT
        if collect_dataset:                                        # 若只想收集 GT 而不做检测/修复
            for _sample in temp_sample:
                all_motions.append(globsmplrifkefeats_to_smpldata(_sample[..., :-1])) # 提取动作部分（去掉标签通道）并保存
            all_lengths.append(length.cpu().numpy())              # 记录长度
            continue                                              # 跳过后续检测/修复

        # Cache GT labels from label channel
        gt_labels = (temp_sample[..., -1] > 0.5).numpy()           # 取标签通道作 GT 布尔
        gt_labels_buf.append(gt_labels)

        # --- Detect ---
        det_out = detect_labels(                                   # 运行检测分支，得到坏帧标签及检测特征
            model=model,                                           # 检测/修复同一模型
            diffusion=diffusion,                                   # 扩散调度器
            args=args,                                             # 运行配置
            input_motions=input_motions,                           # 当前批次输入特征
            length=length,                                         # 每样本长度
            attention_mask=attention_mask,                         # 有效帧掩码
            motion_normalizer=motion_normalizer,                   # 归一化/反归一化工具
        )
        label = det_out["label"]                                   # 预测坏帧掩码
        recon_input_motion = det_out["recon_input_motion"]         # 解码后的检测动作
        re_sample_det_feats = det_out["re_sample_det_feats"]       # 检测特征（设备上）

        all_motions += recon_input_motion                          # 累积检测动作
        all_lengths.append(length.cpu().numpy())                   # 记录长度
        labels.append(label.numpy().copy())                        # 记录预测标签
        all_input_motions_vec.append(input_motions.transpose(1, 2).cpu().numpy())  # 记录原始特征
        print(f"Detected {len(all_motions)} samples")

        # Optionally override with GT labels
        if args.gtdet:                                             # 如需用 GT 覆盖预测
            print("use gt label")
            label = torch.from_numpy(gt_labels.copy())

        # --- Fix ---
        fix_out = fix_motion(                                      # 运行修复分支
            model=model,                                           # 同一扩散模型
            diffusion=diffusion,                                   # 扩散调度器
            args=args,                                             # 运行配置
            input_motions=input_motions,                           # 原始输入特征
            length=length,                                         # 每样本长度
            attention_mask=attention_mask,                         # 有效帧掩码
            motion_normalizer=motion_normalizer,                   # 归一化/反归一化工具
            label=label,                                           # 检测或 GT 的坏帧标签
            re_sample_det_feats=re_sample_det_feats,               # 检测阶段的采样特征，供软修复等使用
            cond_fn=cond_fn,                                       # 可选 classifier guidance
        )
        sample_fix_feats = fix_out["sample_fix_feats"]             # 修复后的特征
        fixed_motion = fix_out["fixed_motion"]                     # 修复后解码的动作

        all_fix_motions_vec.append(sample_fix_feats.transpose(1, 2).cpu().numpy()) # 记录修复特征
        all_motions_fix += fixed_motion                            # 记录修复动作
        print(f"Fixed {len(all_motions_fix)} samples")

        if args.num_samples and args.batch_size * (i + 1) >= args.num_samples:  # 达到采样上限则退出
            break

    # Save results
    all_lengths = np.concatenate(all_lengths, axis=0)              # 将分批长度拼接成一维数组
    os.makedirs(out_path, exist_ok=True)                           # 若目录不存在则创建

    if collect_dataset:                                            # 仅收集 GT 数据的模式
        npy_path = os.path.join(out_path, "results_collected.npy")  # GT 收集文件
        print(f"saving collected motion file to [{npy_path}]")
        np.save(npy_path, {"motion": all_motions, "lengths": all_lengths})  # 仅保存 GT 动作与长度
        with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:        # 另存长度列表便于查看
            fw.write("\n".join([str(l) for l in all_lengths]))
        return                                                     # 收集模式下提前结束

    npy_path = os.path.join(out_path, "results.npy")               # 正常推理结果文件
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,                 # 检测阶段解码的动作
            "motion_fix": all_motions_fix,         # 修复阶段解码的动作
            "label": labels,                       # 预测标签（坏帧掩码）
            "gt_labels": gt_labels_buf,            # GT 标签
            "lengths": all_lengths,                # 每样本长度
            "all_fix_motions_vec": all_fix_motions_vec,   # 修复后的特征序列
            "all_input_motions_vec": all_input_motions_vec, # 原始输入特征序列
        },
    )
    with open(npy_path.replace(".npy", "_len.txt"), "w") as fw:     # 另存长度文本，便于快速查看
        fw.write("\n".join([str(l) for l in all_lengths]))

    # Launch eval
    os.system(f"python -m eval.eval_scripts  --data_path {npy_path} --force_redo")  # 触发评估脚本


if __name__ == "__main__":
    main()
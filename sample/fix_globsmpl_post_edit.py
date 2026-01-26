import os
import numpy as np
from argparse import ArgumentParser


from numpy.random import f
import torch
import einops
from tqdm import tqdm
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True

from utils.fixseed import fixseed
from utils.parser_util import add_base_options, add_sampling_options, parse_and_load_from_model
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata

from ema_pytorch import EMA
from sample.utils import run_cleanup_selection, prepare_cond_fn, choose_sampler, build_output_dir, batch_expander
from sample.PostEdit.Utils import InpaintingOperator, LangevinDynamics, PostEdit_prev_step_PsampleLoop


def post_edit_args():
    """
    解析当前 post edit 脚本所需的命令行参数。
    """
    parser = ArgumentParser(description="PostEdit integration for motion fixing.")
    add_base_options(parser)
    add_sampling_options(parser)
    parser.add_argument(
        "--folder_path",
        type=str,
        default="",
        help="可选：直接遍历某个文件夹，跳过 split 列表。",
    )
    parser.add_argument(
        "--forward_rp_times",
        type=int,
        default=5,
        help="前向复制倍数（候选数量系数）。",
    )
    args = parse_and_load_from_model(parser)
    return args


@torch.no_grad()
def fix_motion(
    *,
    model,                   # 扩散模型主体
    diffusion,               # 扩散调度器
    args,                    # 运行配置
    input_motions,           # [B, C, N] 原始输入动作特征
    length,                  # Tensor[int] shape [B]
    attention_mask,          # [B, N] bool
    motion_normalizer,       # 归一化工具
    cond_fn,
):
    bs, nfeats, nframes = input_motions.shape
    # 1. 执行快速检测获取 Mask (1=好帧, 0=坏帧)
    operator = InpaintingOperator(sigma=0.05)
    mask, y, _re_sample = InpaintingOperator.get_detection_mask(
        model=model,
        diffusion=diffusion, # 传入 diffusion
        input_motions=input_motions,
        length=length,
        attention_mask=attention_mask,
        motion_normalizer=motion_normalizer,
        args=args
    )
    operator.set_mask(mask)

    # 1.5. 准备候选生成 (forward_rp_times)
    forward_rp_times = args.forward_rp_times
    
    # 构造原始修复参数 (batch 大小为 bs)
    inpainting_mask_fixmode = mask.expand(-1, nfeats, -1).bool().clone()
    inpainting_mask_fixmode[:, -1] = True                        # 标签通道始终保留

    # inpaint_motion: 修复起点，标签通道设为 -1.0 占位
    inpaint_motion_fixmode = input_motions.clone()
    inpaint_motion_fixmode[:, -1] = -1.0
    inpaint_cond_fixmode = (~inpainting_mask_fixmode) & attention_mask.unsqueeze(-2)

    model_kwargs_fix = {
        "y": {
            "inpainting_mask": inpainting_mask_fixmode.clone(), 
            "inpainted_motion": inpaint_motion_fixmode.clone(), 
        },
        # "inpaint_cond": torch.ones_like(inpaint_cond_fixmode).bool(), # 全部都需要修改
        "inpaint_cond": inpaint_cond_fixmode.clone(), 
        "length": length,
        "attention_mask": attention_mask,
    }

    # 使用 batch_expander 扩展 batch 大小
    rp_model_kwargs_fix = batch_expander(model_kwargs_fix, forward_rp_times)
    rp_input_motions = einops.repeat(input_motions, "b c l -> (repeat b) c l", repeat=forward_rp_times)
    rp_y = einops.repeat(y, "b c l -> (repeat b) c l", repeat=forward_rp_times)
    rp_mask = einops.repeat(mask, "b c l -> (repeat b) c l", repeat=forward_rp_times)
    operator.set_mask(rp_mask) # 更新算子的 mask 到扩展后的形状

    # 可选软修复步调度 (SITS)
    if args.enable_sits:
        rp_re_sample = einops.repeat(_re_sample, "b c l -> (repeat b) c l", repeat=forward_rp_times)
        soft_inpaint_ts = einops.repeat(rp_re_sample[:, [-1]], 'b c l -> b (repeat c) l', repeat=nfeats)
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        soft_inpaint_ts = torch.ceil((torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * args.diffusion_steps).long()
    else:
        soft_inpaint_ts = None

    # 3. 初始化 Langevin 配置
    lgvd_config = {
        "num_steps": args.lgvd_num_steps,       # 郎之万优化迭代步数
        "lr": args.lgvd_lr,                    # 学习率
        "tau": args.lgvd_tau,                   # 噪声项系数 (增加 tau 以放宽测量约束)          
        "lr_min_ratio": args.lgvd_lr_min_ratio
    }
    lgvd = LangevinDynamics(**lgvd_config)

    # 4. 执行多候选采样的“加噪-去噪-朗之万优化”循环
    sample_fix_candidates = PostEdit_prev_step_PsampleLoop(
        diffusion,
        model,
        (bs * forward_rp_times, nfeats, nframes),
        clip_denoised=False,
        model_kwargs=rp_model_kwargs_fix,
        skip_timesteps=0,            # 从全噪声开始
        init_motions=rp_input_motions,  # 重要：提供原始参考以供锚定混合
        progress=True,
        dump_steps=None,                                             # 不导出中间步
        noise=None,                                                  # 默认随机噪声
        const_noise=False,                                           # 不固定噪声
        soft_inpaint_ts=soft_inpaint_ts,                             # 可选软修复步调度
        cond_fn=cond_fn if args.classifier_scale else None,          # 可选 classifier guidance
        use_postedit=args.use_postedit,           
        operator=operator,                          # 测量算子
        measurement=rp_y,                          # 测量值 y
        lgvd=lgvd,                                 # 郎之万动力学优化器
        w=args.postedit_w,                         # 混合权重
    )

    # # 使用StabelMotion的原始采样器进行对比，其实就是use_postedit = false
    # sample_fn = choose_sampler(diffusion, args.ts_respace)
    # sample_fix_candidates = sample_fn(
    #     model,
    #     (bs * forward_rp_times, nfeats, nframes),
    #     clip_denoised=False,
    #     model_kwargs=rp_model_kwargs_fix,
    #     skip_timesteps=args.skip_timesteps,
    #     init_image=rp_input_motions,
    #     progress=True,
    #     dump_steps=None,                                             # 不导出中间步
    #     noise=None,                                                  # 默认随机噪声
    #     const_noise=False,                                           # 不固定噪声
    #     soft_inpaint_ts=soft_inpaint_ts,                             # 可选软修复步调度
    #     cond_fn=cond_fn if args.classifier_scale else None,          # 可选 classifier guidance
    # )

    # 5. 打分并选择最佳候选 (参考 utils.py run_cleanup_selection)
    eval_times = 25
    device = input_motions.device
    
    # 构造用于打分的检测模式参数
    model_kwargs_detmode = {
        "y": {
            "inpainting_mask": torch.ones_like(sample_fix_candidates).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False),
            "inpainted_motion": sample_fix_candidates.clone().index_fill_(1, torch.tensor([nfeats-1], device=device), 1.0),
        },
        "inpaint_cond": ((~torch.ones_like(sample_fix_candidates).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False))
                        & rp_model_kwargs_fix['attention_mask'].unsqueeze(1)), 
        "length": rp_model_kwargs_fix['length'],
        "attention_mask": rp_model_kwargs_fix['attention_mask'],
    }

    score = 0
    with torch.no_grad():
        t_idx = args.diffusion_steps - 1
        _re_t = torch.ones((bs * forward_rp_times,), device=device).long() * t_idx
        t_input = _re_t
        if hasattr(diffusion, "_scale_timesteps"):
            t_input = diffusion._scale_timesteps(_re_t)
            
        for _ in tqdm(range(eval_times), desc="Scoring Candidates"):
            noise_x = torch.randn_like(model_kwargs_detmode['y']['inpainted_motion'])
            cond = model_kwargs_detmode['inpaint_cond']
            x_gt = model_kwargs_detmode['y']['inpainted_motion']
            x_input = torch.where(cond, noise_x, x_gt)
            score += model(x_input, t_input, **model_kwargs_detmode)[:, -1] # 累加标签通道得分
    
    score /= eval_times
    score = torch.sum((score > 0.0) * rp_model_kwargs_fix['attention_mask'], dim=-1)  # 对有效帧求和得分
    score = einops.rearrange(score, "(repeat b) -> repeat b", repeat=forward_rp_times)     # [forward_rp_times, bs]

    sample_candidates_reshaped = einops.rearrange(sample_fix_candidates, "(repeat b) c l -> repeat b c l", repeat=forward_rp_times)  # 重排候选 [repeat, bs, C, L]
    selected_id = torch.argmin(score, dim=0)                                       # [bs] 选择分数最低的候选
    selected_id_for_gather = selected_id.view(1, bs, 1, 1).expand(1, bs, nfeats, nframes) # 展开成索引形状
    sample_fix = torch.gather(sample_candidates_reshaped, dim=0, index=selected_id_for_gather).squeeze(0) # 挑出对应的最佳候选


    # 6. 解码与后处理
    # 将特征反归一化并拆出身体动作部分
    # 修正：按样本单独处理，确保维度和长度正确
    fixed_motion = []
    sample_fix_cpu = sample_fix.transpose(1, 2).cpu() # [B, N, C]
    for b in range(bs):
        valid_len = int(length[b].item())
        # 仅取有效长度部分进行反归一化和转换
        feat_b = sample_fix_cpu[b, :valid_len, :]
        feat_b_denorm = motion_normalizer.inverse(feat_b.unsqueeze(0))[0] # [N_valid, C]
        fixed_motion.append(globsmplrifkefeats_to_smpldata(feat_b_denorm[..., :-1]))

    # 将 mask 转换回布尔标签
    labels = (mask.squeeze(1) < 0.5).cpu().numpy()

    print(f"sample_fix: {sample_fix}")

    return {
        "sample_fix_feats": sample_fix,   # [B, C, N] 修复后的特征
        "fixed_motion": fixed_motion,     # 解码后的 SMPL 数据列表
        "label": labels,                  # 检测到的坏帧标签
    }


def main():
    args = post_edit_args()                       # 解析检测/修复所需的命令行参数
    fixseed(args.seed)                            # 固定随机种子，保证可复现

    dist_util.setup_dist(args.device)             # 初始化分布式/设备
    device = dist_util.dev()                      # 获取当前设备

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
        sample_mode="sequential",
        sequence_window_size=None,
        sequence_stride=None,
        folder_path=args.folder_path or None,
    )
    motion_normalizer = data.dataset.motion_loader.motion_normalizer # 获取特征归一化/反归一化工具

    # region 创建 model 和 diffusion
    print("创建 model 和 diffusion...")
    print("使用采样器: ", args.ts_respace)
    _model, diffusion = create_model_and_diffusion(args)              # 构建模型与扩散调度
    model = EMA(_model, include_online_model=False) if args.use_ema else _model  # 可选 EMA 包裹
    #endregion

    # region 加载检查点
    print(f"加载检查点： [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location="cpu")      # 读取权重
    model.load_state_dict(state_dict, strict=False)                   # 允许部分松匹配
    model.to(device)                                                  # 移到设备
    model.eval()                                                      # 推理模式
    #endregion

    # region 可选分类器/约束引导
    cond_fn = prepare_cond_fn(args, motion_normalizer, device)        # 可选分类器/约束引导
    if cond_fn is not None:
        cond_fn.keywords["model"] = model                             # 注入模型引用
    #endregion

    # region 缓冲区：存储处理过程中的各种数据
    all_motions = []            # 检测阶段解码的 SMPL 字典
    all_lengths = []            # 每个样本的长度列表
    all_input_motions_vec = []  # 修复前的原始特征
    gt_labels_buf = []          # GT 帧级标签
    all_motions_fix = []        # 修复后的 SMPL 字典
    all_fix_motions_vec = []    # 修复后的特征
    labels = []                 # 预测出的帧级标签
    #endregion

    for i, input_batch in enumerate(data):                            # 遍历数据
        input_batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in input_batch.items()}  # 将 batch 中的张量搬到目标设备
        input_motions = input_batch["x"]                 # [B, C, N] 输入特征序列
        attention_mask = input_batch["mask"].squeeze(1).bool().clone() # 有效帧掩码，保留 [B, N] 形状
        length = input_batch["length"]                                  # 每个样本的有效长度

        print(f"input_motions.shape: {input_motions.shape}")
        print(f"attention_mask.shape: {attention_mask.shape}")
        print(f"length: {length}")

        # Cache GT labels from label channel
        temp_sample_normalized = input_motions.transpose(1, 2).cpu()  # [B, N, C]
        
        # 修正：记录原始动作解码结果，用于可视化对比（按样本单独处理有效长度）
        for b in range(input_motions.shape[0]):
            valid_len = int(length[b].item())
            feat_b = temp_sample_normalized[b, :valid_len, :]
            feat_b_denorm = motion_normalizer.inverse(feat_b.unsqueeze(0))[0] # [N_valid, C]

            smpl_dict = globsmplrifkefeats_to_smpldata(feat_b_denorm[..., :-1])
            smpl_dict = {k: v.numpy() if torch.is_tensor(v) else v for k, v in smpl_dict.items()}
            all_motions.append(smpl_dict)
            
            # 从反归一化后的数据中提取 GT 标签（最后一维）
            gt_labels = (feat_b_denorm[..., -1] > 0.5).numpy()
            gt_labels_buf.append(gt_labels)

        all_input_motions_vec.append(input_motions.transpose(1, 2).cpu().numpy())  # 记录原始特征

        fix_out = fix_motion(                                      # 运行修复分支
            model=model,                                           # 同一扩散模型
            diffusion=diffusion,                                   # 扩散调度器
            args=args,                                             # 运行配置
            input_motions=input_motions,                           # 原始输入特征
            length=length,                                         # 每样本长度
            attention_mask=attention_mask,                         # 有效帧掩码
            motion_normalizer=motion_normalizer,
            cond_fn=cond_fn,
        )
        sample_fix_feats = fix_out["sample_fix_feats"]             # 修复后的特征
        fixed_motion = fix_out["fixed_motion"]                     # 修复后解码的动作
        pred_label = fix_out["label"]                              # 预测出的坏帧标签

        all_fix_motions_vec.append(sample_fix_feats.transpose(1, 2).cpu().numpy()) # 记录修复特征
        all_motions_fix += fixed_motion                            # 记录修复动作
        labels.append(pred_label)                                  # 记录预测标签
        all_lengths.append(length.cpu().numpy())                   # 记录长度

        print(f"Fixed {len(all_motions_fix)} samples")

        if args.num_samples and args.batch_size * (i + 1) >= args.num_samples:  # 达到采样上限则退出
            break

    # Save results
    all_lengths = np.concatenate(all_lengths, axis=0)              # 将分批长度拼接成一维数组
    os.makedirs(out_path, exist_ok=True)                           # 若目录不存在则创建

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
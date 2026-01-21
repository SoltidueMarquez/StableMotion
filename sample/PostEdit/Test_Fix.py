import os
import numpy as np
from argparse import ArgumentParser

from numpy.random import f
import torch
import einops
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cudnn.benchmark = True

from utils.fixseed import fixseed
from utils.parser_util import add_base_options, add_sampling_options, parse_and_load_from_model
from utils.model_util import create_model_and_diffusion
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata

from ema_pytorch import EMA
from sample.utils import run_cleanup_selection, prepare_cond_fn, choose_sampler, build_output_dir
from sample.PostEdit.Utils import InpaintingOperator,PostEdit_prev_step_PsampleLoop


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
    args = parse_and_load_from_model(parser)
    return args


def get_start(diffusion, ref, starting_timestep=999,noise=None):
    """
    根据输入动作序列 ref 添加指定步数的噪声。
    ref: [B, C, N]
    starting_timestep: 起始时间步 (通常是 0 到 T-1 之间的整数)
    """
    device = ref.device
    # 1. 确保 timestep 是张量且在正确设备上
    t = torch.tensor([starting_timestep] * ref.shape[0], device=device).long()
        
    # 修复 Bug 1：处理 SpacedDiffusion 的时间步映射
    # 如果 starting_timestep 是重采样后的索引（如 49），我们需要获取它对应的原始时间步（如 980）
    t_orig = t
    if hasattr(diffusion, "timestep_map"):
        # timestep_map 存储了重采样索引到原始索引的映射
        t_orig = torch.tensor([diffusion.timestep_map[idx.item()] for idx in t], device=device).long()

    # Debug: 检查加噪参数
    print(f"[Debug] get_start (加噪):")
    print(f" - 输入的时间步 t: {starting_timestep}")
    print(f" - 映射过后的时间步 t: {t_orig[0].item()}")
    if hasattr(diffusion, "sqrt_alphas_cumprod"):
        print(f" - sqrt_alphas_cumprod: {diffusion.sqrt_alphas_cumprod[t_orig[0].item()]:.4f}")
        print(f" - sqrt_one_minus_alphas_cumprod: {diffusion.sqrt_one_minus_alphas_cumprod[t_orig[0].item()]:.4f}")

    # 2. 如果没有提供噪声，则生成随机噪声
    if noise is None:
        noise = torch.randn_like(ref)
    # 3. 调用 StableMotion 的 q_sample 进行前向加噪
    x_start = diffusion.q_sample(ref, t_orig, noise=noise)

    return x_start


def sampler_one_step(
        diffusion,
        model,
        timestep, 
        sample, 
        model_kwargs
    ):
        """
        单步采样：从 x_t 预测 x_0。
        适配 StableMotion (OpenAI Diffusion) 架构。
        """
        device = sample.device
        # 1. 准备时间步张量
        t = torch.tensor([timestep] * sample.shape[0], device=device).long()
        
        # 映射到原始时间步 (用于 diffusion 的系数查找)
        t_orig = t
        if hasattr(diffusion, "timestep_map"):
            t_orig = torch.tensor([diffusion.timestep_map[idx.item()] for idx in t], device=device).long()

        # 2. 预测噪声
        # 注意：在 StableMotion 中，model 会根据 model_kwargs 处理 inpainting 等条件
        with torch.no_grad():
            # 检查是否有时间步缩放逻辑 (用于模型 embedding)
            t_input = t
            if hasattr(diffusion, "_scale_timesteps"):
                t_input = diffusion._scale_timesteps(t)
            
            # Debug: 检查预测参数
            print(f"   [Debug] sampler_one_step (去噪预测):")
            print(f"    - 函数输入时间步 t: {timestep}")
            print(f"    - 映射过后时间步 t: {t_orig[0].item()}")
            print(f"    - 缩放过后时间步 t (model input): {t_input[0].item()}")

            model_output = model(sample, t_input, **model_kwargs)
            
        # 3. 处理 Learned Sigma：如果输出通道是输入的两倍，截取前一半
        if model_output.shape[1] == sample.shape[1] * 2:
            model_output, _ = torch.split(model_output, sample.shape[1], dim=1)

        # 4. 根据 model_mean_type 预测 x_0
        from diffusion.gaussian_diffusion import ModelMeanType
        print(f"    - 模型预测类型: {diffusion.model_mean_type}")
        if diffusion.model_mean_type == ModelMeanType.START_X:
            # StableMotion 默认：模型直接输出 x_0
            print(f"    - [OK] START_X 类型，直接使用模型输出作为 pred_x0")
            pred_x0 = model_output
        elif diffusion.model_mean_type == ModelMeanType.EPSILON:
            # 只有预测噪声时，才调用转换公式
            print(f"    - EPSILON 类型，调用公式转换 pred_xstart_from_eps")
            pred_x0 = diffusion._predict_xstart_from_eps(x_t=sample, t=t_orig, eps=model_output)
        else:
            raise NotImplementedError(f"不支持的 model_mean_type: {diffusion.model_mean_type}")

        return pred_x0


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
):
    bs, nfeats, nframes = input_motions.shape
    # 1. 执行快速检测获取 Mask (1=好帧, 0=坏帧)
    operator = InpaintingOperator(sigma=0.05)
    mask, y = InpaintingOperator.get_detection_mask(
        model=model,
        diffusion=diffusion, # 传入 diffusion
        input_motions=input_motions,
        length=length,
        attention_mask=attention_mask,
        motion_normalizer=motion_normalizer,
        args=args
    )
    operator.set_mask(mask)

    # 2. 构造模型预测所需的 model_kwargs
    # model_kwargs_fix = {
    #     "y": {
    #         "inpainting_mask": torch.zeros_like(input_motions).bool(), #这没什么软用
    #         "inpainted_motion": input_motions.clone(), 
    #     },
    #     # 这里都是要修改的
    #     "inpaint_cond": torch.ones_like(input_motions).bool(), 
    #     "length": length,
    #     "attention_mask": attention_mask,
    # }
    model_kwargs_fix = {
        "y": {
            "inpainting_mask": torch.zeros_like(input_motions).bool(), #这没什么软用
            "inpainted_motion": input_motions.clone(), 
        },
        "inpaint_cond": (1.0 - mask).expand(-1, nfeats, -1), 
        "length": length,
        "attention_mask": attention_mask,
    }
    sample_fn = choose_sampler(diffusion, args.ts_respace)

    # 4. 执行“加噪-去噪-优化”循环
    # 注意：init_image 配合 skip_timesteps 可以实现“从中间步加噪修复”
    # 替换原来的加噪和 sampler_one_step 部分
    sample_fix = PostEdit_prev_step_PsampleLoop(
        diffusion,
        model,
        (bs, nfeats, nframes),
        clip_denoised=False,
        model_kwargs=model_kwargs_fix,
        skip_timesteps=0,  # 0 表示从纯噪声开始，如果要 Post-Edit 建议设为 30-40 (对应 50 步采样)
        init_motions=input_motions, # Post-Edit 的原始参考
        progress=True,
        use_postedit=False,
        operator=operator,
        measurement=y,
        lgvd=None,
        w=1.0,
    )

    # 使用StabelMotion的原始采样器进行对比
    # 构造模型预测所需的 model_kwargs
    # sample_fix = sample_fn(
    #     model,
    #     (bs, nfeats, nframes),
    #     clip_denoised=False,
    #     model_kwargs=model_kwargs_fix,
    #     skip_timesteps=0,
    #     init_motions=input_motions,
    #     progress=True,
    # )



    # 5. 解码与后处理
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

    # # region 可选分类器/约束引导
    # cond_fn = prepare_cond_fn(args, motion_normalizer, device)        # 可选分类器/约束引导
    # if cond_fn is not None:
    #     cond_fn.keywords["model"] = model                             # 注入模型引用
    # #endregion

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
            motion_normalizer=motion_normalizer,                   # 归一化/反归一化工具
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
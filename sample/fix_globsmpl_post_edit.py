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
def fix_motion(
    *,
    model,                   # 扩散模型主体（含 inpaint 能力）
    diffusion,               # 扩散调度器/步数配置
    args,                    # 运行配置，包含跳步、集成、软修复等开关
    input_motions,           # [B, C, N] 原始输入动作特征
    length,                  # Tensor[int] shape [B]，每条序列的有效帧长度
    attention_mask,          # [B, N] bool，标记有效帧区域
    motion_normalizer,       # 用于特征归一化/反归一化的工具
    cond_fn=None,            # 可选 classifier guidance 函数
):
    device = input_motions.device
    bs, nfeats, nframes = input_motions.shape

    # TODO: 修改为post edit的修复思路

    # TODO: 这边需要先构造model_kwargs，这里是全局加噪的
    # model_kwargs_fix = {
    #     "y": {
    #         "inpainting_mask": inpainting_mask_fixmode.clone(),  # 告知哪些位置保持、哪些重绘
    #         "inpainted_motion": inpaint_motion_fixmode.clone(),  # 修复的起始内容
    #     },
    #     "inpaint_cond": inpaint_cond_fixmode.clone(),            # 需要预测的位置掩码
    #     "length": length,                                       # 每个样本的有效长度
    #     "attention_mask": attention_mask,                       # 帧级注意力掩码
    # }    

    starting_timestep = 501 # 扩散步的起点 TODO：这里需要和项目保持一致

    # 获取带噪声的潜变量起点
    motions_t = null_inversion.get_start(input_motions, starting_timestep=starting_timestep)
    # 执行带有郎之万优化和注意力控制的采样
    # TODO：这里的Operator是InpaintingOperator，需要调用 set_mask 与 get_detection_mask 来设置和获取当前的mask
    samples = null_inversion.sample_in_batch(motions_t, operator, y, starting_timestep=starting_timestep)
  
    sample_fix = samples[1].unsqueeze(0)

    # 解码：将特征反归一化并拆出身体动作部分
    sample_fix_det = motion_normalizer.inverse(sample_fix.transpose(1, 2).cpu())  # 反归一化到物理尺度
    sample_fix_body = sample_fix_det[..., :-1]                                    # 去掉标签通道，保留动作特征
    fixed_motion = [globsmplrifkefeats_to_smpldata(_s) for _s in sample_fix_body] # 转成 SMPL 数据结构

    return {
        "sample_fix_feats": sample_fix,   # [B, C, N] 修复后的特征（设备上）
        "fixed_motion": fixed_motion,     # list[dict] 解码后的 SMPL 数据
    }


def main():
    args = det_args()                             # 解析检测/修复所需的命令行参数
    fixseed(args.seed)                            # 固定随机种子，保证可复现

    dist_util.setup_dist(args.device)             # 初始化分布式/设备
    device = dist_util.dev()                      # 获取当前设备

    out_path = build_output_dir(args)             # 构建输出目录

    # Data TODO：这边可能需要像long_smart一样，支持folder_path
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

        attention_mask = input_batch["mask"].squeeze().bool().clone() # 有效帧掩码，去掉多余维度并转为 bool
        length = input_batch["length"]                                  # 每个样本的有效长度

        # Cache GT labels from label channel
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())  # 把输入反归一化到原尺度以便读取 GT
        gt_labels = (temp_sample[..., -1] > 0.5).numpy()           # 取标签通道作 GT 布尔
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

    npy_path = os.path.join(out_path, "results.npy")               # 正常推理结果文件
    print(f"saving results file to [{npy_path}]")
    np.save(
        npy_path,
        {
            "motion": all_motions,                 # 检测阶段解码的动作
            "motion_fix": all_motions_fix,         # 修复阶段解码的动作
            "label": labels,                       # 预测标签（坏帧掩码）# TODO:需要补一下
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
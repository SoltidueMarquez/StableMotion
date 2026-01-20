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
from sample.PostEdit.Utils import InpaintingOperator, NullInversion, LangevinDynamics


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
        input_motions=input_motions,
        length=length,
        attention_mask=attention_mask,
        motion_normalizer=motion_normalizer,
        args=args
    )
    operator.set_mask(mask)

    # 2. 构造模型预测所需的 model_kwargs
    # 修正：将检测到的 mask 传递给模型，让模型知道哪些帧是完好的（1），哪些是损坏的（0）
    model_kwargs_fix = {
        "y": {
            # 保持全 False。这意味着在模型的前向传播内部，不会强行用原图替换特征。
            # 这样模型就拥有了对全序列（包括好帧和坏帧）进行全局去噪和调整的自由。
            "inpainting_mask": torch.zeros_like(input_motions).bool(), 
            "inpainted_motion": input_motions.clone(), 
        },
        # 将检测出的 mask (1=好, 0=坏) 传给模型。
        # 这为模型提供了重要的上下文锚点，让它在全局去噪时知道“好帧”应该还原成什么样，
        # 从而避免输出坍缩到全 0 或均值，并能更合理地补全“坏帧”。
        "inpaint_cond": mask.expand(-1, nfeats, -1), 
        "length": length,
        "attention_mask": attention_mask,
    }

    # 3. 初始化 NullInversion 与 Langevin 配置
    # 增加 tau 以减少梯度爆炸风险，减少迭代步数以提高速度和稳定性
    lgvd_config = {
        "num_steps": 50,       # 郎之万优化迭代步数
        "lr": 5e-5,            # 学习率
        "tau": 0.1,            # 噪声项系数 (增加 tau 以放宽测量约束)
        "lr_min_ratio": 0.1
    }
    null_inversion = NullInversion(diffusion, model, lgvd_config)

    # 4. 执行“加噪-去噪-优化”循环
    # 扩散步起点建议与模型推理步数保持一致（如 50 步 DDIM 对应的内部步数）
    starting_timestep = args.diffusion_steps - 1 
    
    # 全局加噪：给原始动作序列添加噪声作为起点
    motions_t = null_inversion.get_start(input_motions, starting_timestep=starting_timestep)
    
    # 采样去噪与朗之万优化
    # annel_interval 控制扩散步的跳跃，w 控制与原图的混合比例
    sample_fix = null_inversion.sample_in_batch(
        x_start=motions_t,
        operator=operator,
        y=y,
        orginal_input_motions=input_motions,
        model_kwargs=model_kwargs_fix,
        starting_timestep=starting_timestep
    )

    # 5. 解码与后处理
    # 将特征反归一化并拆出身体动作部分
    sample_fix_det = motion_normalizer.inverse(sample_fix.transpose(1, 2).cpu())
    sample_fix_body = sample_fix_det[..., :-1] # 去掉最后一维标签通道
    fixed_motion = [globsmplrifkefeats_to_smpldata(_s) for _s in sample_fix_body]

    # 将 mask 转换回布尔标签 [B, N] 用于返回
    labels = (mask.squeeze(1) < 0.5).cpu().numpy()

    print(f"sample_fix: {sample_fix}")

    return {
        "sample_fix_feats": sample_fix,   # [B, C, N] 修复后的特征
        "fixed_motion": fixed_motion,     # 解码后的 SMPL 数据
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
        temp_sample = motion_normalizer.inverse(input_motions.transpose(1, 2).cpu())  # 把输入反归一化到原尺度以便读取 GT
        # 记录原始动作解码结果，用于可视化对比
        all_motions += [globsmplrifkefeats_to_smpldata(_s) for _s in temp_sample[..., :-1]] 
        
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
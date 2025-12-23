
import os
from copy import deepcopy
from functools import partial
import numpy as np
import torch
from utils import dist_util
from copy import deepcopy
from data_loaders.amasstools.globsmplrifke_feats import globsmplrifkefeats_to_smpldata
from eval.eval_motion import compute_foot_sliding_wrapper_torch
from smplx.lbs import batch_rigid_transform
import einops
from tqdm import tqdm
from data_loaders.amasstools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
    axis_angle_to_quaternion,
    quaternion_to_axis_angle,
)

def build_output_dir(args) -> str:
    """Derive output directory name from args and model checkpoint name."""
    out_path = args.output_dir
    if out_path != "":
        return out_path

    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")

    out_path = os.path.join(
        os.path.dirname(args.model_path), f"Fix_{name}_{niter}_seed{args.seed}"
    )
    if args.dataset_path != "":
        out_path += f'_{os.path.split(args.dataset_path)[-1]}'
    if args.skip_timesteps != 0:
        out_path += f"_skip{args.skip_timesteps}"
    if args.enable_cfg:
        out_path += f"_cfg{args.guidance_param}"
    if args.classifier_scale:
        out_path += f"_cs{args.classifier_scale}"
    if args.ProbDetNum:
        out_path += f"_pdn{args.ProbDetNum}"
    if args.enable_sits:
        out_path += "_softi"
    if args.ensemble:
        out_path += "_esnb"
    if args.ext != "":
        out_path += f"_ext{args.ext}"
    return out_path


def prepare_cond_fn(args, motion_normalizer, device):
    """
    Prepare optional foot-locking guidance function (for classifier guidance).
    Returns cond_fn or None.
    """
    if not args.classifier_scale:
        return None

    print("Preparing cond function ...")
    j_regressor_stat = np.load("data_loaders/amasstools/smpl_neutral_nobetas_24J.npz")
    J_regressor = torch.from_numpy(j_regressor_stat["J"]).to(device)
    parents = torch.from_numpy(j_regressor_stat["parents"])
    root_offset = torch.tensor([-0.00179506, -0.22333382, 0.02821918]).to(device)
    std = motion_normalizer.std.clone().to(device)
    mean = motion_normalizer.mean.clone().to(device)

    return partial(
        footlocking_fn,
        model=None,  # set later when model is ready
        mean=mean,
        std=std,
        classifier_scale=args.classifier_scale,
        J_regressor=J_regressor,
        parents=parents,
        root_offset=root_offset,
    )

def choose_sampler(diffusion, ts_respace: bool):
    """
    Pick DDIM or ancestral sampler.
    现有 Enhanced 配置用的是默认q_sample，不是 DDIM
    """
    return diffusion.ddim_sample_loop if ts_respace else diffusion.p_sample_loop

def batch_expander(model_kwargs, repeat_times):
    out_model_kwargs = deepcopy(model_kwargs)
    if 'y' in out_model_kwargs:
        out_model_kwargs['y'] = batch_expander(model_kwargs['y'], repeat_times)
    for k, v in model_kwargs.items():
        if k == 'y':
            continue
        else:
            if isinstance(v, list):
                out_model_kwargs[k] = v * repeat_times
            elif isinstance(v, (torch.Tensor, np.ndarray)):
                out_model_kwargs[k] = einops.repeat(v, 'b ... -> (repeat b) ...', repeat=repeat_times)
    return out_model_kwargs

def footlocking_fn(x, t, model=None, mean=None, std=None, classifier_scale=0., J_regressor=None, parents=None, root_offset=None, **kwargs):
    lengths = kwargs["length"]
    eps = 1e-12
    with torch.autograd.set_detect_anomaly(False):
        with torch.enable_grad():
            inpaint_cond = kwargs['inpaint_cond']
            x_gt = kwargs['y']['inpainted_motion']
            x_in = x.detach().requires_grad_(True)
            loss = 0.
            x_in = torch.where(inpaint_cond, x_in, x_gt)
            x_0 = model(x_in, t, **kwargs)
            x_0 = torch.where(inpaint_cond, x_0, x_gt)
            denorm_x0 = x_0.transpose(1, 2) * (std + eps) + mean
            joints = []
            loss = 0
            B = denorm_x0.shape[0]
            denorm_x0_flatten = einops.rearrange(denorm_x0, "b n d -> (b n) d")
            smpldata = globsmplrifkefeats_to_smpldata(denorm_x0_flatten[..., :-1])
            poses = smpldata["poses"]
            trans = smpldata["trans"]
            poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)
            rot_mat = axis_angle_to_matrix(poses)
            T = rot_mat.shape[0]
            zero_hands_rot = torch.eye(3)[None, None].expand(T, 2, -1, -1).to(dist_util.dev())
            rot_mat = torch.concat((rot_mat, zero_hands_rot), dim=1)
            joints, _ = batch_rigid_transform(
                rot_mat,
                J_regressor[None].expand(T, -1, -1),
                parents,
            )
            joints = joints.squeeze() + trans.unsqueeze(1) - root_offset
            joints = einops.rearrange(joints, "(b n) j d -> b n j d", b=B)
            slide_dist = compute_foot_sliding_wrapper_torch(joints, lengths, upaxis=2, ankle_h=0.1)
            loss = sum(slide_dist)
            grad = torch.autograd.grad(-loss, x_in)[0] * classifier_scale
        
        grad = torch.nan_to_num(grad)
        grad = torch.clip(grad, min=-10, max=10) # [b d n] 
        grad[:, 0] = 0.

    return grad

def run_cleanup_selection(
    model,                 # 扩散模型主体，既可检测也可修复
    model_kwargs_detmode,  # 检测模式用的输入/掩码/长度等条件
    model_kwargs,          # 修复模式用的输入/掩码/长度等条件
    motion_normalizer,     # 归一化/反归一化工具
    sample_fn,             # 采样循环函数（DDIM 或 p_sample）
    cond_fn,               # 可选 classifier guidance 函数
    args,                  # 运行配置（阈值、步数、开关等）
    bs,                    # batch 大小
    nfeats,                # 通道数
    nframes,               # 帧数
    precomputed_label=None,       # 可选：上游检测阶段的坏帧布尔掩码
    precomputed_re_sample=None,   # 可选：与 precomputed_label 对应的重采样特征（模型归一化空间）
):
    """
    封装检测→修复的集成流程：
      - 在检测模式下做多次快速前向，得到平均检测结果
      - 基于检测结果膨胀、构建修复掩码与条件
      - 在修复模式下扩展 batch 重复采样，得到若干候选
      - 再用检测模式对候选打分，挑选最优的修复结果
    返回：
      sample (Tensor): 选中的修复结果 [bs, nfeats, nframes]
    """
    sample_candidates = []
    forward_rp_times = 5  # 固定：前向复制倍数（候选数量系数）
    eval_times = 25  # 固定：快速检测评估次数（做均值）

    # 把检测模式的 kwargs 复制 forward_rp_times 份，扩成更大的 batch，用于后续多次采样。
    rp_model_kwargs_detmode = batch_expander(model_kwargs_detmode, forward_rp_times)  # 复制检测模式条件，扩展 batch

    # 在 torch.no_grad() 下用同一个 model 做了 eval_times 次前向检测（随机噪声起点，仅在待预测位置保留噪声），
    # 把输出累加后取平均，目的是降低单次检测的随机波动，得到更稳定的检测结果；
    # 随后反归一化并在标签通道做阈值，得到坏帧布尔掩码。
    # 也就是在检测模式下做多次快速前向，得到平均检测结果

    if precomputed_re_sample is None :
        with torch.no_grad():                                      # 检测模式下快速均值
            _re_sample = 0                                         # 累积检测输出，与模型在该时间步的输出一致，[bs * forward_rp_times, nfeats, nframes]。
            _re_t = torch.ones((bs * forward_rp_times,), device=dist_util.dev()) * 49  # 固定时间步
            for _ in tqdm(range(eval_times)):                      # 多次快速检测采样
                x = torch.randn_like(rp_model_kwargs_detmode['y']['inpainted_motion'])  # 随机噪声起点
                inpaint_cond = rp_model_kwargs_detmode['inpaint_cond']                  # 需要预测的位置
                x_gt = rp_model_kwargs_detmode['y']['inpainted_motion']                 # 已知位置填原值
                x = torch.where(inpaint_cond, x, x_gt)                                  # 只在待预测位置保持噪声
                _re_sample += model(x, _re_t, **rp_model_kwargs_detmode)                # 前向得到检测输出
            _re_sample = _re_sample / eval_times                                        # 求均值，降低方差

        # 之前的 _re_sample 还在“模型归一化空间”——训练时输入被 mean/std 标准化，模型输出也在同一尺度里。
        # 现在需要反归一化到物理尺度，才能与真实运动数据对比。
        _sample = motion_normalizer.inverse(_re_sample.transpose(1, 2).cpu())           # 反归一化到原尺度
    else :
        _re_sample = precomputed_re_sample.clone()
        _re_sample = einops.repeat(_re_sample, "b c l -> (repeat b) c l", repeat=forward_rp_times).clone().contiguous()


    if precomputed_label is None :
        _label = _sample[..., -1] > args.ProbDetTh                                      # 标签通道阈值化，得到坏帧布尔
    else :
        _label = precomputed_label.clone()
        _label = einops.repeat(_label, "b l -> (repeat b) l", repeat=forward_rp_times).clone().contiguous().cpu()


    # -------------------------------
    # Preparing for Fixing Mode
    # -------------------------------
    temp_labels = _label.clone()                                                     # 备份坏帧标记
    _label[..., 1:] += temp_labels[..., :-1]                                         # 左膨胀 1 帧
    _label[..., :-1] += temp_labels[..., 1:]                                         # 右膨胀 1 帧
    for mids, mlen in enumerate(rp_model_kwargs_detmode['length'].cpu().numpy()):    # 保证末帧为好帧
        _label[mids, ..., mlen - 1] = 0

    det_good_frames_per_sample = {                                                   # 统计每个样本的好帧索引
        sample_i: np.nonzero(~_label.numpy()[sample_i].squeeze())[0].tolist()
        for sample_i in range(len(_label))
    }

    # Break Frame Fix
    inpainting_mask_fixmode = torch.zeros_like(_re_sample).bool().to()               # 修复掩码：True 保留，False 重绘
    for sample_i in range(len(_re_sample)):                                          # 标记好帧为保留
        inpainting_mask_fixmode[sample_i, ..., det_good_frames_per_sample[sample_i]] = True
    inpainting_mask_fixmode[:, -1] = True                                            # 标签通道始终保留

    inpaint_motion_fixmode = rp_model_kwargs_detmode['y']['inpainted_motion'].clone()  # 修复起点
    inpaint_motion_fixmode[:, -1] = -1.0                                               # 标签通道占位
    inpaint_cond_fixmode = (~inpainting_mask_fixmode) & rp_model_kwargs_detmode['attention_mask'].unsqueeze(-2)  # 待预测位置

    rp_model_kwargs = batch_expander(model_kwargs, forward_rp_times)                 # 扩展修复模式条件
    rp_model_kwargs['y']['inpainting_mask'] = inpainting_mask_fixmode.clone()        # 写入修复掩码
    rp_model_kwargs['y']['inpainted_motion'] = inpaint_motion_fixmode.clone()        # 写入修复起点
    rp_model_kwargs['inpaint_cond'] = inpaint_cond_fixmode.clone()                   # 写入需预测位置

    if args.enable_sits:                                                             # 可选软修复步调度
        soft_inpaint_ts = einops.repeat(_re_sample[:, [-1]], 'b c l -> b (repeat c) l', repeat=nfeats)
        soft_inpaint_ts = torch.clip((soft_inpaint_ts + 1 / 2), min=0.0, max=1.0)
        soft_inpaint_ts = torch.ceil((torch.sin(soft_inpaint_ts * torch.pi * 0.5)) * args.diffusion_steps).long()
    else:
        soft_inpaint_ts = None

    sample = sample_fn(
        model,                                                       # 模型
        (bs * forward_rp_times, nfeats, nframes),                    # 采样输出的形状（扩展后的 batch）
        clip_denoised=False,                                         # 不额外裁剪去噪结果
        model_kwargs=rp_model_kwargs,                                # 修复模式的条件（掩码、起点等）
        skip_timesteps=args.skip_timesteps,  # 0 is default          # 可选跳过扩散步
        init_image=rp_model_kwargs['y']['inpainted_motion'],         # 以修复起点作为初始图
        progress=True,                                               # 显示进度
        dump_steps=None,                                             # 不导出中间步
        noise=None,                                                  # 默认随机噪声
        const_noise=False,                                           # 不固定噪声
        soft_inpaint_ts=soft_inpaint_ts,                             # 可选软修复步调度
        cond_fn=cond_fn if args.classifier_scale else None,          # 可选 classifier guidance
    )

    _inpaint_motion_detmode = sample.clone()                         # 拷贝修复结果，用于后续检测模式评估
    _inpaint_motion_detmode[:, -1] = 1.0                             # 将标签通道写为 1.0，占位防止带入原标签
    rp_model_kwargs_detmode['y']['inpainted_motion'] = _inpaint_motion_detmode.clone()  # 更新检测模式的输入内容

    score = 0                                                                       # 初始化累积分数
    with torch.no_grad():                                                           # 用检测模式为候选打分
        _re_t = torch.ones((bs * forward_rp_times,), device=dist_util.dev()) * 49   # 固定时间步 49
        for _ in tqdm(range(eval_times)):                                           # 多次快速检测打分
            x = torch.randn_like(rp_model_kwargs_detmode['y']['inpainted_motion'])  # 随机噪声起点
            inpaint_cond = rp_model_kwargs_detmode['inpaint_cond']                  # 待预测位置
            x_gt = rp_model_kwargs_detmode['y']['inpainted_motion']                 # 已知位置的真值
            x = torch.where(inpaint_cond, x, x_gt)                                  # 只在待预测位置保留噪声
            score += model(x, _re_t, **rp_model_kwargs_detmode)[:, -1]              # 累加标签通道得分
    score /= eval_times                                                             # 求均值，平滑噪声
    score = torch.sum((score > 0.0) * rp_model_kwargs_detmode['attention_mask'], dim=-1)  # 对有效帧求和得分
    score = einops.rearrange(score, "(repeat b) -> repeat b", repeat=forward_rp_times)     # 还原为 [forward_rp_times, bs]

    sample_candidates = einops.rearrange(sample, "(repeat b) c l -> repeat b c l", repeat=forward_rp_times)  # 重排候选 [repeat, bs, C, L]
    selected_id = torch.argmin(score, dim=0)                                       # [bs] 选择分数最低的候选
    selected_id = selected_id[..., None, None].expand(sample_candidates.shape[1:]).unsqueeze(0)  # 展开成索引形状
    sample = torch.gather(sample_candidates, dim=0, index=selected_id).squeeze(0)  # 挑出对应的最佳候选

    return sample
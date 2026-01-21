import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from abc import ABC, abstractmethod


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
    """
    把 [start,end] 区间列表拼成易读的字符串，比如 "[[1,5],[10,12]]"。
    """
    if not intervals:
        return "[]"
    return "[" + ",".join(f"[{s},{e}]" for s, e in intervals) + "]"


# region 算子类
class Operator(ABC):
    """
    算子基类：从 PostEdit 项目移植
    """
    def __init__(self, sigma=0.05):
        self.sigma = sigma

    @abstractmethod
    def __call__(self, x):
        pass

    def measure(self, x): # 测量函数：y = A(x)
        return self(x)

    def error(self, x, y): # 测量误差：\|A(x) - y\|^2  # 注意：这里的 self(x) 会调用 InpaintingOperator 的 __call__，即执行 Mask
        return ((self(x) - y) ** 2).flatten(1).sum(-1)


class InpaintingOperator(Operator):
    """
    动作修补算子：实现 A(x) = x * Mask
    """
    def __init__(self, sigma=0.05):
        super().__init__(sigma)
        self.current_mask = None # 存储当前正在处理的批次的二进制 Mask (1=好帧, 0=坏帧)

    def set_mask(self, mask):
        """
        在优化循环开始前，存入检测到的 Mask。
        mask 形状: [B, 1, N] 或 [B, C, N]
        """
        self.current_mask = mask

    def __call__(self, input_motions_estimate):
        """
        这是 A(x)。在优化过程中，input_motions_estimate 是在不断变化的。
        在优化需要前调用 set_mask 与 get_detection_mask 来设置和获取当前的mask
        """
        if self.current_mask is None:
            raise ValueError("当前 Mask 未设置！请在优化前调用 set_mask。")
        return input_motions_estimate * self.current_mask

    @staticmethod
    def get_detection_mask(
        model,
        diffusion,
        input_motions,
        length,
        attention_mask,
        motion_normalizer,
        args
    ):
        """
        快速前向检测逻辑 (一次性执行)。
        返回: 
          - mask: 二进制掩码 [B, 1, N]，1 表示好帧，0 表示坏帧。
          - y: 测量值 [B, C, N]，即原始动作在好帧处保留，坏帧处置 0。
        """
        bs, nfeats, nframes = input_motions.shape
        device = input_motions.device
        eval_times = 25 

        # 构造检测所需的 kwargs
        model_kwargs_detmode = {
            "y": {
                "inpainting_mask": torch.ones_like(input_motions).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False),
                "inpainted_motion": input_motions.clone().index_fill_(1, torch.tensor([nfeats-1], device=device), 1.0),
            },
            "inpaint_cond": ((~torch.ones_like(input_motions).bool().index_fill_(1, torch.tensor([nfeats-1], device=device), False))
                            & attention_mask.unsqueeze(1)), # 确保与 [B, C, N] 广播一致
            "length": length,
            "attention_mask": attention_mask, # 此时已修正为 [B, N]
        }

        # Debug 打印形状，确保广播正确
        print(f"\n[Debug] get_detection_mask:")
        print(f" - inpainting_mask: {model_kwargs_detmode['y']['inpainting_mask'].shape}")
        print(f" - inpainted_motion: {model_kwargs_detmode['y']['inpainted_motion'].shape}")
        print(f" - attention_mask: {attention_mask.shape}")
        print(f" - inpaint_cond: {model_kwargs_detmode['inpaint_cond'].shape}")
        print(f" - length: {length}")

        with torch.no_grad():
            _re_sample = 0
            # 使用正确的时间步并进行缩放
            t_idx = args.diffusion_steps - 1
            _re_t = torch.ones((bs,), device=device).long() * t_idx
            t_input = _re_t
            if hasattr(diffusion, "_scale_timesteps"):
                t_input = diffusion._scale_timesteps(_re_t)
            
            print(f" - Detection using scaled t: {t_input[0].item()} (from index {t_idx})")

            for _ in tqdm(range(eval_times), desc="Detecting Bad Frames"):
                noise_x = torch.randn_like(model_kwargs_detmode['y']['inpainted_motion'])
                cond = model_kwargs_detmode['inpaint_cond']
                x_gt = model_kwargs_detmode['y']['inpainted_motion']
                x_input = torch.where(cond, noise_x, x_gt)
                _re_sample += model(x_input, t_input, **model_kwargs_detmode)
            _re_sample /= eval_times
            

        # 1. 计算坏帧标签
        _sample_cpu = motion_normalizer.inverse(_re_sample.transpose(1, 2).cpu())
        is_bad = _sample_cpu[..., -1] > args.ProbDetTh 
        
        # 2. 生成二进制掩码 (1=好, 0=坏)
        mask = (~is_bad).to(device).unsqueeze(1).float() 
        
        # 增加 Debug 输出：显示检测到的坏帧区间
        for b in range(bs):
            valid_len = int(length[b].item())
            curr_bad_mask = is_bad[b, :valid_len].cpu().numpy()
            intervals = build_corrupt_intervals(curr_bad_mask)
            print(f" - Sample {b}: {format_intervals(intervals)} (Total bad frames: {curr_bad_mask.sum()}/{valid_len})")

        # 3. 计算测量值 y：使用原始输入 input_motions 而不是重建的 _re_sample
        y = input_motions * mask
        
        return mask, y
# endregion

# region 修复流程封装
@staticmethod
def PostEdit_prev_step_PsampleLoop(
    diffusion,                     # StableMotion diffusion 调度对象
    model,                         # 用于去噪的模型
    shape,                         # 输出张量的形状 (batch, channels, frames)
    noise=None,                    # 可选：固定的初始噪声
    clip_denoised=True,            # 是否将 x_0 预测裁剪到 [-1, 1]
    denoised_fn=None,              # 可选：对 x_0 预测进行额外处理的函数
    cond_fn=None,                  # 可选：classifier guidance 函数
    model_kwargs=None,             # 额外条件（inpainting 掩码、length 等）
    device=None,                   # 采样所用设备
    progress=False,                # 是否显示 tqdm 进度条
    skip_timesteps=0,              # 从倒数第几步开始采样
    init_motions=None,               # 可选：用作 init_image 存在初始图片
    cond_fn_with_grad=False,       # 若 True 则用包含梯度的 cond_fn
    dump_steps=None,               # 需要记录的中间步索引
    const_noise=False,             # 是否使用常量随机噪声
    soft_inpaint_ts: torch.LongTensor=None,  # 软掩码控制每帧的起始步
    use_postedit=False,            # 控制是否开启 PostEdit 优化
    operator=None,                 # InpaintingOperator
    measurement=None,              # 测量值 y
    lgvd=None,                     # LangevinDynamics 对象
    w=1.0,                         # 混合权重
):
    # 记录最终返回的结果
    final = None
    # 如果需要 dump 某些中间步，提前准备存储列表
    dump = [] if dump_steps is not None else None

    # 通过 progressive 版本按时间步迭代采样
    for i, sample in enumerate(
        PostEdit_prev_step_PSampleLoopProgressive(
            diffusion,
            model,
            shape,
            noise=noise,
            clip_denoised=clip_denoised,
            denoised_fn=denoised_fn,
            cond_fn=cond_fn,
            model_kwargs=model_kwargs,
            device=device,
            progress=progress,
            skip_timesteps=skip_timesteps,
            init_motions=init_motions,
            cond_fn_with_grad=cond_fn_with_grad,
            const_noise=const_noise,
            soft_inpaint_ts=soft_inpaint_ts,
            use_postedit=use_postedit,
            operator=operator,
            measurement=measurement,
            lgvd=lgvd,
            w=w,
        )
    ):
        # 可选：保存需要的中间步
        if dump_steps is not None and i in dump_steps:
            dump.append(torch.clone(sample["sample"]))
        # 始终记录当前步最后一次迭代结果
        final = sample

    if dump_steps is not None:
        return dump
    # 返回最后一个时间步的样本
    return final["sample"]

@staticmethod
def PostEdit_prev_step_PSampleLoopProgressive(
    diffusion,                          # StableMotion diffusion 调度对象
    model,                              # 用于去噪的模型
    shape,                              # 输出张量的形状 (batch, channels, frames)
    noise=None,                         # 可选：固定的初始噪声
    clip_denoised=True,                 # 是否将 x_0 预测裁剪到 [-1, 1]
    denoised_fn=None,                   # 可选：对 x_0 预测进行额外处理的函数
    cond_fn=None,                       # 可选：classifier guidance 函数
    model_kwargs=None,                  # 额外条件（inpainting 掩码、length 等）
    device=None,                        # 采样所用设备
    progress=False,                     # 是否显示 tqdm 进度条
    skip_timesteps=0,                   # 从倒数第几步开始采样
    init_motions=None,                  # 可选：用作 init_motions 存在初始动作
    cond_fn_with_grad=False,            # 若 True 则用包含梯度的 cond_fn
    const_noise=False,                  # 是否使用常量随机噪声
    soft_inpaint_ts: torch.LongTensor = None,  # 软掩码控制每帧的起始步
    use_postedit=False,                 # 控制是否开启 PostEdit 优化
    operator=None,                      # InpaintingOperator
    measurement=None,                   # 测量值 y
    lgvd=None,                          # LangevinDynamics 对象
    w=1.0,                              # 混合权重
):
    if device is None:
        device = next(model.parameters()).device
    assert isinstance(shape, (tuple, list))

    # 如果外部传入 noise，则复用；否则随机初始化一个和目标 shape 一致的噪声张量
    if noise is not None:
        motions = noise
    else:
        motions = torch.randn(*shape, device=device)

    # skip_timesteps>0 时表示我们想从某个中间时间步开始，此时没有 init_motions 就填 0 张量占位
    if skip_timesteps and init_motions is None:
        init_motions = torch.zeros_like(motions)

    # 从当前 time step（num_timesteps - skip）开始往前遍历，倒序走完剩余步数
    indices = list(range(diffusion.num_timesteps - skip_timesteps))[::-1]

    # 如果提供了 init_motions，就先把它扩散（q_sample）到迭代起始时间步，作为第一帧输入
    if init_motions is not None:
        my_t = torch.ones([shape[0]], device=device, dtype=torch.long) * indices[0]
        motions = diffusion.q_sample(init_motions, my_t, motions)

    if progress:
        # 只有在需要显示进度条时才导入 tqdm，避免对外部依赖的硬链接
        from tqdm.auto import tqdm
        indices = tqdm(indices)
        
    for i in indices:
        # 构建当前时间步的张量
        t = torch.tensor([i] * shape[0], device=device)

        # 加噪
        if soft_inpaint_ts is not None:
            noise_motions = diffusion.q_sample(init_motions, t)
            # 将需要重新还原的帧替换为 forward diffused（即“加噪”）后的样本
            motions = torch.where(soft_inpaint_ts <= i+1, noise_motions, motions)
            
        # 执行去噪预测
        with torch.no_grad():
            sample_fn = diffusion.p_sample_with_grad if cond_fn_with_grad else diffusion.p_sample
            out = sample_fn(
                model,
                motions,
                t,
                clip_denoised=clip_denoised,
                denoised_fn=denoised_fn,
                cond_fn=cond_fn,
                model_kwargs=model_kwargs,
                const_noise=const_noise,
            )

            # 实现 PostEdit 优化逻辑
            if use_postedit and lgvd is not None and operator is not None and measurement is not None:
                # 1. 获取模型预测的 x0
                pred_x0 = out["pred_xstart"]
                
                # 2. 获取当前步的噪声水平 sigma
                t_orig_idx = i
                if hasattr(diffusion, "timestep_map"):
                    t_orig_idx = diffusion.timestep_map[i]
                sigma = diffusion.sqrt_one_minus_alphas_cumprod[t_orig_idx]
                
                # 3. 郎之万动力学优化
                # 计算当前进度比例 ratio，用于动态调整学习率
                ratio = (diffusion.num_timesteps - i) / diffusion.num_timesteps
                optimized_x0 = lgvd.sample(pred_x0, operator, measurement, sigma, ratio)
                
                # 4. 条件锚定混合 (Conditional Anchor Blending)
                # 在“好帧”位置 (mask=1)，强制将预测结果向原始输入 init_motions 靠拢
                mixed_x0 = torch.where(operator.current_mask.bool(), (1 - w) * optimized_x0 + w * init_motions, optimized_x0)
                
                # 5. 重新加噪得到下一步的采样输入 x_{t-1}
                # 这里我们需要将混合后的 x0 扩散到前一个时间步 t-1
                t_prev = torch.tensor([max(0, i - 1)] * shape[0], device=device)
                t_prev_orig = t_prev
                if hasattr(diffusion, "timestep_map"):
                    t_prev_orig = torch.tensor([diffusion.timestep_map[idx.item()] for idx in t_prev], device=device).long()
                
                motions = diffusion.q_sample(mixed_x0, t_prev_orig)
                
                # 更新返回字典，确保 yield 的是优化后的结果
                out["sample"] = motions
                out["pred_xstart"] = mixed_x0
            else:
                # 常规流程
                motions = out["sample"]

            yield out    
            
# endregion

# region 郎之万动力学类
class LangevinDynamics(nn.Module):
    """
    郎之万动力学（Langevin Dynamics）：用于在潜空间进行采样优化，
    通过结合误差项和正则项来指导潜变量的更新，增强生成结果的保真度。
    """
    def __init__(self, num_steps, lr, tau=0.01, lr_min_ratio=0.01):
        super().__init__()
        self.num_steps = num_steps # 迭代步数
        self.lr = float(lr)        # 学习率
        self.tau = tau             # 噪声项系数
        self.lr_min_ratio = lr_min_ratio

    def sample(self, x0hat, operator, measurement, sigma, ratio, verbose=False, scores=None, steps=None):
        """
        采样优化函数
        x0hat: 初始猜测的 x0 (模型预测输出)
        operator: 前向算子 (InpaintingOperator)
        measurement: 实际观测到的测量值 y
        sigma: 当前去噪过程中的噪声水平 (sqrt(1-alpha_t))
        ratio: 进度比例 (用于动态调整学习率)
        """
        num_steps = self.num_steps if steps is None else steps
        print("根据算子（operator）和测量值（measurement）优化潜变量:")
        pbar = tqdm(range(num_steps), desc="      Langevin Optim", leave=False) if verbose else range(num_steps)
        lr = self.get_lr(ratio)
        x = x0hat.clone().detach().requires_grad_(True)
        optimizer = torch.optim.SGD([x], lr)
        
        # 使用 enable_grad 确保即使在外层是 no_grad 的情况下也能计算梯度
        with torch.enable_grad():
            for _ in pbar:
                optimizer.zero_grad()
                # 损失函数包含两部分：
                # 1. 测量误差项（Data Fidelity）：保证优化后的结果经过算子后与测量值一致
                data_loss = operator.error(x, measurement).sum() / (2 * self.tau ** 2)
                # 2. 正则项（Prior）：保证优化后的结果不偏离初始猜测太远
                # 修复 Bug 2：为正则项权重设置上限，防止 sigma 趋近 0 时爆炸，同时增加梯度裁剪
                reg_weight = 1 / (2 * max(sigma, 1e-3) ** 2)
                reg_loss = ((x - x0hat.detach()) ** 2).sum() * reg_weight
                loss = data_loss + reg_loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_([x], max_norm=5.0) # 限制梯度模长
                optimizer.step()
                # 添加随机扰动 (Langevin term)
                # 修正：将随机噪声与当前 sigma 挂钩，防止在去噪后期引入过大抖动
                with torch.no_grad():
                    epsilon = torch.randn_like(x)
                    x.data = x.data + np.sqrt(2 * lr) * epsilon * sigma # 必须乘以 sigma

                if verbose and _ % 5 == 0:
                    pbar.set_postfix({"Loss": f"{loss.item():.4f}", "Data": f"{data_loss.item():.4f}", "Reg": f"{reg_loss.item():.4f}"})

                if torch.isnan(x).any():
                    print("[Warning] Langevin optimization triggered NaN! Resetting to zeros.")
                    return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio, p=1):
        """动态调整学习率"""
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
# endregion

import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
from abc import ABC, abstractmethod

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
            _re_t = torch.ones((bs,), device=device) * 49
            for _ in tqdm(range(eval_times), desc="Detecting Bad Frames"):
                noise_x = torch.randn_like(model_kwargs_detmode['y']['inpainted_motion'])
                cond = model_kwargs_detmode['inpaint_cond']
                x_gt = model_kwargs_detmode['y']['inpainted_motion']
                x_input = torch.where(cond, noise_x, x_gt)
                _re_sample += model(x_input, _re_t, **model_kwargs_detmode)
            _re_sample /= eval_times

        # 1. 计算坏帧标签
        _sample_cpu = motion_normalizer.inverse(_re_sample.transpose(1, 2).cpu())
        is_bad = _sample_cpu[..., -1] > args.ProbDetTh 
        
        # 2. 生成二进制掩码 (1=好, 0=坏)
        mask = (~is_bad).to(device).unsqueeze(1).float() 
        
        # 增加 Debug 输出：显示检测到的坏帧区间
        for b in range(bs):
            bad_indices = is_bad[b, :length[b]] # 只看有效长度内的
            intervals = []
            start = None
            for idx, flag in enumerate(bad_indices):
                if flag:
                    if start is None:
                        start = idx
                elif start is not None:
                    intervals.append((start + 1, idx)) # 1-based
                    start = None
            if start is not None:
                intervals.append((start + 1, int(length[b])))
            
            interval_str = "[" + ",".join(f"[{s},{e}]" for s, e in intervals) + "]"
            print(f" [动作序列 {b}] 检测到的损坏区间: {interval_str}")

        # 3. 计算测量值 y：使用原始输入 input_motions 而不是重建的 _re_sample
        y = input_motions * mask
        
        return mask, y
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
                loss = operator.error(x, measurement).sum() / (2 * self.tau ** 2)
                # 2. 正则项（Prior）：保证优化后的结果不偏离初始猜测太远
                loss += ((x - x0hat.detach()) ** 2).sum() / (2 * sigma ** 2)
                loss.backward()
                optimizer.step()
                # 添加随机扰动（郎之万项）
                with torch.no_grad():
                    epsilon = torch.randn_like(x)
                    x.data = x.data + np.sqrt(2 * lr) * epsilon

                if torch.isnan(x).any():
                    print("[Warning] Langevin optimization triggered NaN! Resetting to zeros.")
                    return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio, p=1):
        """动态调整学习率"""
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
# endregion

# region 空反向传播类（Null Inversion）：负责动作序列的加噪与反向优化采样。
class NullInversion: 
    def __init__(
        self, 
        diffusion,   # 这里对应 StableMotion 的 SpacedDiffusion 对象
        model,       # 这里对应 StableMotion 的模型主体
        lgvd_config
    ):
        self.diffusion = diffusion
        self.model = model
        self.lgvd = LangevinDynamics(**lgvd_config)

    def get_start(self, ref, starting_timestep=999,noise=None):
        """
        根据输入动作序列 ref 添加指定步数的噪声。
        ref: [B, C, N]
        starting_timestep: 起始时间步 (通常是 0 到 T-1 之间的整数)
        """
        device = ref.device
        # 1. 确保 timestep 是张量且在正确设备上
        t = torch.tensor([starting_timestep] * ref.shape[0], device=device).long()
        # 2. 如果没有提供噪声，则生成随机噪声
        if noise is None:
            noise = torch.randn_like(ref)
        # 3. 调用 StableMotion 的 q_sample 进行前向加噪
        x_start = self.diffusion.q_sample(ref, t, noise=noise)
        
        return x_start
    
    def prev_step(
        self, 
        timestep, 
        sample, 
        operator, 
        measurement, 
        orginal_input_motions,
        model_kwargs,
        annel_interval=5, 
        w = 0.25
    ):
        """
        在采样循环中执行反向步处理，并结合郎之万动力学（Langevin Dynamics）进行迭代优化。
        这是 PostEdit 的核心，用于在去噪过程中强制让生成结果符合测量值（y）。
        """
        # 1. 计算总 de 迭代步数
        num_steps = int((timestep - 1) / annel_interval)
        print(f"反向步数num_steps: {num_steps}")
        print(f"初始时间步timestep: {timestep}")

        # 初始化返回变量，确保在循环未执行时也有默认返回
        final_x0 = sample 

        # 使用进度条显示扩散步的进度
        step_pbar = tqdm(range(num_steps), desc="    Diffusion Steps", leave=False)
        for step in step_pbar:
            # 2. 预测去噪后的原始动作估计 (x0)
            # 通过当前的采样状态 sample 预测出对应的干净动作估计 pred_original_sample
            pred_original_sample = self.sampler_one_step(timestep, sample, model_kwargs)
            # 更新最终结果：记录当前这一步估计
            final_x0 = pred_original_sample

            # 3. 郎之万动力学优化
            # 调用 LGVD 模块，对预测出的 x0 进行优化。
            # 使其既接近模型预测的结果，又能够通过 operator 满足测量值 measurement。
            # 这里的噪声水平 sigma 直接从 diffusion 对象的预计算系数中获取
            sigma = self.diffusion.sqrt_one_minus_alphas_cumprod[timestep]
            pred_x0 = self.lgvd.sample(pred_original_sample, operator, measurement, sigma, step / num_steps, verbose=True)
            
            # 4. 时间步递减
            timestep = max(0, timestep - annel_interval)
            
            # 5. 混合与重新加噪
            # 策略：条件锚定混合 (Conditional Anchor Blending)
            # 在“好帧”位置 (mask=1)，将优化后的结果与原始输入按比例 w 混合，强制锚定到原始运动轨迹，增强保真度。
            # 在“坏帧”位置 (mask=0)，完全采用优化后的预测值 pred_x0，防止原始输入中的损坏数据（噪声或跳变）干扰去噪过程。
            mixed_x0 = torch.where(operator.current_mask.bool(), (1 - w) * pred_x0 + w * orginal_input_motions, pred_x0)
            
            # # 更新最终结果：记录当前这一步优化并混合后的最佳估计
            # final_x0 = mixed_x0

            # 将混合后的最佳动作估计重新加上对应时间步的噪声，得到下一个时间步的采样状态 sample (x_{t-1})
            sample = self.get_start(mixed_x0, timestep)
            
        return final_x0

    def sampler_one_step(
        self, 
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
        
        # 2. 预测噪声
        # 注意：在 StableMotion 中，model 会根据 model_kwargs 处理 inpainting 等条件
        with torch.no_grad():
            # 检查是否有时间步缩放逻辑
            t_input = t
            if hasattr(self.diffusion, "_scale_timesteps"):
                t_input = self.diffusion._scale_timesteps(t)
            
            model_output = self.model(sample, t_input, **model_kwargs)
            
        # 3. 解析模型输出
        # 如果模型输出包含方差信息（如 Learned Sigma），则需要截取前一半作为噪声预测
        # 在 StableMotion 中，通常 C 维度是 [2*original_C]，后半段是方差预测
        if model_output.shape[1] == sample.shape[1] * 2:
            eps, _ = torch.split(model_output, sample.shape[1], dim=1)
        else:
            eps = model_output
            
        # 4. 根据公式从 eps 预测 x_0
        # x_0 = (1/sqrt(alpha_cumprod)) * x_t - (sqrt(1/alpha_cumprod - 1)) * eps
        pred_x0 = self.diffusion._predict_xstart_from_eps(x_t=sample, t=t, eps=eps)

        return pred_x0
    
    def sample_in_batch(self, x_start, operator, y, orginal_input_motions, model_kwargs, starting_timestep=999):
        """批量执行反向步"""
        samples = self.prev_step(starting_timestep, x_start, operator, y, orginal_input_motions, model_kwargs)
        return samples
# endregion
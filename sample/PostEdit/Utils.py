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
                             & attention_mask.unsqueeze(-2)),
            "length": length,
            "attention_mask": attention_mask,
        }

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
                return torch.zeros_like(x)

        return x.detach()

    def get_lr(self, ratio, p=1):
        """动态调整学习率"""
        multiplier = (1 ** (1 / p) + ratio * (self.lr_min_ratio ** (1 / p) - 1 ** (1 / p))) ** p
        return multiplier * self.lr
# endregion
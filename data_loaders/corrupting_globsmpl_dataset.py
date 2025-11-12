import torch
import numpy as np
import os
import codecs as cs
from tqdm import tqdm
from torch.utils.data import Dataset

from data_loaders.smpl_collate import collate_motion
from data_loaders.dataset_utils import foot_slidedetect_zup, motion_artifacts_smpl
from data_loaders.amasstools.globsmplrifke_base_feats import (
    smpldata_to_globsmplrifkefeats,
    ungroup,
    canonicalize_rotation,
)
from smplx.lbs import batch_rigid_transform
import einops
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


class UtilAMASSMotionLoader:
    """
    Utility loader that:
      1) Loads SMPL sequences (.npz).
      2) Injects synthetic artifacts (optional).
      3) Recomputes joints.
      4) Canonicalizes rotation/translation.
      5) Saves corrupted + canonicalized sequences to `save_dir`.
      6) Returns Global SMPL RIFKE features (+ optional label channel).

    Args:
        base_dir (str): Root folder with input sequences (npz).
        fps (int): Motion framerate (unused here; kept for parity).
        disable (bool): If True, skip normalization/label add-ons (not used here).
        ext (str): Extension of input files ('.npz' expected).
        mode (str): 'train' or 'test' (controls artifact ranges).
        artifacts (bool): Enable/disable artifact injection.
        save_dir (str): Output root to store processed npz files.
        **kwargs: Ignored extra args for interface compatibility.
    """

    def __init__(
        self, base_dir, fps=20, disable: bool = False, ext=".npz", mode="train", artifacts=True, save_dir=None, enable_slidedet=True, **kwargs
    ):
        self.fps = fps
        self.base_dir = base_dir
        self.motions = {}
        self.disable = disable
        self.ext = ext
        self.artifacts = artifacts
        self.mode = mode
        self.save_dir = save_dir
        self.enable_slidedet = enable_slidedet  # 脚部滑动检测开关
        assert self.save_dir is not None, "请提供 save_dir 参数以存储损坏后的数据"

        # 加载 SMPL 模型的关节回归器和父子关系
        # J_regressor: 用于从顶点位置回归到关节位置的矩阵
        # parents: 关节的父子关系，用于构建骨骼层次结构
        j_regressor_stat = np.load("data_loaders/amasstools/smpl_neutral_nobetas_24J.npz")
        self.J_regressor = torch.from_numpy(j_regressor_stat["J"]).to(torch.double)  # 关节回归器矩阵
        self.parents = torch.from_numpy(j_regressor_stat["parents"])  # 关节父子关系数组

    def __call__(self, path):
        """
        Process a single sequence by relative path (without extension).
        Returns:
            dict: {"x": Tensor[T, F], "length": T}
        """
        # 如果该路径的数据尚未加载，则从文件加载
        if path not in self.motions:
            # 规范化路径：将路径中的正斜杠转换为系统分隔符
            # 这样可以处理 split 文件中使用 '/' 但系统使用 '\' 的情况
            normalized_path = path.replace('/', os.sep).replace('\\', os.sep)
            motion_path = os.path.join(self.base_dir, normalized_path + self.ext)
            
            # 检查文件是否存在
            if not os.path.exists(motion_path):
                error_msg = f"文件不存在: {motion_path}"
                print(error_msg)
                raise FileNotFoundError(error_msg)
            
            try:
                motion = np.load(motion_path)
                self.motions[path] = motion  # 缓存加载的数据
            except Exception as e:
                error_msg = f"无法加载文件: {motion_path}, 错误: {str(e)}"
                print(error_msg)
                raise RuntimeError(error_msg) from e

        # 确保数据已成功加载
        if path not in self.motions:
            raise KeyError(f"数据未加载: {path}")
        
        motion = self.motions[path]

        if self.ext == ".npz":
            smpl_data = motion
            poses = smpl_data["poses"].copy()   # (T, 66) axis-angle flattened
            trans = smpl_data["trans"].copy()   # (T, 3)

            # Convert axis-angle -> quaternion for artifact injection, then back.
            _poses = einops.rearrange(poses, "k (l t) -> k l t", t=3)  # (T, 22, 3)
            poses_quat = axis_angle_to_quaternion(torch.from_numpy(_poses)).numpy()

            poses_quat, trans, det_mask = motion_artifacts_smpl(poses_quat, trans, self.mode, self.artifacts)
            _poses = quaternion_to_axis_angle(torch.from_numpy(poses_quat))  # (T, 22, 3)
            poses = einops.rearrange(_poses, "k l t -> k (l t)", t=3)
            trans = torch.from_numpy(trans)  # (T, 3)
            det_mask = torch.from_numpy(det_mask)

            # Recompute joints (append two dummy hand joints to match 24).
            rot_mat = axis_angle_to_matrix(_poses)  # (T, 22, 3, 3)
            T = rot_mat.shape[0]
            zero_hands_rot = torch.eye(3)[None, None].expand(T, 2, -1, -1)
            rot_mat = torch.concat((rot_mat, zero_hands_rot), dim=1).to(torch.double)

            joints, _ = batch_rigid_transform(
                rot_mat,
                self.J_regressor[None].expand(T, -1, -1),
                self.parents,
            )
            joints = joints.squeeze() + trans.unsqueeze(1)

            smpl_data = {
                "poses": poses.to(torch.double),
                "trans": trans.to(torch.double),
                "joints": joints.to(torch.double),
            }

            # ========== 步骤 4: 规范化旋转和平移 ==========
            # 将数据转换到规范坐标系（使根节点朝向一致）
            cano_smpl_data = canonicalize_rotation(smpl_data)
            
            # ========== 步骤 5: 检测脚部滑动（可选） ==========
            # 在规范化空间中检测脚部滑动伪影
            if self.enable_slidedet:
                # 检测脚部在 Z 轴（向上）方向的滑动
                slide_label = foot_slidedetect_zup(cano_smpl_data["joints"].clone())
            else:
                # 如果禁用滑动检测，创建零标签
                slide_label = torch.zeros_like(det_mask)

            det_mask = ((det_mask + slide_label.squeeze()) > 0).to(torch.float)
            cano_smpl_data["labels"] = det_mask
            cano_smpl_data = {k: v.numpy() for k, v in cano_smpl_data.items()}

            # Save canonicalized (possibly corrupted) npz.
            motion_path = os.path.join(self.base_dir, path + self.ext)
            save_path = os.path.join(self.save_dir, path + self.ext)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            np.savez(save_path, **cano_smpl_data)

            # Return features (+ 1 label channel).
            motion = smpldata_to_globsmplrifkefeats(smpl_data).to(torch.float)
            motion = torch.cat([motion, det_mask[:, None]], axis=-1)
        else:
            raise NotImplementedError

        return {"x": motion, "length": len(motion)}


def read_split(path, split):
    """
    Read a split file and return the list of ids (one per line).
    """
    split_file = os.path.join(path, "splits", split + ".txt")
    id_list = []
    with cs.open(split_file, "r") as f:
        for line in f.readlines():
            id_list.append(line.strip())
    return id_list


class MotionDataset(Dataset):
    """
    Thin dataset wrapper around a motion loader.
    """

    def __init__(self, motion_loader, split: str = "train", preload: bool = False, skip_missing: bool = True):
        """
        初始化数据集
        
        参数:
            motion_loader: 运动数据加载器实例（UtilAMASSMotionLoader）
            split (str): 数据集分割名称（'train', 'test', 'val' 等）
            preload (bool): 是否在初始化时预加载所有数据到内存（默认 False）
            skip_missing (bool): 是否跳过缺失的文件（默认 True）
        """
        self.collate_fn = collate_motion  # 批处理函数，用于将多个样本组合成批次
        self.split = split  # 数据集分割名称
        all_keyids = read_split("data_loaders", split)  # 读取该分割的所有样本 ID
        self.motion_loader = motion_loader  # 运动数据加载器
        self.is_training = "train" in split  # 是否为训练模式

        # 验证文件是否存在，过滤掉缺失的文件
        if skip_missing:
            valid_keyids = []
            base_dir = motion_loader.base_dir
            ext = motion_loader.ext
            
            print(f"验证 {split} 分割中的文件...")
            for keyid in tqdm(all_keyids, desc="验证文件"):
                file_path = keyid.strip(".npy")
                # 规范化路径
                normalized_path = file_path.replace('/', os.sep).replace('\\', os.sep)
                full_path = os.path.join(base_dir, normalized_path + ext)
                
                if os.path.exists(full_path):
                    valid_keyids.append(keyid)
                else:
                    print(f"警告: 文件不存在，将跳过: {full_path}")
            
            self.keyids = valid_keyids
            skipped_count = len(all_keyids) - len(valid_keyids)
            print(f"验证完成: {len(valid_keyids)}/{len(all_keyids)} 个文件有效")
            if skipped_count > 0:
                print(f"已跳过 {skipped_count} 个缺失的文件")
        else:
            self.keyids = all_keyids

        # 如果启用预加载，遍历所有样本以提前加载到内存
        # 这可以加快训练时的数据加载速度，但会占用更多内存
        if preload:
            for _ in tqdm(self, desc="预加载数据集"):
                continue

    def __len__(self):
        return len(self.keyids)

    def __getitem__(self, index):
        keyid = self.keyids[index]
        return self.load_keyid(keyid)

    def load_keyid(self, keyid):
        """
        根据样本 ID 加载运动数据
        
        参数:
            keyid (str): 样本的唯一标识符
            
        返回:
            dict: 包含以下键的字典
                - "x": Tensor[T, F] - 运动特征，T 为帧数，F 为特征维度
                - "keyid": str - 样本 ID
                - "length": int - 序列长度（帧数）
                
        异常:
            FileNotFoundError: 如果文件不存在
            RuntimeError: 如果文件加载失败
        """
        # 移除可能的 .npy 扩展名，获取文件路径（不含扩展名）
        file_path = keyid.strip(".npy")
        
        try:
            # 使用运动加载器加载数据
            motion_x_dict = self.motion_loader(path=file_path)
            x = motion_x_dict["x"]  # 提取特征
            length = motion_x_dict["length"]  # 提取序列长度
            
            return {"x": x, "keyid": keyid, "length": length}
        except (FileNotFoundError, RuntimeError, KeyError) as e:
            # 重新抛出异常，让调用者知道哪个文件有问题
            raise RuntimeError(f"加载样本失败 {keyid}: {e}") from e


if __name__ == "__main__":
    from utils.fixseed import fixseed
    from torch.utils.data import DataLoader
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument("--mode", type=str)
    args = parser.parse_args()

    fixseed(42)
    mode = args.mode
    enable_slidedet = True  # 启用脚部滑动检测

    motion_loader = UtilAMASSMotionLoader(
        base_dir="dataset/AMASS_20.0_fps_nh_globsmpl_base_cano",
        ext=".npz",
        mode=mode,
        save_dir="dataset/AMASS_20.0_fps_nh_globsmpl_corrupted_cano",
        enable_slidedet=enable_slidedet,  # 传递脚部滑动检测开关
    )
    dataset = MotionDataset(motion_loader=motion_loader, split=mode, preload=False)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=8, drop_last=False, collate_fn=dataset.collate_fn)

    for _ in tqdm(loader):
        pass
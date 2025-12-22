"""
可视化 AMASS_20.0_fps_nh_globsmpl_base_cano 文件夹中的 SMPL 运动序列

功能：
    遍历指定文件夹下的所有 npz 文件，对每个运动序列进行可视化
    支持两种可视化方式：
    1. SMPL mesh 渲染（如果数据包含 vertices 或可以从 poses 计算）
    2. 关节骨架渲染（如果数据包含 joints）

使用方法：
    python visualize/visualize_amass_cano.py --data_path dataset/AMASS_20.0_fps_nh_globsmpl_base_cano --save_path outputs/visualizations
"""
import numpy as np
import os
import torch
from glob import glob
from tqdm import tqdm
from argparse import ArgumentParser

from utils.renderer.humor import HumorRenderer
from utils.renderer.matplotlib import MatplotlibRender
from data_loaders.amasstools.smplh_layer import SMPLH
from data_loaders.amasstools.extract_joints import extract_joints_smpldata


def visualize_npz_file(
    npz_path,
    save_path,
    smplh=None,
    smpl_renderer=None,
    joints_renderer=None,
    rendersmpl=True,
    fps=20.0,
    value_from='smpl',
    auto_fallback=True,
):
    """
    可视化单个 npz 文件
    
    参数:
        npz_path: npz 文件路径
        save_path: 输出视频保存路径
        smplh: SMPLH 模型（用于计算 vertices 和 joints）
        smpl_renderer: SMPL mesh 渲染器
        joints_renderer: 关节骨架渲染器
        rendersmpl: 是否渲染 SMPL mesh（True）还是关节骨架（False）
        fps: 帧率
        value_from: 数据来源模式（'smpl' 或 'joints'）
    """
    # 加载数据
    try:
        data = np.load(npz_path, allow_pickle=True)
    except Exception as e:
        print(f"无法加载文件 {npz_path}: {e}")
        return False
    
    # 检查必要的数据字段
    if 'poses' not in data or 'trans' not in data:
        print(f"文件 {npz_path} 缺少必要字段 (poses 或 trans)")
        return False
    
    poses = data['poses']
    trans = data['trans']
    
    # 确保数据是 numpy 数组
    if isinstance(poses, torch.Tensor):
        poses = poses.numpy()
    if isinstance(trans, torch.Tensor):
        trans = trans.numpy()
    
    # 获取序列长度
    length = len(trans)
    if length < 1:
        print(f"文件 {npz_path} 序列长度为 0")
        return False
    
    # 创建输出目录
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    try:
        if rendersmpl and smpl_renderer is not None:
            # 尝试渲染 SMPL mesh
            try:
                if 'vertices' in data:
                    # 如果数据中已有 vertices，直接使用
                    vertices = data['vertices']
                    if isinstance(vertices, torch.Tensor):
                        vertices = vertices.numpy()
                    vertices = vertices[:length]
                elif smplh is not None:
                    # 从 poses 和 trans 计算 vertices
                    smpldata = {
                        'poses': torch.from_numpy(poses[:length]).float(),
                        'trans': torch.from_numpy(trans[:length]).float(),
                        'joints': None,
                    }
                    output = extract_joints_smpldata(
                        smpldata=smpldata,
                        fps=fps,
                        value_from=value_from,
                        smpl_layer=smplh,
                    )
                    vertices = output['vertices']
                else:
                    raise ValueError("无法渲染 SMPL mesh：缺少 vertices 且未提供 smplh 模型")
                
                # 调整 z 轴，使模型站在地面上（与 render_scripts.py 保持一致）
                vertices[..., 2] -= np.min(vertices[..., 2])
                
                # 渲染
                smpl_renderer(
                    vertices,
                    title="",
                    output=save_path,
                )
            except Exception as smpl_error:
                # SMPL mesh 渲染失败（可能是无头服务器），回退到关节骨架渲染
                error_msg = str(smpl_error)
                if not auto_fallback:
                    # 如果不允许自动回退，直接抛出错误
                    raise
                
                if "NoSuchDisplayException" in error_msg or "Cannot connect" in error_msg:
                    print(f"警告：SMPL mesh 渲染失败（可能是无头服务器），自动回退到关节骨架渲染")
                    print(f"错误信息: {error_msg}")
                else:
                    # 其他错误，也尝试回退
                    print(f"警告：SMPL mesh 渲染失败，自动回退到关节骨架渲染")
                    print(f"错误信息: {error_msg}")
                
                # 更新输出文件名（从 _smpl.mp4 改为 _joints_fallback.mp4）
                if save_path.endswith('_smpl.mp4'):
                    save_path = save_path.replace('_smpl.mp4', '_joints_fallback.mp4')
                elif save_path.endswith('.mp4'):
                    save_path = save_path.replace('.mp4', '_joints_fallback.mp4')
                
                # 回退到关节骨架渲染
                rendersmpl = False
                # 继续执行下面的关节骨架渲染代码
        if not rendersmpl:
            # 渲染关节骨架
            if 'joints' in data:
                # 如果数据中已有 joints，直接使用
                joints = data['joints']
                if isinstance(joints, torch.Tensor):
                    joints = joints.numpy()
                joints = joints[:length]
            elif smplh is not None:
                # 从 poses 和 trans 计算 joints
                smpldata = {
                    'poses': torch.from_numpy(poses[:length]).float(),
                    'trans': torch.from_numpy(trans[:length]).float(),
                    'joints': None,
                }
                output = extract_joints_smpldata(
                    smpldata=smpldata,
                    fps=fps,
                    value_from=value_from,
                    smpl_layer=smplh,
                )
                joints = output['joints']
            else:
                print(f"无法渲染关节骨架：缺少 joints 且未提供 smplh 模型")
                return False
            
            # 渲染
            if joints_renderer is not None:
                joints_renderer(
                    joints,
                    title="",
                    output=save_path,
                    canonicalize=False,
                )
            else:
                print(f"未提供 joints_renderer")
                return False
        
        return True
    except Exception as e:
        print(f"渲染文件 {npz_path} 时出错: {e}")
        import traceback
        traceback.print_exc()
        return False


def visualize_all_npz_files(
    data_path,
    save_path,
    rendersmpl=True,
    fps=20.0,
    smplh_folder="data_loaders/amasstools/deps/smplh",
    value_from='smpl',
    auto_fallback=True,
):
    """
    可视化 data_path 下所有 npz 文件
    
    参数:
        data_path: 输入数据文件夹路径
        save_path: 输出视频保存文件夹路径
        rendersmpl: 是否渲染 SMPL mesh（True）还是关节骨架（False）
        fps: 帧率
        smplh_folder: SMPL-H 模型文件夹路径
        value_from: 数据来源模式（'smpl' 或 'joints'）
    """
    # 检查输入路径
    if not os.path.exists(data_path):
        print(f"错误：数据路径不存在: {data_path}")
        return
    
    # 创建输出目录
    os.makedirs(save_path, exist_ok=True)
    
    # 查找所有 npz 文件
    match_str = os.path.join(data_path, "**/*.npz")
    npz_files = glob(match_str, recursive=True)
    
    if len(npz_files) == 0:
        print(f"在 {data_path} 中未找到任何 npz 文件")
        return
    
    print(f"找到 {len(npz_files)} 个 npz 文件")
    
    # 初始化渲染器
    smpl_renderer = None
    joints_renderer = None
    smplh = None
    
    # 初始化渲染器（如果 auto_fallback 为 True，需要同时初始化两种渲染器）
    smpl_renderer = None
    joints_renderer = None
    
    if rendersmpl:
        # 初始化 SMPL mesh 渲染器
        smpl_renderer = HumorRenderer(
            fps=fps,
            imw=224,
            imh=224,
            cam_offset=[0.0, -2.2, 0.9],
            cam_rot=[
                [1.0000000, 0.0000000, 0.0000000],
                [0.0000000, 0.0000000, -1.0000000],
                [0.0000000, 1.0000000, 0.0000000],
            ],
        )
    
    # 如果 auto_fallback 为 True 或需要渲染关节骨架，初始化关节骨架渲染器
    if auto_fallback or not rendersmpl:
        joints_renderer = MatplotlibRender(
            jointstype='smpljoints',
            fps=fps,
            colors=["black", "magenta", "red", "green", "blue"],
            figsize=4,
            canonicalize=True,
        )
    
    # 初始化 SMPLH 模型
    if os.path.exists(smplh_folder):
        # 根据渲染模式选择 jointstype
        if rendersmpl and auto_fallback:
            jointstype = 'both'  # 需要同时支持 vertices 和 joints
        elif rendersmpl:
            jointstype = 'both'
        else:
            jointstype = 'smpljoints'
        
        smplh = SMPLH(
            path=smplh_folder,
            jointstype=jointstype,
            input_pose_rep="axisangle",
            gender="neutral",
        )
        print("已加载 SMPLH 模型")
    else:
        print(f"警告：SMPLH 模型文件夹不存在: {smplh_folder}")
        print("将尝试使用数据中已有的 vertices/joints")
        smplh = None
    
    # 处理每个文件
    success_count = 0
    fail_count = 0
    
    for npz_file in tqdm(npz_files, desc="处理文件"):
        # 计算相对路径
        rel_path = os.path.relpath(npz_file, data_path)
        
        # 构建输出路径（保持相对路径结构）
        base_name = os.path.splitext(rel_path)[0]
        if rendersmpl:
            output_file = os.path.join(save_path, f"{base_name}_smpl.mp4")
        else:
            output_file = os.path.join(save_path, f"{base_name}_joints.mp4")
        
        # 可视化
        # 如果自动回退，需要确保 joints_renderer 可用
        current_rendersmpl = rendersmpl
        current_joints_renderer = joints_renderer if (auto_fallback or not rendersmpl) else None
        
        if visualize_npz_file(
            npz_file,
            output_file,
            smplh=smplh,
            smpl_renderer=smpl_renderer,
            joints_renderer=current_joints_renderer,
            rendersmpl=current_rendersmpl,
            fps=fps,
            value_from=value_from,
            auto_fallback=auto_fallback,
        ):
            success_count += 1
        else:
            fail_count += 1
    
    print(f"\n处理完成！")
    print(f"成功: {success_count} 个文件")
    print(f"失败: {fail_count} 个文件")
    print(f"输出目录: {save_path}")


if __name__ == '__main__':
    parser = ArgumentParser(description="可视化 AMASS canonicalized 数据集中的所有 npz 文件")
    parser.add_argument("--data_path", type=str, required=True,
                        help="输入数据文件夹路径（包含 npz 文件）")
    parser.add_argument("--save_path", type=str, required=True,
                        help="输出视频保存文件夹路径")
    parser.add_argument("--rendersmpl", action='store_true',
                        help="渲染 SMPL mesh（默认渲染关节骨架）")
    parser.add_argument("--fps", type=float, default=20.0,
                        help="帧率（默认 20.0）")
    parser.add_argument("--smplh_folder", type=str,
                        default="data_loaders/amasstools/deps/smplh",
                        help="SMPL-H 模型文件夹路径")
    parser.add_argument("--value_from", type=str, default='smpl',
                        choices=['smpl', 'joints'],
                        help="数据来源模式：'smpl' 从模型计算，'joints' 直接使用数据中的 joints")
    parser.add_argument("--auto_fallback", action='store_true', default=True,
                        help="当 SMPL mesh 渲染失败时自动回退到关节骨架渲染（默认启用）")
    parser.add_argument("--no_auto_fallback", dest='auto_fallback', action='store_false',
                        help="禁用自动回退功能")
    
    args = parser.parse_args()
    
    visualize_all_npz_files(
        data_path=args.data_path,
        save_path=args.save_path,
        rendersmpl=args.rendersmpl,
        fps=args.fps,
        smplh_folder=args.smplh_folder,
        value_from=args.value_from,
        auto_fallback=args.auto_fallback,
    )


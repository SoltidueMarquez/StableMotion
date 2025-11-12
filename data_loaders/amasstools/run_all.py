#!/usr/bin/env python
"""
依次运行所有数据处理模块

该脚本会按顺序运行以下数据处理步骤：
1. fix_fps: 将运动数据统一到 20.0 fps，移除手部参数
2. smpl_mirroring: 创建镜像数据用于数据增强（保存在 M/ 目录）
3. extract_joints: 从 SMPL 姿态参数提取关节位置
4. get_globsmplrifke_base: 生成规范化后的全局 SMPL RIFKE 特征

注意：
- 每个步骤都会处理原始数据和镜像数据（M/ 目录）
- 如果某个步骤失败，脚本会停止并报告错误
- 建议在运行前检查 CUDA 是否可用，如果不可用，设置环境变量 FORCE_CPU=1
"""
import subprocess
import sys
import os

modules = [
    {
        "name": "data_loaders.amasstools.fix_fps",
        "description": "统一帧率到 20.0 fps，移除手部参数"
    },
    {
        "name": "data_loaders.amasstools.smpl_mirroring",
        "description": "创建镜像数据用于数据增强（保存在 M/ 目录）"
    },
    {
        "name": "data_loaders.amasstools.extract_joints",
        "description": "从 SMPL 姿态参数提取关节位置（包括 M/ 目录下的镜像数据）"
    },
    {
        "name": "data_loaders.amasstools.get_globsmplrifke_base",
        "description": "生成规范化后的全局 SMPL RIFKE 特征（包括 M/ 目录下的镜像数据）"
    },
]

def run_module(module_info):
    """
    运行指定的模块
    
    参数:
        module_info: 包含模块名称和描述的字典
        
    返回:
        bool: 是否成功执行
    """
    module_name = module_info["name"]
    description = module_info.get("description", "")
    
    print(f"\n{'='*60}")
    print(f"正在运行: {module_name}")
    if description:
        print(f"描述: {description}")
    print(f"{'='*60}\n")
    
    # 运行模块
    result = subprocess.run(
        [sys.executable, "-m", module_name],
        cwd=None,
        capture_output=False  # 显示实时输出
    )
    
    if result.returncode != 0:
        print(f"\n{'='*60}")
        print(f"错误: {module_name} 执行失败")
        print(f"退出码: {result.returncode}")
        print(f"{'='*60}\n")
        return False
    
    print(f"\n✓ {module_name} 执行成功\n")
    return True

if __name__ == "__main__":
    print("="*60)
    print("AMASS 数据处理流程")
    print("="*60)
    print("\n该脚本将依次运行以下步骤：")
    for i, module in enumerate(modules, 1):
        print(f"  {i}. {module['name']}")
        if module.get("description"):
            print(f"     {module['description']}")
    print("\n提示：")
    print("  - 如果遇到 CUDA 错误，可以设置环境变量 FORCE_CPU=1 强制使用 CPU")
    print("  - 处理过程可能需要较长时间，请耐心等待")
    print("  - 每个步骤都会处理原始数据和镜像数据（M/ 目录）")
    print("="*60)
    
    # 检查是否强制使用 CPU
    if os.environ.get("FORCE_CPU", "").lower() in ("1", "true", "yes"):
        print("\n注意: 检测到 FORCE_CPU=1，将强制使用 CPU 处理")
    
    input("\n按 Enter 键开始处理，或按 Ctrl+C 取消...")
    
    print("\n开始运行所有数据处理模块...\n")
    
    for i, module in enumerate(modules, 1):
        print(f"\n[{i}/{len(modules)}] 处理步骤 {i}")
        if not run_module(module):
            print(f"\n{'='*60}")
            print(f"处理中断: {module['name']} 执行失败")
            print(f"请检查错误信息并修复后重新运行")
            print(f"{'='*60}\n")
            sys.exit(1)
    
    print(f"\n{'='*60}")
    print("✓ 所有模块执行完成！")
    print(f"{'='*60}\n")
    print("数据处理流程已完成。现在可以运行：")
    print("  python -m data_loaders.corrupting_globsmpl_dataset --mode train")
    print("  python -m data_loaders.corrupting_globsmpl_dataset --mode test")
    print()


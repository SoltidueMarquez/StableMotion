#!/usr/bin/env python
"""
检查 PyTorch 和 CUDA 兼容性

该脚本会检查：
1. PyTorch 版本
2. CUDA 是否可用
3. GPU 计算能力是否被支持
4. 提供升级建议
"""
import sys

try:
    import torch
    print("="*60)
    print("PyTorch 环境检查")
    print("="*60)
    
    # 基本信息
    print(f"\nPyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"cuDNN 版本: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
        print(f"\n检测到的 GPU:")
        for i in range(torch.cuda.device_count()):
            print(f"  [{i}] {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"      计算能力: {props.major}.{props.minor} (sm_{props.major}{props.minor})")
        
        # 测试 GPU 是否真的可用
        print(f"\n测试 GPU 可用性...")
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            print("✓ GPU 测试成功！可以正常使用 CUDA")
        except RuntimeError as e:
            print(f"✗ GPU 测试失败: {e}")
            print("\n可能的原因：")
            print("  1. PyTorch 版本不支持当前 GPU 的计算能力")
            print("  2. CUDA 驱动版本不兼容")
            print("\n解决方案：")
            print("  1. 升级 PyTorch 到最新版本（支持 sm_120）")
            print("  2. 或设置环境变量 FORCE_CPU=1 使用 CPU")
    else:
        print("\nCUDA 不可用，将使用 CPU")
        print("如果您的系统有 GPU，可能需要：")
        print("  1. 安装 CUDA 驱动")
        print("  2. 安装支持 CUDA 的 PyTorch 版本")
    
    print("\n" + "="*60)
    print("建议：")
    print("="*60)
    
    if torch.cuda.is_available():
        try:
            test_tensor = torch.zeros(1).cuda()
            _ = test_tensor + 1
            print("✓ 当前 PyTorch 版本支持您的 GPU，可以正常使用")
        except RuntimeError:
            print("✗ 当前 PyTorch 版本不支持您的 GPU")
            print("\n升级 PyTorch 命令：")
            print("  pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124")
            print("\n或者使用 CPU 模式（设置环境变量）：")
            print("  Windows PowerShell: $env:FORCE_CPU=1")
            print("  Windows CMD: set FORCE_CPU=1")
    else:
        print("当前使用 CPU 模式")
        if "+cpu" in torch.__version__:
            print("\n⚠️  检测到您安装的是 CPU 版本的 PyTorch")
            print("需要安装支持 CUDA 的版本才能使用 GPU")
            print("\n安装 CUDA 版本的 PyTorch（支持 RTX 5070 Ti）:")
            print("  1. 卸载 CPU 版本:")
            print("     pip uninstall torch torchvision torchaudio")
            print("  2. 安装 CUDA 版本（与原命令风格一致）:")
            print("     pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124")
            print("\n或者使用 CPU 模式（设置环境变量）:")
            print("  Windows PowerShell: $env:FORCE_CPU=1")
            print("  Windows CMD: set FORCE_CPU=1")
        else:
            print("如需使用 GPU，请安装支持 CUDA 的 PyTorch")
    
    print("="*60)
    
except ImportError:
    print("错误: 未安装 PyTorch")
    print("请运行: pip install torch")
    sys.exit(1)


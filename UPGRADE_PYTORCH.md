# PyTorch 升级指南 - 支持 RTX 5070 Ti (sm_120)

## 问题说明

RTX 5070 Ti GPU 的计算能力是 sm_120，当前安装的 PyTorch 2.4.1 不支持此计算能力。

## 解决方案

### 方案 1：升级到 PyTorch 2.5+（推荐，与原命令风格一致）

**保持与原安装命令风格一致，使用 `--extra-index-url`：**

```bash
# 方法 1：安装最新稳定版本（推荐）
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124

# 方法 2：指定最低版本（确保支持 sm_120）
pip install "torch>=2.5.0" "torchvision>=0.20.0" "torchaudio>=2.5.0" --extra-index-url https://download.pytorch.org/whl/cu124
```

**与原命令的对比：**
- 原命令：`pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124`
- 新命令：`pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124`

**区别说明：**
- ✅ 保持使用 `--extra-index-url`（添加额外索引，不替换默认索引）
- ✅ 不指定具体版本号，安装最新稳定版本（支持 sm_120）
- ✅ 添加了 `torchaudio`（原命令中没有，但建议安装）

### 方案 2：使用 CPU 模式（临时方案）

如果暂时不想升级 PyTorch，可以强制使用 CPU：

#### Windows PowerShell:
```powershell
$env:FORCE_CPU=1
python -m data_loaders.amasstools.run_all
```

#### Windows CMD:
```cmd
set FORCE_CPU=1
python -m data_loaders.amasstools.run_all
```

## 升级步骤

### 步骤 1：卸载旧版本（可选，但推荐）
```bash
pip uninstall torch torchvision torchaudio
```

### 步骤 2：安装新版本
```bash
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu124
```

### 步骤 3：验证安装
运行检查脚本：
```bash
python check_pytorch.py
```

或者手动验证：
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    # 测试 GPU
    x = torch.zeros(1).cuda()
    print("✓ GPU 测试成功！")
```

## 注意事项

1. **保持索引URL风格一致**：使用 `--extra-index-url` 而不是 `--index-url`
   - `--extra-index-url`：添加额外索引，保留默认 PyPI 索引
   - `--index-url`：替换默认索引，只使用指定的索引

2. **版本兼容性**：
   - PyTorch 2.5+ 支持 sm_120（RTX 5070 Ti）
   - 确保 CUDA 驱动版本 >= 12.4

3. **依赖兼容性**：
   - 升级后可能需要重新安装其他依赖（如 smplx）
   - 检查其他包是否与 PyTorch 2.5+ 兼容

4. **如果升级后出现问题**：
   - 可以回退到原版本：`pip install torch==2.4.1+cu124 torchvision==0.19.1+cu124 --extra-index-url https://download.pytorch.org/whl/cu124`
   - 然后使用 CPU 模式：设置 `FORCE_CPU=1`

## 推荐方案

**建议使用方案 1（升级 PyTorch）**，因为：
1. RTX 5070 Ti 是高性能 GPU，使用 GPU 可以大幅加速处理
2. PyTorch 2.5+ 已经稳定，支持 sm_120
3. 数据处理任务（特别是 `extract_joints`）在 GPU 上会快很多
4. 命令风格与原安装命令一致，不会影响其他依赖

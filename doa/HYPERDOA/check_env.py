"""
环境检查脚本 - 检查HYPERDOA所需依赖
在Windows conda环境中运行: conda activate isac && python check_env.py
"""

import sys

def check_conda_env():
    """检查conda环境"""
    print("=" * 50)
    print("1. Conda 环境信息")
    print("=" * 50)
    import platform
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print(f"系统平台: {platform.platform()}")
    print()

def check_packages():
    """检查必需的包"""
    print("=" * 50)
    print("2. 依赖包检查")
    print("=" * 50)

    packages = {
        "torch": "PyTorch (>=2.0.0)",
        "torchhd": "torch-hd (>=0.6.0) [关键HDC库]",
        "numpy": "NumPy (>=1.24.0)",
        "matplotlib": "Matplotlib (>=3.7.0)",
    }

    results = {}
    for pkg, desc in packages.items():
        try:
            if pkg == "torchhd":
                mod = __import__("torchhd")
                ver = getattr(mod, "__version__", "unknown")
            elif pkg == "torch":
                import torch
                ver = torch.__version__
            else:
                mod = __import__(pkg)
                ver = getattr(mod, "__version__", "unknown")

            # 检查torch-hd特殊模块
            if pkg == "torchhd":
                try:
                    import torchhd as hd
                    if hasattr(hd, 'embeddings'):
                        print(f"✓ {desc}: v{ver}")
                    else:
                        print(f"✗ {desc}: v{ver} (缺少embeddings子模块)")
                except Exception as e:
                    print(f"✗ {desc}: v{ver} (导入错误: {e})")
            else:
                print(f"✓ {desc}: v{ver}")
            results[pkg] = True

        except ImportError:
            print(f"✗ {desc}: 未安装")
            results[pkg] = False

    print()
    return results

def check_cuda():
    """检查CUDA支持"""
    print("=" * 50)
    print("3. CUDA/GPU支持")
    print("=" * 50)

    try:
        import torch
        if torch.cuda.is_available():
            print(f"✓ CUDA可用")
            print(f"  GPU数量: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"  CUDA版本: {torch.version.cuda}")
        else:
            print("✗ CUDA不可用 (将使用CPU运行)")
    except:
        print("✗ 无法检查CUDA状态")

    print()

def check_gpu_compute():
    """检查GPU计算能力"""
    print("=" * 50)
    print("4. GPU信息 (nvidia-smi)")
    print("=" * 50)

    import subprocess
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader'],
                              capture_output=True, text=True)
        if result.returncode == 0:
            print(result.stdout)
        else:
            print("nvidia-smi 不可用或无NVIDIA GPU")
    except FileNotFoundError:
        print("nvidia-smi 未找到")
    except Exception as e:
        print(f"nvidia-smi 错误: {e}")

    print()

def install_missing(results):
    """安装缺失的包"""
    print("=" * 50)
    print("5. 安装指南")
    print("=" * 50)

    if not results["torch"]:
        print("PyTorch未安装:")
        print("  conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia")
        print("  或 pip install torch")

    if not results["torchhd"]:
        print("torch-hd未安装 (关键!):")
        print("  pip install torch-hd")
        print("  或 conda install torch-hd -c conda-forge")

    if not results["numpy"]:
        print("NumPy未安装:")
        print("  conda install numpy")
        print("  或 pip install numpy")

    if not results["matplotlib"]:
        print("Matplotlib未安装:")
        print("  conda install matplotlib")
        print("  或 pip install matplotlib")

    print()

def main():
    print("\n" + "=" * 50)
    print("    HYPERDOA 环境检查脚本")
    print("=" * 50 + "\n")

    check_conda_env()
    results = check_packages()
    check_cuda()
    check_gpu_compute()

    if not all(results.values()):
        print("\n" + "=" * 50)
        print("    缺少依赖 - 请安装后再试")
        print("=" * 50 + "\n")
        install_missing(results)
        return 1
    else:
        print("\n" + "=" * 50)
        print("    ✓ 所有依赖已就绪!")
        print("=" * 50 + "\n")
        return 0

if __name__ == "__main__":
    sys.exit(main())

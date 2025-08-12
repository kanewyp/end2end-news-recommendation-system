import torch
import sys
import platform
import os

print(f"System: {platform.system()} {platform.release()}")
print(f"Python: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check if we're in a virtual environment or container
print(f"Virtual environment: {os.environ.get('VIRTUAL_ENV', 'None')}")
print(f"Conda environment: {os.environ.get('CONDA_DEFAULT_ENV', 'None')}")

print("\n=== DEVICE AVAILABILITY CHECK ===")

# Check CUDA (NVIDIA GPU)
print("Checking CUDA...")
if torch.cuda.is_available():
    print(f"✓ CUDA available: {torch.cuda.device_count()} device(s)")
    for i in range(torch.cuda.device_count()):
        print(f"  - GPU {i}: {torch.cuda.get_device_name(i)}")
    try:
        cuda_device = torch.device('cuda')
        test_tensor = torch.randn(2, 2, device=cuda_device)
        print(f"✓ CUDA test successful: {test_tensor.device}")
    except Exception as e:
        print(f"✗ CUDA test failed: {e}")
else:
    print("✗ CUDA not available")

# Check Intel XPU
print("\nChecking Intel XPU...")
try:
    import intel_extension_for_pytorch as ipex
    print(f"✓ Intel Extension imported, version: {ipex.__version__}")
    
    print(f"  - torch has 'xpu' attribute: {hasattr(torch, 'xpu')}")
    
    if hasattr(torch, 'xpu'):
        try:
            xpu_available = torch.xpu.is_available()
            print(f"  - XPU available: {xpu_available}")
            
            if xpu_available:
                device_count = torch.xpu.device_count()
                print(f"  - XPU device count: {device_count}")
                
                xpu_device = torch.device('xpu')
                test_tensor = torch.randn(2, 2, device=xpu_device)
                print(f"✓ XPU test successful: {test_tensor.device}")
            else:
                print("✗ XPU not available on this system")
                
        except Exception as e:
            print(f"✗ XPU error: {e}")
    else:
        print("✗ torch.xpu not found")
        
except ImportError as e:
    print(f"✗ Intel Extension import failed: {e}")
except Exception as e:
    print(f"✗ Intel Extension error: {e}")

# Check CPU (always available)
print("\nChecking CPU...")
try:
    cpu_device = torch.device('cpu')
    test_tensor = torch.randn(2, 2, device=cpu_device)
    print(f"✓ CPU test successful: {test_tensor.device}")
except Exception as e:
    print(f"✗ CPU test failed: {e}")

# Check environment variables that might affect GPU detection
print("\n=== ENVIRONMENT VARIABLES ===")
gpu_env_vars = [
    'CUDA_VISIBLE_DEVICES', 
    'CUDA_DEVICE_ORDER',
    'INTEL_DEVICE_ORDER',
    'SYCL_DEVICE_SELECTOR',
    'ONEAPI_ROOT'
]

for var in gpu_env_vars:
    value = os.environ.get(var, 'Not set')
    print(f"{var}: {value}")

# Final recommendation
print("\n=== RECOMMENDATION ===")
if torch.cuda.is_available():
    print("🎯 Use CUDA (NVIDIA GPU) - fastest option")
    print("   device = torch.device('cuda')")
elif hasattr(torch, 'xpu') and torch.xpu.is_available():
    print("🎯 Use Intel XPU")
    print("   device = torch.device('xpu')")
else:
    print("🎯 Use CPU - reliable and works for your project size")
    print("   device = torch.device('cpu')")
    print("   Your CNN model will train fine on CPU, just slower")
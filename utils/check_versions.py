import torch

# Check PyTorch version
print(f"PyTorch version: {torch.__version__}")

# Check CUDA version
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available.")

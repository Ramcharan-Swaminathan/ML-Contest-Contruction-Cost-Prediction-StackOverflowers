import torch
import sys

print(f"PyTorch Version: {torch.__version__}")
try:
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"Device Count: {torch.cuda.device_count()}")
        print(f"Current Device: {torch.cuda.current_device()}")
        print(f"Device Name: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA is NOT available.")
except Exception as e:
    print(f"Error checking CUDA: {e}")

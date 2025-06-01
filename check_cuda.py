import torch
import whisper

def check_cuda():
    print("=== CUDA 检查 ===")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA 版本: {torch.version.cuda}")
        print(f"当前GPU设备: {torch.cuda.get_device_name(0)}")
        print(f"GPU数量: {torch.cuda.device_count()}")
    
    print("\n=== Whisper 模型信息 ===")
    model = whisper.load_model("tiny")
    print(f"模型设备: {next(model.parameters()).device}")
    
    print("\n=== PyTorch 信息 ===")
    print(f"PyTorch 版本: {torch.__version__}")
    print(f"CUDA 是否可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"当前CUDA设备: {torch.cuda.current_device()}")
        print(f"CUDA设备名称: {torch.cuda.get_device_name()}")

if __name__ == "__main__":
    check_cuda() 
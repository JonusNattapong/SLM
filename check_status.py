import os
import json
import time
from datetime import datetime
import torch


def check_training_progress():
    """Check current training progress and system status"""
    print("=" * 60)
    print("🔍 THAI SLM TRAINING STATUS CHECK")
    print("=" * 60)
    
    # Check system info
    print(f"\n📊 System Information:")
    print(f"   PyTorch Version: {torch.__version__}")
    print(f"   CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   GPU Memory: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB")
    else:
        print("   Device: CPU (Training will be slower)")
    
    # Check directories
    print(f"\n📁 Directory Status:")
    directories = ["data", "checkpoints", "logs", "outputs", "./thai_slm_moe_model"]
    for directory in directories:
        exists = "✅" if os.path.exists(directory) else "❌"
        print(f"   {exists} {directory}")
    
    # Check model files
    print(f"\n🤖 Model Status:")
    model_dir = "./thai_slm_moe_model"
    if os.path.exists(model_dir):
        model_files = ["pytorch_model.bin", "config.json", "tokenizer.json"]
        for file in model_files:
            file_path = os.path.join(model_dir, file)
            if os.path.exists(file_path):
                size = os.path.getsize(file_path) // (1024 * 1024)  # MB
                print(f"   ✅ {file} ({size} MB)")
            else:
                print(f"   ❌ {file}")
    else:
        print("   ❌ Model not found - run training first")
    
    # Check for checkpoints
    print(f"\n💾 Training Checkpoints:")
    if os.path.exists(model_dir):
        checkpoints = [d for d in os.listdir(model_dir) if d.startswith("checkpoint-")]
        if checkpoints:
            print(f"   Found {len(checkpoints)} checkpoints:")
            for checkpoint in sorted(checkpoints)[-3:]:  # Show last 3
                checkpoint_path = os.path.join(model_dir, checkpoint)
                if os.path.isdir(checkpoint_path):
                    mtime = os.path.getmtime(checkpoint_path)
                    time_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
                    print(f"   ✅ {checkpoint} (saved: {time_str})")
        else:
            print("   ❌ No checkpoints found")
    
    # Check training data
    print(f"\n📄 Training Data:")
    if os.path.exists("training_data.pt"):
        size = os.path.getsize("training_data.pt") // (1024 * 1024)  # MB
        print(f"   ✅ training_data.pt ({size} MB)")
        
        # Try to load and check data
        try:
            data = torch.load("training_data.pt", map_location="cpu")
            print(f"   📊 Training examples: {len(data)}")
        except:
            print("   ⚠️ Could not load training data details")
    else:
        print("   ❌ training_data.pt not found")
    
    # Check tokenizer
    print(f"\n🔤 Tokenizer Status:")
    tokenizer_dir = "./thai_tokenizer"
    if os.path.exists(tokenizer_dir):
        tokenizer_files = ["tokenizer.json", "tokenizer_config.json"]
        for file in tokenizer_files:
            file_path = os.path.join(tokenizer_dir, file)
            exists = "✅" if os.path.exists(file_path) else "❌"
            print(f"   {exists} {file}")
    else:
        print("   ❌ Tokenizer directory not found")
    
    # Recommendations
    print(f"\n💡 Recommendations:")
    
    if not torch.cuda.is_available():
        print("   🐌 CPU Training Detected:")
        print("      - Training will be slower but functional")
        print("      - Consider using Google Colab for GPU training")
        print("      - Reduce model size in config if memory issues occur")
    
    if not os.path.exists(model_dir):
        print("   🚀 Ready to Start Training:")
        print("      - Run: python train.py")
        print("      - Or: run_training.bat")
    elif os.path.exists(os.path.join(model_dir, "pytorch_model.bin")):
        print("   ✅ Model Trained Successfully:")
        print("      - Test: python inference.py")
        print("      - Web UI: python gradio_app.py")
        print("      - Evaluate: python evaluate.py")
    else:
        print("   ⚠️ Training In Progress or Incomplete:")
        print("      - Check logs for training status")
        print("      - Resume training if interrupted")
    
    print("=" * 60)


def monitor_training(interval=30):
    """Monitor training progress in real-time"""
    print("🔄 Starting training monitor (Ctrl+C to stop)")
    print(f"Checking every {interval} seconds...\n")
    
    try:
        while True:
            check_training_progress()
            print(f"\n⏰ Next check in {interval} seconds...")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n👋 Monitoring stopped by user")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "monitor":
        monitor_training()
    else:
        check_training_progress()

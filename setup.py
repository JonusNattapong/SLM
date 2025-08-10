import subprocess
import sys
import os
import platform


def install_requirements():
    """Install required packages"""
    print("Installing required packages...")
    
    requirements = [
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.12.0",
        "tokenizers>=0.13.0",
        "huggingface-hub>=0.15.0",
        "wandb>=0.15.0",
        "accelerate>=0.20.0",
        "evaluate>=0.4.0",
        "sentencepiece>=0.1.99",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "tqdm>=4.65.0",
        "scikit-learn>=1.3.0",
        "gradio>=3.40.0"
    ]
    
    for package in requirements:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to install {package}: {e}")


def setup_wandb():
    """Setup Weights & Biases"""
    print("\nSetting up Weights & Biases...")
    try:
        import wandb
        print("✅ WandB is available")
        print("Please run 'wandb login' to authenticate if you haven't already")
    except ImportError:
        print("❌ WandB not installed")


def check_gpu():
    """Check GPU availability"""
    print("\nChecking GPU availability...")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"✅ GPU available: {gpu_name} (Count: {gpu_count})")
            print(f"CUDA Version: {torch.version.cuda}")
        else:
            print("⚠️ No GPU available - training will use CPU (much slower)")
    except ImportError:
        print("❌ PyTorch not installed")


def create_directories():
    """Create necessary directories"""
    print("\nCreating directories...")
    
    directories = [
        "data",
        "checkpoints",
        "logs",
        "outputs"
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✅ Created directory: {directory}")


def check_dataset_access():
    """Check access to the dataset"""
    print("\nChecking dataset access...")
    try:
        from datasets import load_dataset
        # Try to load a small sample to test connectivity
        print("Testing dataset access...")
        dataset = load_dataset("ZombitX64/Wikipedia-Thai", split="train", streaming=True)
        sample = next(iter(dataset))
        print("✅ Dataset access successful")
        print(f"Sample data keys: {list(sample.keys())}")
    except Exception as e:
        print(f"❌ Dataset access failed: {e}")
        print("Please check your internet connection and dataset availability")


def print_next_steps():
    """Print next steps for the user"""
    print("\n" + "="*60)
    print("🎉 Setup Complete! Next Steps:")
    print("="*60)
    
    print("\n1. 📊 Login to Weights & Biases (optional):")
    print("   wandb login")
    
    print("\n2. 🏋️ Start Training:")
    if platform.system() == "Windows":
        print("   run_training.bat")
    else:
        print("   chmod +x run_training.sh")
        print("   ./run_training.sh")
    
    print("\n3. 🧪 Test the Model:")
    print("   python inference.py")
    
    print("\n4. 🌐 Launch Web Interface:")
    print("   python gradio_app.py")
    
    print("\n5. 📤 Upload to Hugging Face:")
    print("   export HF_TOKEN='your_token'  # Get from https://huggingface.co/settings/tokens")
    print("   python upload_to_hf.py")
    
    print("\n📚 For detailed instructions, see README.md")
    print("🐛 For issues, check the troubleshooting section in README.md")


def main():
    """Main setup function"""
    print("🚀 Thai SLM MoE Setup")
    print("="*40)
    
    # Install requirements
    install_requirements()
    
    # Setup wandb
    setup_wandb()
    
    # Check GPU
    check_gpu()
    
    # Create directories
    create_directories()
    
    # Check dataset access
    check_dataset_access()
    
    # Print next steps
    print_next_steps()


if __name__ == "__main__":
    main()

import os
import sys
import subprocess
import importlib.util

def check_package_installed(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def install_requirements():
    """Install required packages for fine-tuning"""
    
    required_packages = [
        "torch",
        "transformers>=4.21.0",
        "datasets",
        "accelerate",
        "peft",  # For LoRA
        "wandb",  # For logging
        "safetensors",
        "sentencepiece",  # For some tokenizers
        "protobuf",
    ]
    
    print("🔧 Setting up Thai Fine-tuning Environment")
    print("=" * 50)
    
    missing_packages = []
    
    # Check which packages are missing
    for package in required_packages:
        package_name = package.split(">=")[0].split("==")[0]
        if not check_package_installed(package_name):
            missing_packages.append(package)
        else:
            print(f"✅ {package_name} is already installed")
    
    if missing_packages:
        print(f"\n📦 Installing {len(missing_packages)} missing packages...")
        
        for package in missing_packages:
            print(f"Installing {package}...")
            try:
                subprocess.check_call([
                    sys.executable, "-m", "pip", "install", package, "--upgrade"
                ])
                print(f"✅ {package} installed successfully")
            except subprocess.CalledProcessError as e:
                print(f"❌ Failed to install {package}: {e}")
                return False
    else:
        print("\n✅ All required packages are already installed!")
    
    print("\n🎉 Environment setup complete!")
    return True

def check_environment():
    """Check if environment is properly set up"""
    
    print("\n🔍 Environment Check")
    print("=" * 30)
    
    # Check Python version
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version < (3, 8):
        print("⚠️  Warning: Python 3.8+ recommended")
    else:
        print("✅ Python version OK")
    
    # Check CUDA availability
    try:
        import torch
        cuda_available = torch.cuda.is_available()
        print(f"CUDA available: {cuda_available}")
        
        if cuda_available:
            print(f"CUDA devices: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
        else:
            print("💡 Running on CPU (consider using GPU for faster training)")
            
    except ImportError:
        print("❌ PyTorch not available")
        return False
    
    # Check other important packages
    packages_to_check = ["transformers", "datasets", "peft"]
    
    for package in packages_to_check:
        try:
            module = importlib.import_module(package)
            version = getattr(module, "__version__", "unknown")
            print(f"✅ {package}: {version}")
        except ImportError:
            print(f"❌ {package} not available")
            return False
    
    return True

def create_example_scripts():
    """Create example scripts for different use cases"""
    
    print("\n📝 Creating example scripts...")
    
    # Example training script
    train_example = '''#!/usr/bin/env python3
"""
Example: Fine-tune DialoGPT for Thai language
Usage: python train_example.py
"""

from finetune_pretrained import ThaiPretrainedFineTuner

def main():
    # Option 1: DialoGPT-medium (good for conversation)
    fine_tuner = ThaiPretrainedFineTuner(
        model_name="microsoft/DialoGPT-medium",
        output_dir="./thai_dialogpt_model",
        use_lora=True  # Use LoRA for efficient training
    )
    
    # Start fine-tuning
    fine_tuner.fine_tune(max_samples=3000)
    
    # Test the model
    fine_tuner.test_generation()

if __name__ == "__main__":
    main()
'''

    # Example inference script
    inference_example = '''#!/usr/bin/env python3
"""
Example: Use fine-tuned Thai model for inference
Usage: python inference_example.py
"""

from inference_pretrained import ThaiFinetunedInference
import os

def main():
    model_path = "./thai_finetuned_model"
    
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}")
        print("Please train a model first using train_example.py")
        return
    
    # Load model
    inference = ThaiFinetunedInference(model_path)
    
    # Test generation
    test_prompts = [
        "ประเทศไทยมีเมืองหลวงคือ",
        "วิทยาศาสตร์และเทคโนโลยีสมัยใหม่",
        "อาหารไทยที่มีชื่อเสียงได้แก่",
    ]
    
    print("🧪 Testing Thai text generation...")
    print("=" * 50)
    
    for prompt in test_prompts:
        result = inference.generate_thai_text(prompt, max_new_tokens=80)
        print(f"Input: {prompt}")
        print(f"Output: {result}")
        print("-" * 30)

if __name__ == "__main__":
    main()
'''

    # Model comparison script
    comparison_example = '''#!/usr/bin/env python3
"""
Example: Compare different pre-trained models for Thai fine-tuning
Usage: python compare_models.py
"""

from finetune_pretrained import ThaiPretrainedFineTuner

def compare_models():
    """Compare different base models"""
    
    models_to_test = [
        {
            "name": "microsoft/DialoGPT-medium",
            "description": "Conversational model, good for chat",
            "output_dir": "./comparison/dialogpt"
        },
        {
            "name": "gpt2",
            "description": "Standard GPT-2, general purpose",
            "output_dir": "./comparison/gpt2"
        },
        {
            "name": "distilgpt2", 
            "description": "Smaller, faster GPT-2",
            "output_dir": "./comparison/distilgpt2"
        }
    ]
    
    print("🔄 Comparing different base models for Thai fine-tuning")
    print("=" * 60)
    
    for model_info in models_to_test:
        print(f"\\n📊 Testing: {model_info['name']}")
        print(f"Description: {model_info['description']}")
        
        try:
            fine_tuner = ThaiPretrainedFineTuner(
                model_name=model_info["name"],
                output_dir=model_info["output_dir"],
                use_lora=True
            )
            
            # Train with smaller sample for comparison
            fine_tuner.fine_tune(max_samples=1000)
            
            print(f"✅ {model_info['name']} completed successfully")
            
        except Exception as e:
            print(f"❌ {model_info['name']} failed: {e}")
        
        print("-" * 40)

if __name__ == "__main__":
    compare_models()
'''

    # Save examples
    examples = [
        ("train_example.py", train_example),
        ("inference_example.py", inference_example),
        ("compare_models.py", comparison_example)
    ]
    
    for filename, content in examples:
        with open(filename, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"✅ Created {filename}")

def main():
    """Main setup function"""
    
    print("🇹🇭 Thai Pre-trained Model Fine-tuning Setup")
    print("=" * 50)
    
    # Install requirements
    if not install_requirements():
        print("❌ Failed to install requirements")
        return
    
    # Check environment
    if not check_environment():
        print("❌ Environment check failed")
        return
    
    # Create examples
    create_example_scripts()
    
    print("\n🎉 Setup Complete!")
    print("=" * 30)
    print("\nNext steps:")
    print("1. Run: python train_example.py")
    print("2. Test: python inference_example.py")
    print("3. Compare models: python compare_models.py")
    print("\nOR use the main scripts directly:")
    print("- python finetune_pretrained.py --model gpt2 --samples 5000")
    print("- python inference_pretrained.py ./thai_finetuned_model --mode chat")

if __name__ == "__main__":
    main()

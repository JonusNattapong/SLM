#!/usr/bin/env python3
"""
Quick training script for Thai language fine-tuning
Usage: python quick_train.py
"""

from finetune_pretrained import ThaiPretrainedFineTuner
import os
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def main():
    """Quick training with optimized settings"""
    
    print("🚀 Quick Thai Language Model Fine-tuning")
    print("=" * 50)
    
    # Choose best model for Thai
    model_options = [
        {
            "name": "microsoft/DialoGPT-medium",
            "description": "Best for conversational Thai (Recommended)",
            "output_dir": "./thai_dialogpt_model"
        },
        {
            "name": "gpt2",
            "description": "General purpose, faster training",
            "output_dir": "./thai_gpt2_model"
        },
        {
            "name": "distilgpt2",
            "description": "Smallest, fastest option",
            "output_dir": "./thai_distilgpt2_model"
        }
    ]
    
    print("Available models:")
    for i, model in enumerate(model_options, 1):
        print(f"{i}. {model['name']} - {model['description']}")
    
    # Auto-select or ask user
    choice = input("\nEnter choice (1-3) or press Enter for default (1): ").strip()
    
    if choice == "":
        choice = "1"
    
    try:
        selected_model = model_options[int(choice) - 1]
    except (ValueError, IndexError):
        print("Invalid choice, using default (DialoGPT-medium)")
        selected_model = model_options[0]
    
    print(f"\n✅ Selected: {selected_model['name']}")
    print(f"Output directory: {selected_model['output_dir']}")
    
    # Ask for sample size
    sample_size = input("\nEnter number of training samples (default: 3000): ").strip()
    
    try:
        sample_size = int(sample_size) if sample_size else 3000
    except ValueError:
        sample_size = 3000
        
    print(f"Training samples: {sample_size}")
    
    # Confirm training
    confirm = input("\nStart training? (y/N): ").strip().lower()
    
    if confirm != 'y':
        print("Training cancelled")
        return
    
    print("\n🔥 Starting fine-tuning...")
    print("-" * 30)
    
    try:
        # Create fine-tuner
        fine_tuner = ThaiPretrainedFineTuner(
            model_name=selected_model["name"],
            output_dir=selected_model["output_dir"],
            use_lora=True  # Always use LoRA for efficiency
        )
        
        # Start training
        fine_tuner.fine_tune(max_samples=sample_size)
        
        print("\n🎉 Training completed successfully!")
        print(f"Model saved to: {selected_model['output_dir']}")
        
        # Test the model
        print("\n🧪 Testing the model...")
        fine_tuner.test_generation()
        
        print("\n✅ All done! You can now use the model:")
        print(f"python inference_pretrained.py {selected_model['output_dir']} --mode chat")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"❌ Training failed: {e}")
        print("\nTroubleshooting tips:")
        print("1. Check your internet connection")
        print("2. Ensure you have enough disk space")
        print("3. Try reducing the sample size")
        print("4. Make sure all dependencies are installed")

if __name__ == "__main__":
    main()

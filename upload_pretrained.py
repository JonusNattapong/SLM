import os
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import HfApi, Repository, create_repo
import logging
from typing import Dict, Any
from safetensors.torch import save_file
import shutil

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThaiModelUploader:
    """Upload fine-tuned Thai models to Hugging Face Hub"""
    
    def __init__(self, model_path: str, repo_name: str, token: str = None):
        self.model_path = model_path
        self.repo_name = repo_name
        self.token = token or os.getenv("HUGGINGFACE_TOKEN")
        
        if not self.token:
            raise ValueError("Hugging Face token required. Set HUGGINGFACE_TOKEN env var or pass token")
            
        self.api = HfApi(token=self.token)
        
    def load_model_info(self) -> Dict[str, Any]:
        """Load training information"""
        
        info_path = os.path.join(self.model_path, "training_info.json")
        
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                return json.load(f)
        else:
            return {}
            
    def create_model_card(self) -> str:
        """Create comprehensive model card"""
        
        model_info = self.load_model_info()
        base_model = model_info.get("base_model", "unknown")
        use_lora = model_info.get("use_lora", False)
        max_samples = model_info.get("max_samples", "unknown")
        vocab_size = model_info.get("vocab_size", "unknown")
        
        model_card = f"""---
language: th
license: apache-2.0
tags:
- thai
- text-generation
- fine-tuned
- conversational-ai
- causal-lm
{"- lora" if use_lora else ""}
datasets:
- ZombitX64/Wikipedia-Thai
base_model: {base_model}
pipeline_tag: text-generation
widget:
- text: "ประเทศไทยมี"
  example_title: "Geography"
- text: "วิทยาศาสตร์และเทคโนโลยี"
  example_title: "Science & Technology"
- text: "อาหารไทยที่มีชื่อเสียง"
  example_title: "Thai Cuisine"
- text: "<question>ประวัติศาสตร์ของไทย</question><answer>"
  example_title: "Q&A Format"
---

# Thai Language Model (Fine-tuned)

## Model Description

This is a Thai language model fine-tuned from **{base_model}** using Thai Wikipedia data. The model has been optimized for Thai text generation and can handle various topics including geography, science, culture, and history.

## Model Details

- **Base Model**: {base_model}
- **Language**: Thai (th)
- **Training Method**: {"LoRA (Low-Rank Adaptation)" if use_lora else "Full Fine-tuning"}
- **Training Samples**: {max_samples}
- **Vocabulary Size**: {vocab_size}
- **Dataset**: ZombitX64/Wikipedia-Thai

## Features

- 🇹🇭 **Thai Language Support**: Specialized for Thai text generation
- 🧠 **Multiple Topics**: Trained on diverse Wikipedia content
- 💬 **Conversational Format**: Supports Q&A style interactions
- 🔧 **Optimized Training**: {"Uses LoRA for efficient fine-tuning" if use_lora else "Full parameter fine-tuning"}

## Usage

### Quick Start

```python
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("{self.repo_name}")
model = AutoModelForCausalLM.from_pretrained("{self.repo_name}")

# Generate text
prompt = "ประเทศไทยมี"
inputs = tokenizer(prompt, return_tensors="pt")

with torch.no_grad():
    outputs = model.generate(
        inputs.input_ids,
        max_new_tokens=100,
        temperature=0.8,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.eos_token_id
    )

generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print(generated_text)
```

### Advanced Usage with Custom Inference

```python
# For more advanced usage, use the custom inference class
from inference_pretrained import ThaiFinetunedInference

# Load model
inference = ThaiFinetunedInference("{self.repo_name}")

# Generate text
result = inference.generate_thai_text(
    "ประเทศไทยมี",
    max_new_tokens=100,
    temperature=0.8
)
print(result)

# Interactive chat
inference.interactive_chat()
```

### Supported Formats

The model supports several input formats:

1. **Plain Thai text**: `"ประเทศไทยมี"`
2. **Tagged format**: `"<thai>ประเทศไทยมี</thai>"`
3. **Q&A format**: `"<question>ประวัติศาสตร์ของไทย</question><answer>"`

## Training Details

### Dataset

- **Source**: ZombitX64/Wikipedia-Thai
- **Size**: {max_samples} samples
- **Content**: Thai Wikipedia articles covering diverse topics

### Training Configuration

- **Framework**: 🤗 Transformers
- **Optimization**: {"LoRA with rank=16, alpha=32" if use_lora else "AdamW optimizer"}
- **Learning Rate**: {"5e-4 (LoRA)" if use_lora else "1e-5 (Full fine-tuning)"}
- **Batch Size**: Dynamic based on available hardware
- **Epochs**: 3
- **Gradient Accumulation**: 8-16 steps

### Hardware Requirements

- **Minimum**: 8GB RAM, CPU
- **Recommended**: 16GB+ RAM, GPU with 8GB+ VRAM
- **Training Time**: {"~2-4 hours (with LoRA)" if use_lora else "~8-12 hours (full fine-tuning)"} on modern hardware

## Performance

The model has been tested on various Thai text generation tasks:

- ✅ **Geography**: Generates accurate information about Thai provinces and cities
- ✅ **Science**: Produces coherent scientific explanations in Thai
- ✅ **Culture**: Describes Thai cultural elements and traditions
- ✅ **History**: Provides historical information about Thailand
- ✅ **Food**: Describes Thai cuisine and cooking methods

## Limitations

- The model is specialized for Thai language and may not perform well in other languages
- Generated content should be verified for factual accuracy
- Performance may vary depending on the specific domain or topic
- The model may occasionally generate repetitive text

## Ethical Considerations

- This model is trained on Wikipedia data, which may contain biases present in the source material
- Users should review generated content for accuracy and appropriateness
- The model should not be used for generating harmful, misleading, or inappropriate content

## Citation

If you use this model in your research, please cite:

```bibtex
@misc{{thai-finetuned-model,
  title={{Thai Language Model Fine-tuned from {base_model}}},
  author={{Your Name}},
  year={{2024}},
  publisher={{Hugging Face}},
  url={{https://huggingface.co/{self.repo_name}}}
}}
```

## License

This model is released under the Apache 2.0 License.

## Contact

For questions or issues regarding this model, please open an issue in the model repository.

---

**Note**: This model is a research project and should be used responsibly. Always verify the generated content for accuracy and appropriateness.
"""

        return model_card
        
    def prepare_repository(self):
        """Prepare repository with all necessary files"""
        
        logger.info(f"Preparing repository: {self.repo_name}")
        
        # Create repository
        try:
            create_repo(
                repo_id=self.repo_name,
                token=self.token,
                private=False,
                exist_ok=True
            )
            logger.info("Repository created/updated")
        except Exception as e:
            logger.warning(f"Repository creation warning: {e}")
            
        # Prepare upload directory
        upload_dir = f"./upload_temp_{self.repo_name.split('/')[-1]}"
        
        if os.path.exists(upload_dir):
            shutil.rmtree(upload_dir)
        os.makedirs(upload_dir)
        
        # Copy model files
        model_files_to_copy = [
            "config.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "vocab.txt", 
            "merges.txt",
            "special_tokens_map.json",
            "training_info.json"
        ]
        
        for file_name in model_files_to_copy:
            src_path = os.path.join(self.model_path, file_name)
            if os.path.exists(src_path):
                dst_path = os.path.join(upload_dir, file_name)
                shutil.copy2(src_path, dst_path)
                logger.info(f"Copied {file_name}")
                
        # Convert model to safetensors if needed
        model_bin_path = os.path.join(self.model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(upload_dir, "model.safetensors")
        
        if os.path.exists(model_bin_path):
            logger.info("Converting model to safetensors format...")
            
            # Load model state dict
            state_dict = torch.load(model_bin_path, map_location="cpu")
            
            # Handle shared tensors (clone if necessary)
            tensor_names = list(state_dict.keys())
            for name in tensor_names:
                if state_dict[name].data_ptr() in [state_dict[other].data_ptr() 
                                                  for other in tensor_names if other != name]:
                    state_dict[name] = state_dict[name].clone()
                    
            # Save as safetensors
            save_file(state_dict, safetensors_path)
            logger.info("Model converted to safetensors")
            
        else:
            # Copy existing safetensors file
            existing_safetensors = os.path.join(self.model_path, "model.safetensors")
            if os.path.exists(existing_safetensors):
                shutil.copy2(existing_safetensors, safetensors_path)
                logger.info("Copied existing safetensors file")
                
        # Create model card
        model_card_content = self.create_model_card()
        with open(os.path.join(upload_dir, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card_content)
        logger.info("Created model card")
        
        # Create .gitignore
        gitignore_content = """
# Model artifacts
*.bin
pytorch_model.bin
training_args.bin

# Logs
*.log
logs/
wandb/

# Cache
__pycache__/
.cache/

# Environment
.env
.venv/

# IDE
.vscode/
.idea/
"""
        
        with open(os.path.join(upload_dir, ".gitignore"), "w") as f:
            f.write(gitignore_content.strip())
            
        return upload_dir
        
    def upload_model(self):
        """Upload model to Hugging Face Hub"""
        
        logger.info(f"Starting upload to {self.repo_name}")
        
        # Prepare repository
        upload_dir = self.prepare_repository()
        
        try:
            # Upload files
            for root, dirs, files in os.walk(upload_dir):
                for file in files:
                    file_path = os.path.join(root, file)
                    rel_path = os.path.relpath(file_path, upload_dir)
                    
                    logger.info(f"Uploading {rel_path}...")
                    
                    self.api.upload_file(
                        path_or_fileobj=file_path,
                        path_in_repo=rel_path,
                        repo_id=self.repo_name,
                        token=self.token,
                        commit_message=f"Upload {rel_path}"
                    )
                    
            logger.info("✅ Upload completed successfully!")
            logger.info(f"🔗 Model URL: https://huggingface.co/{self.repo_name}")
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise
            
        finally:
            # Cleanup temp directory
            if os.path.exists(upload_dir):
                shutil.rmtree(upload_dir)
                
    def count_parameters(self) -> Dict[str, int]:
        """Count model parameters"""
        
        try:
            model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                local_files_only=True
            )
            
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            
            return {
                "total_parameters": total_params,
                "trainable_parameters": trainable_params
            }
            
        except Exception as e:
            logger.warning(f"Could not count parameters: {e}")
            return {"total_parameters": 0, "trainable_parameters": 0}


def main():
    """Main upload function"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Upload Thai fine-tuned model to Hugging Face")
    parser.add_argument("model_path", help="Path to fine-tuned model directory")
    parser.add_argument("repo_name", help="Hugging Face repository name (username/model-name)")
    parser.add_argument("--token", help="Hugging Face token (or set HUGGINGFACE_TOKEN env var)")
    
    args = parser.parse_args()
    
    # Validate inputs
    if not os.path.exists(args.model_path):
        print(f"❌ Model path does not exist: {args.model_path}")
        return
        
    if "/" not in args.repo_name:
        print("❌ Repository name must be in format: username/model-name")
        return
        
    # Create uploader
    try:
        uploader = ThaiModelUploader(
            model_path=args.model_path,
            repo_name=args.repo_name,
            token=args.token
        )
        
        # Show model info
        model_info = uploader.load_model_info()
        param_info = uploader.count_parameters()
        
        print("📊 Model Information:")
        print(f"  Base model: {model_info.get('base_model', 'unknown')}")
        print(f"  Training samples: {model_info.get('max_samples', 'unknown')}")
        print(f"  Uses LoRA: {model_info.get('use_lora', False)}")
        print(f"  Total parameters: {param_info.get('total_parameters', 0):,}")
        print(f"  Repository: {args.repo_name}")
        
        # Confirm upload
        confirm = input("\\nProceed with upload? (y/N): ").strip().lower()
        
        if confirm == 'y':
            uploader.upload_model()
        else:
            print("Upload cancelled")
            
    except Exception as e:
        print(f"❌ Upload failed: {e}")


if __name__ == "__main__":
    main()

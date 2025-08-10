import os
import json
import torch
from huggingface_hub import HfApi, Repository, upload_folder
from transformers import PreTrainedTokenizerFast
from safetensors.torch import save_file
from model import SLMConfig, SLMForCausalLM
import logging

# Load environment variables from .env file manually
def load_env_file():
    env_path = ".env"
    if os.path.exists(env_path):
        with open(env_path, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    os.environ[key] = value

load_env_file()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HuggingFaceUploader:
    """Upload model to Hugging Face Hub"""
    
    def __init__(self, model_path: str, hf_token: str = None):
        self.model_path = model_path
        self.api = HfApi(token=hf_token)
        self.hf_token = hf_token
    
    def _count_parameters(self) -> str:
        """Count parameters from model file"""
        try:
            # Try safetensors first
            safetensors_path = os.path.join(self.model_path, "model.safetensors")
            pytorch_path = os.path.join(self.model_path, "pytorch_model.bin")
            
            if os.path.exists(safetensors_path):
                from safetensors.torch import load_file
                state_dict = load_file(safetensors_path)
            elif os.path.exists(pytorch_path):
                state_dict = torch.load(pytorch_path, map_location="cpu")
            else:
                return "Unknown"
            
            total_params = sum(p.numel() for p in state_dict.values())
            return f"{total_params:,}"
        except Exception:
            return "Unknown"
    
    def create_model_card(self, model_name: str, config: SLMConfig) -> str:
        """Create model card for the repository"""
        
        model_card = f"""---
language:
- th
license: apache-2.0
tags:
- thai
- language-model
- mixture-of-experts
- small-language-model
- transformers
datasets:
- ZombitX64/Wikipedia-Thai
widget:
- text: "ประเทศไทยมีจังหวัด"
  example_title: "Thai Geography"
- text: "วิทยาศาสตร์และเทคโนโลยี"
  example_title: "Science and Technology"
- text: "อาหารไทยที่มีชื่อเสียง"
  example_title: "Thai Cuisine"
---

# Thai Small Language Model with Mixture of Experts (SLM-MoE)

## Model Description

This is a Small Language Model (SLM) with Mixture of Experts (MoE) architecture specifically designed for the Thai language. The model was trained from scratch using the ZombitX64/Wikipedia-Thai dataset.

### Model Architecture

- **Base Architecture**: Transformer decoder with MoE layers
- **Parameters**: ~{self._count_parameters()}
- **Hidden Size**: {config.hidden_size}
- **Layers**: {config.num_hidden_layers}
- **Attention Heads**: {config.num_attention_heads}
- **Experts**: {config.num_experts}
- **Experts per Token**: {config.num_experts_per_token}
- **Vocabulary Size**: {config.vocab_size:,}
- **Max Sequence Length**: {config.max_position_embeddings}

### Key Features

- **Mixture of Experts (MoE)**: Efficient scaling with {config.num_experts} experts per layer
- **Rotary Position Embedding (RoPE)**: Better position encoding for longer sequences
- **SwiGLU Activation**: Modern activation function for better performance
- **Thai Language Optimized**: Custom tokenizer and training for Thai text

### Training Details

- **Dataset**: ZombitX64/Wikipedia-Thai
- **Training Framework**: PyTorch
- **Tokenizer**: Custom ByteLevelBPE tokenizer trained on Thai text
- **Optimization**: AdamW with cosine annealing learning rate schedule
- **Regularization**: Load balancing and router z-loss for MoE stability

## Usage

### Installation

```bash
pip install torch transformers tokenizers
```

### Basic Usage

```python
import torch
from transformers import PreTrainedTokenizerFast

# Load model and tokenizer
model_name = "{model_name}"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# For inference, you'll need to load the custom model architecture
# (See the repository for the complete model code)

# Generate text
prompt = "ประเทศไทยมีจังหวัด"
inputs = tokenizer(prompt, return_tensors="pt")
# ... (generation code)
```

### Training Code

The complete training code is available in the repository, including:
- Custom model architecture (`model.py`)
- Dataset preprocessing (`dataset.py`)
- Training script (`train.py`)
- Inference utilities (`inference.py`)

## Performance

This model is designed for efficient inference while maintaining good quality for Thai text generation tasks.

### Intended Use

- Thai text completion
- Creative writing assistance
- Educational content generation
- Research in Thai NLP

### Limitations

- Trained on Wikipedia data, may not cover all domains
- Small model size may limit complex reasoning
- Generated content should be verified for accuracy

## Training Data

The model was trained on the [ZombitX64/Wikipedia-Thai](https://huggingface.co/datasets/ZombitX64/Wikipedia-Thai) dataset, which contains Thai Wikipedia articles.

## Ethical Considerations

- The model may reflect biases present in the training data
- Generated content should not be considered factual without verification
- Use responsibly and consider potential impacts

## Citation

```bibtex
@misc{{thai-slm-moe,
  title={{Thai Small Language Model with Mixture of Experts}},
  author={{Your Name}},
  year={{2024}},
  howpublished={{\\url{{https://huggingface.co/{model_name}}}}},
}}
```

## Acknowledgments

- Dataset: ZombitX64/Wikipedia-Thai
- Inspired by modern language model architectures
- Built with PyTorch and Transformers library

---

*This model was created for research and educational purposes. Please use responsibly.*
"""
        return model_card
    
    def create_config_json(self, config: SLMConfig) -> dict:
        """Create config.json for Hugging Face"""
        hf_config = {
            "architectures": ["SLMForCausalLM"],
            "model_type": "slm_moe",
            "vocab_size": config.vocab_size,
            "hidden_size": config.hidden_size,
            "num_hidden_layers": config.num_hidden_layers,
            "num_attention_heads": config.num_attention_heads,
            "intermediate_size": config.intermediate_size,
            "hidden_dropout_prob": config.hidden_dropout_prob,
            "attention_dropout_prob": config.attention_dropout_prob,
            "max_position_embeddings": config.max_position_embeddings,
            "layer_norm_eps": config.layer_norm_eps,
            "initializer_range": config.initializer_range,
            "use_cache": config.use_cache,
            "pad_token_id": config.pad_token_id,
            "bos_token_id": config.bos_token_id,
            "eos_token_id": config.eos_token_id,
            "num_experts": config.num_experts,
            "num_experts_per_token": config.num_experts_per_token,
            "expert_capacity_factor": config.expert_capacity_factor,
            "aux_loss_alpha": config.aux_loss_alpha,
            "router_z_loss_alpha": config.router_z_loss_alpha,
            "torch_dtype": "float32",
            "transformers_version": "4.30.0"
        }
        return hf_config
    
    def prepare_upload_files(self, model_name: str):
        """Prepare files for upload"""
        
        # Load config
        with open(os.path.join(self.model_path, "config.json"), "r") as f:
            config_dict = json.load(f)
        
        # Filter config to only include SLMConfig parameters
        slm_config_params = {
            'vocab_size', 'hidden_size', 'num_hidden_layers', 'num_attention_heads',
            'intermediate_size', 'hidden_dropout_prob', 'attention_dropout_prob',
            'max_position_embeddings', 'layer_norm_eps', 'initializer_range',
            'use_cache', 'pad_token_id', 'bos_token_id', 'eos_token_id',
            'num_experts', 'num_experts_per_token', 'expert_capacity_factor',
            'aux_loss_alpha', 'router_z_loss_alpha'
        }
        
        filtered_config = {k: v for k, v in config_dict.items() if k in slm_config_params}
        config = SLMConfig(**filtered_config)
        
        # Create HF-compatible config
        hf_config = self.create_config_json(config)
        with open(os.path.join(self.model_path, "config.json"), "w") as f:
            json.dump(hf_config, f, indent=2)
        
        # Create model card
        model_card = self.create_model_card(model_name, config)
        with open(os.path.join(self.model_path, "README.md"), "w", encoding="utf-8") as f:
            f.write(model_card)
        
        # Create requirements.txt for the model
        requirements = """torch>=2.0.0
transformers>=4.30.0
tokenizers>=0.13.0
safetensors>=0.3.0
"""
        with open(os.path.join(self.model_path, "requirements.txt"), "w") as f:
            f.write(requirements)
        
        # Convert model to safetensors if needed
        self.convert_to_safetensors()
        
        # Copy model files
        model_files = [
            "model.safetensors",
            "tokenizer.json",
            "tokenizer_config.json", 
            "special_tokens_map.json"
        ]
        
        logger.info("Files prepared for upload:")
        for file in os.listdir(self.model_path):
            if os.path.isfile(os.path.join(self.model_path, file)):
                logger.info(f"  - {file}")
    
    def convert_to_safetensors(self):
        """Convert PyTorch model to safetensors format"""
        pytorch_model_path = os.path.join(self.model_path, "pytorch_model.bin")
        safetensors_path = os.path.join(self.model_path, "model.safetensors")
        
        if os.path.exists(pytorch_model_path) and not os.path.exists(safetensors_path):
            logger.info("Converting model to safetensors format...")
            
            # Load the state dict
            state_dict = torch.load(pytorch_model_path, map_location="cpu")
            
            # Handle shared tensors (weight tying)
            # The lm_head.weight and model.embed_tokens.weight are tied in our model
            # We need to create separate copies to avoid shared memory issues
            processed_state_dict = {}
            for key, tensor in state_dict.items():
                # Create a copy of the tensor to break memory sharing
                processed_state_dict[key] = tensor.clone()
            
            # Save as safetensors with the processed state dict
            save_file(processed_state_dict, safetensors_path)
            
            logger.info(f"Model converted and saved as {safetensors_path}")
        elif os.path.exists(safetensors_path):
            logger.info("Safetensors file already exists")
        else:
            logger.warning("No PyTorch model file found to convert")
    
    def upload_to_hub(self, 
                      repo_name: str,
                      organization: str = None,
                      private: bool = False):
        """Upload model to Hugging Face Hub"""
        
        # repo_name should already include username (e.g., "username/model-name")
        full_repo_name = repo_name
        
        logger.info(f"Uploading to: {full_repo_name}")
        
        try:
            # Create repository
            self.api.create_repo(
                repo_id=full_repo_name,
                exist_ok=True,
                private=private,
                repo_type="model"
            )
            
            # Prepare files
            self.prepare_upload_files(full_repo_name)
            
            # Upload files
            self.api.upload_folder(
                folder_path=self.model_path,
                repo_id=full_repo_name,
                repo_type="model",
                commit_message="Upload Thai SLM MoE model"
            )
            
            logger.info(f"Model successfully uploaded to: https://huggingface.co/{full_repo_name}")
            
        except Exception as e:
            logger.error(f"Upload failed: {e}")
            raise


def upload_model():
    """Main upload function"""
    
    model_path = "./thai_slm_moe_model"
    
    if not os.path.exists(model_path):
        logger.error(f"Model not found at {model_path}")
        logger.error("Please train the model first using train.py")
        return
    
    # Get Hugging Face token
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        logger.error("Hugging Face token not found.")
        logger.error("Please set the HF_TOKEN environment variable")
        logger.error("Get your token from: https://huggingface.co/settings/tokens")
        return
    
    # Initialize uploader
    uploader = HuggingFaceUploader(model_path, hf_token)
    
    # Get username from token (we'll create the repo under your username)
    try:
        whoami = uploader.api.whoami()
        username = whoami["name"]
        logger.info(f"Authenticated as: {username}")
    except Exception as e:
        logger.error(f"Failed to get user info: {e}")
        return
    
    # Upload configuration - Updated with new name
    repo_name = f"{username}/thai-slm-moe-v2"  # New repository name
    
    # Upload
    uploader.upload_to_hub(
        repo_name=repo_name,
        organization=None,
        private=False  # Set to True for private repo
    )


if __name__ == "__main__":
    upload_model()

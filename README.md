# Thai Small Language Model with Mixture of Experts (SLM-MoE)

This is a Small Language Model (SLM) with Mixture of Experts (MoE) architecture specifically designed for the Thai language. The model was trained from scratch using the ZombitX64/Wikipedia-Thai dataset.


## 🌟 Features

- **Mixture of Experts (MoE)**: Efficient scaling with 4 experts per layer, top-2 routing
- **Rotary Position Embedding (RoPE)**: Qwen/DeepSeek-style scaling for long context
- **SwiGLU Activation**: Modern activation for better performance
- **RMSNorm**: Stable normalization for deep transformers
- **GQA**: Grouped Query Attention for memory efficiency
- **Thai Language Optimized**: Custom tokenizer and training for Thai text


## 🏗️ Architecture

- **Base Architecture**: Transformer decoder with MoE layers
- **Parameters**: ~140M
- **Hidden Size**: 512
- **Layers**: 8
- **Attention Heads**: 8
- **Experts**: 4
- **Experts per Token**: 2
- **Vocabulary Size**: 30,000
- **Max Sequence Length**: 512


## 📋 Requirements

```bash
pip install torch transformers tokenizers
```


## 🚀 Quick Start

### Basic Usage

```python
import torch
from transformers import PreTrainedTokenizerFast
from model import SLMForCausalLM, SLMConfig

# Load tokenizer
model_name = "JonusNattapong/thai-slm-moe"
tokenizer = PreTrainedTokenizerFast.from_pretrained(model_name)

# Load model config and weights
config = SLMConfig.from_pretrained(model_name)
model = SLMForCausalLM(config)
model.load_state_dict(torch.load("thai_slm_moe_model/pytorch_model.bin", map_location="cpu"))
model.eval()

# Generate text
prompt = "ประเทศไทยมีจังหวัด"
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    output = model.generate(inputs["input_ids"], max_length=64)
    print(tokenizer.decode(output[0], skip_special_tokens=True))
```


## 📁 Project Structure

```
SLM/
├── model.py              # Model architecture (MoE, RoPE, SwiGLU, RMSNorm, GQA)
├── dataset.py            # Thai dataset preprocessing and tokenizer
├── train.py              # Training script with wandb logging
├── inference.py          # Text generation and chat interface
├── upload_to_hf.py       # Hugging Face Hub upload utility
├── gradio_app.py         # Web interface with Gradio
├── requirements.txt      # Python dependencies
├── run_training.bat      # Windows training script
└── run_training.sh       # Linux/Mac training script
```


## 🎯 Usage Examples

### Basic Text Generation

```python
from inference import ThaiSLMInference

# Load model
model = ThaiSLMInference("./thai_slm_moe_model")

# Generate text
prompt = "ประเทศไทยมีจังหวัด"
generated = model.generate_text(
    prompt=prompt,
    max_length=100,
    temperature=0.8
)
print(generated)
```

### Interactive Chat

```python
# Chat mode
model = ThaiSLMInference("./thai_slm_moe_model")
response = model.chat("สวัสดีครับ")
print(response)
```

## 🔧 Model Configuration

```python
SLMConfig(
    vocab_size=30000,
    hidden_size=768,
    num_hidden_layers=12,
    num_attention_heads=12,
    intermediate_size=3072,
    max_position_embeddings=1024,
    num_experts=8,
    num_experts_per_token=2,
    aux_loss_alpha=0.01,
    router_z_loss_alpha=0.001
)
```

## 📊 Training Details

- **Dataset**: ZombitX64/Wikipedia-Thai
- **Tokenizer**: Custom ByteLevelBPE (30K vocab)
- **Optimizer**: AdamW with cosine annealing
- **Batch Size**: 4 (with gradient accumulation)
- **Learning Rate**: 1e-4
- **Epochs**: 3
- **Hardware**: CUDA-enabled GPU recommended

## 🎛️ Generation Parameters

- **Temperature**: Controls creativity (0.1-2.0)
- **Top-K**: Number of top tokens to consider (1-100)
- **Top-P**: Cumulative probability threshold (0.1-1.0)
- **Max Length**: Maximum generation length

## 📈 Monitoring

Training progress is logged to Weights & Biases (wandb):

- Loss curves
- Learning rate schedule
- Auxiliary losses (load balancing, router z-loss)
- GPU utilization

## 🌐 Web Interface

The Gradio web interface provides:

- Interactive text generation
- Parameter adjustment
- Example prompts
- Model information
- Thai language support

## 📤 Hugging Face Integration

Upload your trained model to Hugging Face Hub:

1. Get your token from [HuggingFace Settings](https://huggingface.co/settings/tokens)
2. Set environment variable: `export HF_TOKEN="your_token"`
3. Run: `python upload_to_hf.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if needed
5. Submit a pull request

## 📄 License

This project is licensed under the Apache 2.0 License.

## 🙏 Acknowledgments

- **Dataset**: [ZombitX64/Wikipedia-Thai](https://huggingface.co/datasets/ZombitX64/Wikipedia-Thai)
- **Inspiration**: Modern language model architectures (GPT, PaLM, Switch Transformer)
- **Framework**: PyTorch, Transformers, Gradio

## 📚 Technical Details

### Mixture of Experts (MoE)

The model uses MoE layers for efficient scaling:
- 8 experts per layer
- Top-2 routing (2 experts active per token)
- Load balancing loss for expert utilization
- Router z-loss for training stability

### Rotary Position Embedding (RoPE)

RoPE provides better position encoding:
- Relative position encoding
- Better extrapolation to longer sequences
- Efficient implementation with precomputed values

### SwiGLU Activation

Modern activation function in MoE experts:
- Gated linear units with Swish activation
- Better performance than ReLU/GELU
- Used in recent large language models

## 🐛 Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce batch size or use gradient checkpointing
2. **Dataset loading fails**: Check internet connection and dataset availability
3. **Model not converging**: Adjust learning rate or warmup steps
4. **Generation quality poor**: Try different temperature/top-p values

### Performance Tips

- Use mixed precision training (FP16)
- Enable gradient checkpointing for large models
- Use multiple GPUs with DataParallel/DistributedDataParallel
- Optimize tokenizer vocabulary size for your data

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Search existing issues on GitHub
3. Create a new issue with detailed information
4. Join our community discussions

---

**Built with ❤️ for the Thai NLP community**

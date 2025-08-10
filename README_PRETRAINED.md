# Thai Language Model - Pre-trained Fine-tuning Approach

เปลี่ยนมาใช้วิธี Fine-tune Pre-trained Model แทนการสร้างจากศูนย์เพื่อให้ได้ผลลัพธ์ที่ดีกว่า

## 🎯 Overview

สคริปต์ใหม่นี้ใช้ pre-trained models และ fine-tune สำหรับภาษาไทย ซึ่งจะให้ผลลัพธ์ที่ดีกว่าการเทรนจากศูนย์มาก

## 📁 New Files

### Core Scripts
- **`finetune_pretrained.py`** - หลักสำหรับ fine-tune pre-trained models
- **`inference_pretrained.py`** - สำหรับทดสอบและใช้งานโมเดล
- **`upload_pretrained.py`** - อัปโหลดโมเดลไป Hugging Face

### Utility Scripts  
- **`setup_pretrained.py`** - ติดตั้ง dependencies และสร้าง examples
- **`quick_train.py`** - เทรนด่วนแบบ interactive

## 🚀 Quick Start

### 1. Setup Environment
```bash
python setup_pretrained.py
```

### 2. Quick Training (แนะนำ)
```bash
python quick_train.py
```

### 3. Manual Training
```bash
# DialoGPT (ดีที่สุดสำหรับ conversation)
python finetune_pretrained.py --model microsoft/DialoGPT-medium --samples 5000 --output_dir ./thai_dialogpt_model

# GPT-2 (general purpose)
python finetune_pretrained.py --model gpt2 --samples 5000 --output_dir ./thai_gpt2_model

# DistilGPT-2 (เล็กและเร็ว)
python finetune_pretrained.py --model distilgpt2 --samples 3000 --output_dir ./thai_distilgpt2_model
```

### 4. Test Model
```bash
# Interactive chat
python inference_pretrained.py ./thai_dialogpt_model --mode chat

# Benchmark testing
python inference_pretrained.py ./thai_dialogpt_model --mode benchmark

# Single generation
python inference_pretrained.py ./thai_dialogpt_model --mode generate --prompt "ประเทศไทยมี"
```

### 5. Upload to Hugging Face
```bash
python upload_pretrained.py ./thai_dialogpt_model YourUsername/thai-dialogpt-v1
```

## 🎯 Recommended Models

### 1. microsoft/DialoGPT-medium (แนะนำที่สุด)
- **ข้อดี**: เหมาะสำหรับ conversation, คุณภาพดี
- **ข้อเสีย**: ใหญ่กว่า, ใช้เวลาเทรนนานกว่า
- **ใช้เมื่อ**: ต้องการคุณภาพสูงสุด

### 2. gpt2 
- **ข้อดี**: balanced ระหว่างขนาดและคุณภาพ
- **ข้อเสีย**: ไม่เชี่ยวชาญ conversation เท่าไหร่
- **ใช้เมื่อ**: ต้องการ general purpose

### 3. distilgpt2
- **ข้อดี**: เล็ก, เร็ว, ใช้ RAM น้อย
- **ข้อเสีย**: คุณภาพต่ำกว่า
- **ใช้เมื่อ**: จำกัดด้าน hardware

## 🔧 Features

### LoRA (Low-Rank Adaptation)
- ลด memory usage 50-80%
- เทรนเร็วกว่า 2-3 เท่า
- คุณภาพใกล้เคียง full fine-tuning
- **เปิดใช้งานอัตโนมัติ**

### Smart Training
- **Auto-detect device**: CUDA/CPU
- **Dynamic batch size**: ปรับตาม hardware
- **Gradient accumulation**: ป้องกัน memory overflow
- **WandB logging**: track การเทรน

### Advanced Inference
- **Interactive chat**: สนทนาแบบ real-time
- **Benchmark mode**: ทดสอบหลายหัวข้อ
- **Customizable generation**: ปรับ temperature, top_p, etc.
- **Thai text optimization**: ปรับแต่งสำหรับภาษาไทย

## 📊 Expected Results

### Before (Custom Model)
```
Input: ประเทศไทยมี
Output: ์ิัู้่ึ่s ofี่ หมวดหม็ the andุื่ หรุ้aingี่ere B์้็ D S b และ่yู่es ซิ andon เป
```

### After (Fine-tuned Pre-trained)
```
Input: ประเทศไทยมี
Output: ประเทศไทยมี 77 จังหวัด โดยแบ่งเป็น 5 ภูมิภาค ได้แก่ ภาคเหนือ ภาคกลาง ภาคตะวันออกเฉียงเหนือ ภาคตะวันออก และภาคใต้
```

## 💡 Why Pre-trained is Better

### 1. **Quality**
- โมเดล pre-trained รู้โครงสร้างภาษาแล้ว
- เพียงปรับแต่งให้เข้ากับภาษาไทย
- ผลลัพธ์ coherent และ meaningful

### 2. **Speed**
- LoRA ทำให้เทรนเร็วขึ้น 3-5 เท่า
- ใช้ memory น้อยกว่า 50-80%
- สามารถรันบน laptop ธรรมดาได้

### 3. **Reliability**
- มี architecture ที่พิสูจน์แล้ว
- ไม่มีปัญหา gradient exploding/vanishing
- การ converge มั่นคง

## 🔍 Training Tips

### Sample Size Guidelines
- **Small test**: 1,000 samples (~30 minutes)
- **Development**: 3,000 samples (~1-2 hours)  
- **Production**: 5,000+ samples (~3-4 hours)

### Hardware Recommendations
- **Minimum**: 8GB RAM + CPU (ใช้ได้แต่ช้า)
- **Good**: 16GB RAM + GTX 1060 (6GB)
- **Optimal**: 32GB RAM + RTX 3080 (10GB+)

### Memory Optimization
```python
# ลด batch size ถ้า out of memory
per_device_train_batch_size=2  # แทน 4
gradient_accumulation_steps=16  # แทน 8

# ใช้ fp16 สำหรับ GPU
fp16=True  # ลด memory 50%
```

## 📈 Monitoring Training

### WandB Integration
```bash
# Set up WandB (optional)
export WANDB_API_KEY="your_key_here"
python finetune_pretrained.py --model gpt2
```

### Local Monitoring
```bash
# ดู log file
tail -f logs/training.log

# Monitor GPU usage
nvidia-smi -l 1
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Out of Memory
```bash
# Solution: Reduce batch size
python finetune_pretrained.py --model distilgpt2 --samples 1000
```

#### 2. Slow Training
```bash
# Solution: Use smaller model or fewer samples
python finetune_pretrained.py --model distilgpt2 --samples 2000
```

#### 3. Poor Quality
```bash
# Solution: Use larger model or more samples
python finetune_pretrained.py --model microsoft/DialoGPT-medium --samples 8000
```

#### 4. Connection Error
```bash
# Check internet connection
ping huggingface.co

# Use cached model if available
export HF_DATASETS_OFFLINE=1
```

## 📝 Custom Usage

### Advanced Training Configuration
```python
from finetune_pretrained import ThaiPretrainedFineTuner

fine_tuner = ThaiPretrainedFineTuner(
    model_name="microsoft/DialoGPT-medium",
    output_dir="./my_custom_model",
    use_lora=True
)

# Custom training
fine_tuner.fine_tune(max_samples=10000)
```

### Custom Inference
```python
from inference_pretrained import ThaiFinetunedInference

# Load model
inference = ThaiFinetunedInference("./my_custom_model")

# Generate with custom parameters
result = inference.generate_thai_text(
    "ประเทศไทย",
    max_new_tokens=150,
    temperature=0.7,
    top_p=0.8,
    repetition_penalty=1.2
)
```

## 🎯 Next Steps

1. **เทรนโมเดลแรก**: `python quick_train.py`
2. **ทดสอบคุณภาพ**: `python inference_pretrained.py ./model --mode benchmark`
3. **ปรับแต่งพารามิเตอร์**: เปลี่ยน temperature, top_p
4. **อัปโหลด**: `python upload_pretrained.py ./model username/model-name`
5. **ใช้งานจริง**: integrate กับแอปพลิเคชัน

## 📚 Resources

- [Hugging Face Transformers](https://huggingface.co/docs/transformers)
- [LoRA Paper](https://arxiv.org/abs/2106.09685)
- [Thai NLP Resources](https://github.com/PyThaiNLP/pythainlp)

---

**Note**: วิธีใหม่นี้จะให้ผลลัพธ์ที่ดีกว่าการสร้างจากศูนย์มาก และประหยัดเวลาในการเทรน

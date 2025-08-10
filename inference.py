import torch
import json
import os
from transformers import PreTrainedTokenizerFast
from model import SLMForCausalLM, SLMConfig
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThaiSLMInference:
    """Inference class for Thai Small Language Model"""
    
    def __init__(self, model_path: str, device: str = None):
        self.model_path = model_path
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load config
        config_path = os.path.join(model_path, "config.json")
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = json.load(f)
        
        # Only keep keys that are valid for SLMConfig
        slm_config_keys = {f.name for f in SLMConfig.__dataclass_fields__.values()}
        filtered_config = {k: v for k, v in config_dict.items() if k in slm_config_keys}
        self.config = SLMConfig(**filtered_config)
        
        # Load tokenizer
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(model_path)
        
        # Collect special token ids to avoid generating them
        self.bad_token_ids = []
        specials = [
            getattr(self.tokenizer, 'pad_token_id', None),
            getattr(self.tokenizer, 'unk_token_id', None),
            getattr(self.tokenizer, 'cls_token_id', None),
            getattr(self.tokenizer, 'mask_token_id', None),
            getattr(self.tokenizer, 'sep_token_id', None),
        ]
        # Also add tokens that might cause issues (very low/high frequency)
        vocab_size = len(self.tokenizer)
        bad_ranges = list(range(0, 10)) + list(range(vocab_size-10, vocab_size))
        self.bad_token_ids = [tid for tid in specials if tid is not None] + bad_ranges
        
        # Load model
        self.model = SLMForCausalLM(self.config)
        
        # Try to load safetensors first, fallback to pytorch_model.bin
        safetensors_path = os.path.join(model_path, "model.safetensors")
        pytorch_path = os.path.join(model_path, "pytorch_model.bin")
        
        if os.path.exists(safetensors_path):
            try:
                from safetensors.torch import load_file
                model_state = load_file(safetensors_path, device=str(self.device))
                logger.info("Loaded model from safetensors format")
            except ImportError:
                logger.warning("safetensors not installed, falling back to pytorch format")
                model_state = torch.load(pytorch_path, map_location=self.device)
        elif os.path.exists(pytorch_path):
            model_state = torch.load(pytorch_path, map_location=self.device)
            logger.info("Loaded model from pytorch format")
        else:
            raise FileNotFoundError(f"No model file found in {model_path}")
        
        self.model.load_state_dict(model_state)
        self.model.to(self.device)
        self.model.eval()
        
        logger.info(f"Model loaded from {model_path}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Vocab size: {len(self.tokenizer)}")
        logger.info(f"Bad token ids masked during generation: {self.bad_token_ids}")
    
    def generate_text(self,
                     prompt: str,
                     max_length: int = 100,  # ลดลงเพื่อคุณภาพดีขึ้น
                     temperature: float = 0.7,  # ลดความ random
                     top_k: int = 40,  # จำกัด choices
                     top_p: float = 0.8,  # ลดนิดหน่อย
                     do_sample: bool = True,
                     num_return_sequences: int = 1,
                     repetition_penalty: float = 1.2,  # เพิ่มการลงโทษ repetition
                     frequency_penalty: float = 0.3,  # ลดลง
                     presence_penalty: float = 0.2,   # ลดลง
                     penalty_window: int = 64) -> str:  # ลด window
        """Generate text from prompt"""
        
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_position_embeddings - max_length
        )
        input_ids = inputs["input_ids"].to(self.device)
        
        # Generate
        with torch.no_grad():
            generated_ids = self.model.generate(
                input_ids=input_ids,
                max_length=input_ids.size(1) + max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=repetition_penalty,
                frequency_penalty=frequency_penalty,
                presence_penalty=presence_penalty,
                penalty_window=penalty_window,
                bad_token_ids=self.bad_token_ids,
            )
        
        # Decode generated text
        generated_text = self.tokenizer.decode(
            generated_ids[0],
            skip_special_tokens=True
        )
        
        # Remove the input prompt from output
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        return generated_text
    
    def chat(self, prompt: str, max_length: int = 50) -> str:  # ลดขนาด
        """Interactive chat interface"""
        return self.generate_text(
            prompt=prompt,
            max_length=max_length,
            temperature=0.6,  # ลด temperature
            top_k=30,  # ลด top_k
            top_p=0.7,  # ลด top_p
            do_sample=True
        )
    
    def complete_text(self, text: str, max_length: int = 50) -> str:  # ลดขนาด
        """Complete given text"""
        return self.generate_text(
            prompt=text,
            max_length=max_length,
            temperature=0.5,  # ลด temperature มากขึ้น
            top_k=25,  # ลด top_k
            do_sample=True
        )


def test_model():
    """Test the trained model"""
    
    # Thai test prompts
    test_prompts = [
        "ประเทศไทยมีจังหวัด",
        "วิทยาศาสตร์และเทคโนโลยี",
        "การศึกษาในยุคดิจิทัล",
        "อาหารไทยที่มีชื่อเสียง",
        "ประวัติศาสตร์ของไทย"
    ]
    
    try:
        # Initialize inference
        inference = ThaiSLMInference("./thai_slm_moe_model")
        
        print("=== Thai SLM MoE Model Testing ===\n")
        
        for prompt in test_prompts:
            print(f"Prompt: {prompt}")
            generated = inference.generate_text(
                prompt=prompt,
                max_length=100,
                temperature=0.8
            )
            print(f"Generated: {generated}")
            print("-" * 50)
            
    except Exception as e:
        logger.error(f"Error during testing: {e}")
        print("Model not found. Please train the model first using train.py")


def interactive_chat():
    """Interactive chat with the model"""
    
    try:
        inference = ThaiSLMInference("./thai_slm_moe_model")
        
        print("=== Thai SLM MoE Interactive Chat ===")
        print("Type 'quit' to exit")
        print("Type 'help' for commands")
        print()
        
        while True:
            prompt = input("You: ").strip()
            
            if prompt.lower() == 'quit':
                break
            elif prompt.lower() == 'help':
                print("Commands:")
                print("  quit - Exit chat")
                print("  help - Show this help")
                print("  temp:<value> - Set temperature (e.g., temp:0.8)")
                continue
            elif prompt.startswith('temp:'):
                try:
                    temp = float(prompt.split(':')[1])
                    print(f"Temperature set to {temp}")
                    continue
                except:
                    print("Invalid temperature value")
                    continue
            
            if prompt:
                response = inference.chat(prompt)
                print(f"Bot: {response}")
            print()
            
    except Exception as e:
        logger.error(f"Error during chat: {e}")
        print("Model not found. Please train the model first using train.py")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "chat":
        interactive_chat()
    else:
        test_model()

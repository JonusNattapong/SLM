import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import logging
import json
import os
from typing import Dict, Any

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThaiFinetunedInference:
    """Advanced inference for fine-tuned Thai language models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load model info
        self.load_model_info()
        self.load_model()
        
    def load_model_info(self):
        """Load training information"""
        info_path = os.path.join(self.model_path, "training_info.json")
        
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.model_info = json.load(f)
            logger.info(f"Loaded model info: {self.model_info}")
        else:
            self.model_info = {}
            logger.warning("No training info found")
            
    def load_model(self):
        """Load tokenizer and model"""
        
        logger.info(f"Loading model from {self.model_path}")
        
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                local_files_only=True
            )
            
            self.model.eval()
            
            logger.info(f"Model loaded successfully on {self.device}")
            logger.info(f"Vocabulary size: {len(self.tokenizer)}")
            
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
            
    def generate_thai_text(self, 
                          prompt: str, 
                          max_new_tokens: int = 100,
                          temperature: float = 0.8,
                          top_p: float = 0.9,
                          top_k: int = 50,
                          repetition_penalty: float = 1.1,
                          do_sample: bool = True) -> str:
        """Generate Thai text with advanced parameters"""
        
        # Prepare prompt with Thai formatting
        if not any(tag in prompt for tag in ["<thai>", "<question>", "<answer>"]):
            formatted_prompt = f"<thai>{prompt}"
        else:
            formatted_prompt = prompt
            
        logger.info(f"Generating text for: {formatted_prompt[:50]}...")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                formatted_prompt,
                return_tensors="pt",
                truncation=True,
                max_length=512
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                    do_sample=do_sample,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.eos_token_id,
                    early_stopping=True,
                    no_repeat_ngram_size=3
                )
                
            # Decode output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up output
            if generated_text.startswith(formatted_prompt):
                generated_text = generated_text[len(formatted_prompt):].strip()
                
            # Remove formatting tags
            for tag in ["<thai>", "</thai>", "<question>", "</question>", "<answer>", "</answer>"]:
                generated_text = generated_text.replace(tag, "")
                
            return generated_text.strip()
            
        except Exception as e:
            logger.error(f"Error during generation: {e}")
            return "เกิดข้อผิดพลาดในการสร้างข้อความ"
            
    def interactive_chat(self):
        """Interactive chat with the model"""
        
        print("\n🇹🇭 Thai Language Model - Interactive Chat")
        print("=" * 50)
        print("Type 'quit' to exit")
        print("Type 'settings' to adjust generation parameters")
        print("=" * 50)
        
        # Default settings
        settings = {
            "max_new_tokens": 100,
            "temperature": 0.8,
            "top_p": 0.9,
            "repetition_penalty": 1.1
        }
        
        while True:
            try:
                user_input = input("\nYou: ").strip()
                
                if user_input.lower() == 'quit':
                    print("ลาก่อน! (Goodbye!)")
                    break
                    
                elif user_input.lower() == 'settings':
                    print("\nCurrent settings:")
                    for key, value in settings.items():
                        print(f"  {key}: {value}")
                    print("\nTo change a setting, type: set <parameter> <value>")
                    continue
                    
                elif user_input.lower().startswith('set '):
                    parts = user_input.split()
                    if len(parts) == 3:
                        param, value = parts[1], parts[2]
                        if param in settings:
                            try:
                                if param == "max_new_tokens":
                                    settings[param] = int(value)
                                else:
                                    settings[param] = float(value)
                                print(f"Updated {param} to {settings[param]}")
                            except ValueError:
                                print("Invalid value")
                        else:
                            print(f"Unknown parameter: {param}")
                    continue
                    
                if not user_input:
                    continue
                    
                # Generate response
                response = self.generate_thai_text(
                    user_input,
                    **settings
                )
                
                print(f"\nBot: {response}")
                
            except KeyboardInterrupt:
                print("\n\nลาก่อน! (Goodbye!)")
                break
            except Exception as e:
                print(f"\nError: {e}")
                
    def benchmark_generation(self):
        """Benchmark the model with various prompts"""
        
        test_cases = [
            {
                "prompt": "ประเทศไทยมี",
                "description": "Geography/Basic Facts"
            },
            {
                "prompt": "วิทยาศาสตร์และเทคโนโลยี",
                "description": "Science & Technology"
            },
            {
                "prompt": "อาหารไทยที่มีชื่อเสียง",
                "description": "Thai Cuisine"
            },
            {
                "prompt": "ประวัติศาสตร์ของประเทศไทย",
                "description": "Thai History"
            },
            {
                "prompt": "วัฒนธรรมไทย",
                "description": "Thai Culture"
            },
            {
                "prompt": "<question>เศรษฐกิจไทยเป็นอย่างไร</question><answer>",
                "description": "Q&A Format - Economy"
            },
            {
                "prompt": "การศึกษาในประเทศไทย",
                "description": "Education"
            },
            {
                "prompt": "ภาษาไทยมีลักษณะ",
                "description": "Thai Language"
            }
        ]
        
        print("\n🧪 Model Benchmark Results")
        print("=" * 70)
        
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n📝 Test {i}: {test_case['description']}")
            print(f"Prompt: {test_case['prompt']}")
            print("-" * 50)
            
            try:
                result = self.generate_thai_text(
                    test_case['prompt'],
                    max_new_tokens=80,
                    temperature=0.7
                )
                print(f"Generated: {result}")
                
            except Exception as e:
                print(f"Error: {e}")
                
            print("-" * 50)
            
    def analyze_model_capabilities(self):
        """Analyze model capabilities and statistics"""
        
        print("\n📊 Model Analysis")
        print("=" * 50)
        
        # Model info
        if self.model_info:
            print("Training Information:")
            for key, value in self.model_info.items():
                print(f"  {key}: {value}")
            print()
            
        # Model statistics
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        print("Model Statistics:")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Trainable parameters: {trainable_params:,}")
        print(f"  Vocabulary size: {len(self.tokenizer)}")
        print(f"  Device: {self.device}")
        print()
        
        # Special tokens
        print("Special Tokens:")
        special_tokens = [
            ("pad_token", self.tokenizer.pad_token),
            ("eos_token", self.tokenizer.eos_token),
            ("bos_token", getattr(self.tokenizer, 'bos_token', None)),
            ("unk_token", self.tokenizer.unk_token),
        ]
        
        for name, token in special_tokens:
            if token:
                print(f"  {name}: {token} (ID: {self.tokenizer.convert_tokens_to_ids(token)})")
                
        # Test tokenization
        print("\nTokenization Test:")
        test_text = "สวัสดีครับ ผมชื่อโมเดลภาษาไทย"
        tokens = self.tokenizer.tokenize(test_text)
        print(f"  Text: {test_text}")
        print(f"  Tokens: {tokens}")
        print(f"  Token count: {len(tokens)}")


def main():
    """Main function for inference"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Thai Fine-tuned Model Inference")
    parser.add_argument("model_path", help="Path to fine-tuned model")
    parser.add_argument("--mode", choices=["chat", "benchmark", "analyze", "generate"], 
                       default="chat", help="Operation mode")
    parser.add_argument("--prompt", help="Prompt for generate mode")
    parser.add_argument("--max_tokens", type=int, default=100, 
                       help="Maximum tokens to generate")
    
    args = parser.parse_args()
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model path {args.model_path} does not exist")
        return
        
    # Load model
    try:
        inference = ThaiFinetunedInference(args.model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return
        
    # Run based on mode
    if args.mode == "chat":
        inference.interactive_chat()
        
    elif args.mode == "benchmark":
        inference.benchmark_generation()
        
    elif args.mode == "analyze":
        inference.analyze_model_capabilities()
        
    elif args.mode == "generate":
        if not args.prompt:
            print("Error: --prompt required for generate mode")
            return
            
        result = inference.generate_thai_text(
            args.prompt,
            max_new_tokens=args.max_tokens
        )
        print(f"Prompt: {args.prompt}")
        print(f"Generated: {result}")
        
    else:
        print(f"Unknown mode: {args.mode}")


if __name__ == "__main__":
    main()

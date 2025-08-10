import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    TrainingArguments, 
    Trainer,
    DataCollatorForLanguageModeling,
    GPT2LMHeadModel,
    GPT2Config
)
from datasets import load_dataset
import logging
import os
from typing import Dict, Any
import wandb
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import json

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThaiPretrainedFineTuner:
    """Fine-tune pre-trained models for Thai language"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 output_dir: str = "./thai_finetuned_model",
                 use_lora: bool = True):
        """
        Initialize with pre-trained model options:
        - microsoft/DialoGPT-medium (English, good for chat)
        - gpt2 (standard GPT-2)
        - distilgpt2 (smaller, faster)
        - facebook/opt-350m (Facebook OPT)
        """
        self.model_name = model_name
        self.output_dir = output_dir
        self.use_lora = use_lora
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        logger.info(f"Using model: {model_name}")
        logger.info(f"Device: {self.device}")
        logger.info(f"Using LoRA: {use_lora}")
        
    def load_model_and_tokenizer(self):
        """Load pre-trained model and tokenizer"""
        
        logger.info("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        
        # Add pad token if not exists
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Add Thai special tokens
        special_tokens = {
            "additional_special_tokens": [
                "<thai>", "</thai>", 
                "<question>", "</question>",
                "<answer>", "</answer>"
            ]
        }
        self.tokenizer.add_special_tokens(special_tokens)
        
        logger.info("Loading model...")
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            device_map="auto" if self.device == "cuda" else None
        )
        
        # Resize token embeddings for new tokens
        self.model.resize_token_embeddings(len(self.tokenizer))
        
        # Setup LoRA if enabled
        if self.use_lora:
            self.setup_lora()
            
        logger.info(f"Model loaded with {sum(p.numel() for p in self.model.parameters()):,} parameters")
        
    def setup_lora(self):
        """Setup LoRA (Low-Rank Adaptation) for efficient fine-tuning"""
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["c_attn", "c_proj", "c_fc"]  # GPT-2 style modules
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        
    def prepare_thai_dataset(self, max_samples: int = 5000):
        """Prepare Thai Wikipedia dataset for fine-tuning"""
        
        logger.info("Loading Thai Wikipedia dataset...")
        dataset = load_dataset("ZombitX64/Wikipedia-Thai", split="train", streaming=True)
        
        # Extract texts
        texts = []
        for i, item in enumerate(dataset):
            if i >= max_samples:
                break
            if len(item["text"]) > 100:  # Filter short texts
                # Format text for Thai language modeling
                formatted_text = f"<thai>{item['text']}</thai>"
                texts.append(formatted_text)
                
            if (i + 1) % 1000 == 0:
                logger.info(f"Processed {i + 1} texts")
                
        logger.info(f"Collected {len(texts)} Thai texts")
        
        # Tokenize texts
        logger.info("Tokenizing texts...")
        tokenized_texts = []
        
        for text in texts:
            tokens = self.tokenizer(
                text,
                truncation=True,
                max_length=512,
                padding=False,
                return_tensors=None
            )
            tokenized_texts.append(tokens)
            
        return tokenized_texts
        
    def create_trainer(self, train_dataset, eval_dataset=None):
        """Create Hugging Face Trainer"""
        # Data collator for language modeling
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,  # We're doing causal LM, not masked LM
            pad_to_multiple_of=8 if self.device == "cuda" else None
        )

        # Prepare arguments for TrainingArguments
        training_args_kwargs = dict(
            output_dir=self.output_dir,
            overwrite_output_dir=True,
            num_train_epochs=3,
            per_device_train_batch_size=4 if self.device == "cuda" else 2,
            per_device_eval_batch_size=4 if self.device == "cuda" else 2,
            gradient_accumulation_steps=8 if self.device == "cuda" else 16,
            learning_rate=5e-4 if self.use_lora else 1e-5,
            weight_decay=0.01,
            warmup_steps=500,
            logging_steps=100,
            save_steps=1000,
            save_total_limit=3,
            fp16=self.device == "cuda",
            dataloader_pin_memory=False,
            report_to="wandb" if os.getenv("WANDB_API_KEY") else None,
            run_name=f"thai-finetune-{self.model_name.split('/')[-1]}",
            remove_unused_columns=False,
        )

        # Only add eval_steps and evaluation_strategy if eval_dataset is provided
        if eval_dataset:
            try:
                # Try to add, but if error, fallback gracefully
                training_args_kwargs["eval_steps"] = 500
                training_args_kwargs["evaluation_strategy"] = "steps"
                training_args = TrainingArguments(**training_args_kwargs)
            except TypeError:
                # Remove unsupported keys for older transformers
                training_args_kwargs.pop("eval_steps", None)
                training_args_kwargs.pop("evaluation_strategy", None)
                training_args = TrainingArguments(**training_args_kwargs)
        else:
            # Don't add eval_steps/evaluation_strategy at all
            training_args = TrainingArguments(**training_args_kwargs)

        # Create trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )
        return trainer
        
    def fine_tune(self, max_samples: int = 5000):
        """Fine-tune the model on Thai data"""
        
        # Load model and tokenizer
        self.load_model_and_tokenizer()
        
        # Prepare dataset
        tokenized_texts = self.prepare_thai_dataset(max_samples)
        
        # Split into train/val
        split_idx = int(0.9 * len(tokenized_texts))
        train_dataset = tokenized_texts[:split_idx]
        val_dataset = tokenized_texts[split_idx:] if len(tokenized_texts) > 100 else None
        
        logger.info(f"Train samples: {len(train_dataset)}")
        if val_dataset:
            logger.info(f"Validation samples: {len(val_dataset)}")
            
        # Create trainer
        trainer = self.create_trainer(train_dataset, val_dataset)
        
        # Start training
        logger.info("Starting fine-tuning...")
        trainer.train()
        
        # Save final model
        logger.info("Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save training info
        training_info = {
            "base_model": self.model_name,
            "use_lora": self.use_lora,
            "max_samples": max_samples,
            "device": self.device,
            "vocab_size": len(self.tokenizer),
        }
        
        with open(os.path.join(self.output_dir, "training_info.json"), "w") as f:
            json.dump(training_info, f, indent=2)
            
        logger.info(f"Fine-tuning completed! Model saved to {self.output_dir}")
        
    def merge_and_save_lora(self, merged_output_dir: str = None):
        """Merge LoRA weights into base model and save"""
        
        if not self.use_lora:
            logger.warning("Model was not trained with LoRA, nothing to merge")
            return
            
        if merged_output_dir is None:
            merged_output_dir = self.output_dir + "_merged"
            
        logger.info("Merging LoRA weights into base model...")
        
        # Load the base model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        
        # Load LoRA model
        model_with_lora = PeftModel.from_pretrained(base_model, self.output_dir)
        
        # Merge LoRA weights
        merged_model = model_with_lora.merge_and_unload()
        
        # Save merged model
        os.makedirs(merged_output_dir, exist_ok=True)
        merged_model.save_pretrained(merged_output_dir)
        self.tokenizer.save_pretrained(merged_output_dir)
        
        # Save merged model info
        merged_info = {
            "base_model": self.model_name,
            "lora_merged": True,
            "original_lora_dir": self.output_dir,
            "device": self.device,
            "vocab_size": len(self.tokenizer),
        }
        
        with open(os.path.join(merged_output_dir, "training_info.json"), "w") as f:
            json.dump(merged_info, f, indent=2)
            
        logger.info(f"Merged model saved to {merged_output_dir}")
        return merged_output_dir
        
    def test_generation(self, prompts: list = None):
        """Test text generation with the fine-tuned model"""
        
        if prompts is None:
            prompts = [
                "<thai>ประเทศไทยมีจังหวัด",
                "<thai>วิทยาศาสตร์และเทคโนโลยี",
                "<thai>อาหารไทยที่มีชื่อเสียง",
                "<question>ประวัติศาสตร์ของไทย</question><answer>",
            ]
            
        logger.info("Testing generation...")
        
        for prompt in prompts:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_new_tokens=100,
                    temperature=0.8,
                    top_p=0.9,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
            logger.info(f"Prompt: {prompt}")
            logger.info(f"Generated: {generated_text}")
            logger.info("-" * 50)


class ThaiInferenceFromPretrained:
    """Inference class for fine-tuned Thai models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # Load training info
        info_path = os.path.join(model_path, "training_info.json")
        if os.path.exists(info_path):
            with open(info_path, "r") as f:
                self.training_info = json.load(f)
        else:
            self.training_info = {}
            
        self.load_model()
        
    def load_model(self):
        """Load the fine-tuned model"""
        
        logger.info(f"Loading model from {self.model_path}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        
        # Check if this is a merged LoRA model or regular model
        if self.training_info.get("lora_merged", False):
            # Load as regular model (LoRA already merged)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            logger.info("Loaded merged LoRA model")
            
        elif self.training_info.get("use_lora", False):
            # Load LoRA model
            base_model_name = self.training_info.get("base_model", "microsoft/DialoGPT-medium")
            base_model = AutoModelForCausalLM.from_pretrained(
                base_model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            
            # Resize embeddings for new tokens
            base_model.resize_token_embeddings(len(self.tokenizer))
            
            # Load LoRA weights
            self.model = PeftModel.from_pretrained(base_model, self.model_path)
            logger.info("Loaded LoRA model")
            
        else:
            # Load as regular fine-tuned model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None
            )
            logger.info("Loaded regular fine-tuned model")
        
        self.model.eval()
        logger.info("Model loaded successfully")
        
    def generate_text(self, prompt: str, max_length: int = 150, **kwargs):
        """Generate text from prompt"""
        
        # Add Thai formatting if not present
        if not prompt.startswith("<thai>"):
            prompt = f"<thai>{prompt}"
            
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_new_tokens=max_length,
                temperature=kwargs.get("temperature", 0.8),
                top_p=kwargs.get("top_p", 0.9),
                top_k=kwargs.get("top_k", 50),
                do_sample=kwargs.get("do_sample", True),
                pad_token_id=self.tokenizer.eos_token_id,
                repetition_penalty=kwargs.get("repetition_penalty", 1.1)
            )
            
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Remove input prompt
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
            
        # Clean up Thai tags
        generated_text = generated_text.replace("<thai>", "").replace("</thai>", "")
        
        return generated_text


def merge_lora_weights(lora_model_dir: str, output_dir: str = None):
    """Standalone function to merge LoRA weights"""
    
    if output_dir is None:
        output_dir = lora_model_dir + "_merged"
    
    # Load training info
    info_path = os.path.join(lora_model_dir, "training_info.json")
    if not os.path.exists(info_path):
        raise FileNotFoundError(f"Training info not found at {info_path}")
        
    with open(info_path, "r") as f:
        training_info = json.load(f)
    
    if not training_info.get("use_lora", False):
        raise ValueError("Model was not trained with LoRA")
        
    base_model_name = training_info.get("base_model")
    if not base_model_name:
        raise ValueError("Base model name not found in training info")
    
    logger.info(f"Loading base model: {base_model_name}")
    base_model = AutoModelForCausalLM.from_pretrained(base_model_name)
    
    logger.info(f"Loading LoRA model from: {lora_model_dir}")
    model_with_lora = PeftModel.from_pretrained(base_model, lora_model_dir)
    
    logger.info("Merging LoRA weights...")
    merged_model = model_with_lora.merge_and_unload()
    
    # Save merged model
    os.makedirs(output_dir, exist_ok=True)
    merged_model.save_pretrained(output_dir)
    
    # Load and save tokenizer
    tokenizer = AutoTokenizer.from_pretrained(lora_model_dir)
    tokenizer.save_pretrained(output_dir)
    
    # Update training info
    merged_info = training_info.copy()
    merged_info["lora_merged"] = True
    merged_info["original_lora_dir"] = lora_model_dir
    
    with open(os.path.join(output_dir, "training_info.json"), "w") as f:
        json.dump(merged_info, f, indent=2)
    
    logger.info(f"Merged model saved to: {output_dir}")
    return output_dir


def main():
    """Main function to run fine-tuning"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="Fine-tune pre-trained model for Thai")
    parser.add_argument("--model", default="microsoft/DialoGPT-medium", 
                       help="Pre-trained model to use")
    parser.add_argument("--samples", type=int, default=5000,
                       help="Number of training samples")
    parser.add_argument("--output_dir", default="./thai_finetuned_model",
                       help="Output directory")
    parser.add_argument("--use_lora", action="store_true", default=True,
                       help="Use LoRA for efficient fine-tuning")
    parser.add_argument("--test_only", action="store_true",
                       help="Only test existing model")
    parser.add_argument("--merge_lora", action="store_true",
                       help="Merge LoRA weights into base model")
    parser.add_argument("--lora_dir", default=None,
                       help="LoRA model directory to merge (if different from output_dir)")
    
    args = parser.parse_args()
    
    if args.merge_lora:
        # Merge LoRA weights
        lora_dir = args.lora_dir or args.output_dir
        merged_dir = merge_lora_weights(lora_dir, lora_dir + "_merged")
        print(f"LoRA weights merged and saved to: {merged_dir}")
        
    elif args.test_only:
        # Test existing model
        if os.path.exists(args.output_dir):
            inference = ThaiInferenceFromPretrained(args.output_dir)
            
            test_prompts = [
                "ประเทศไทยมีจังหวัด",
                "วิทยาศาสตร์และเทคโนโลยี", 
                "อาหารไทยที่มีชื่อเสียง",
                "ประวัติศาสตร์ของไทย"
            ]
            
            for prompt in test_prompts:
                result = inference.generate_text(prompt, max_length=100)
                print(f"Prompt: {prompt}")
                print(f"Generated: {result}")
                print("-" * 50)
        else:
            print(f"Model not found at {args.output_dir}")
            
    else:
        # Fine-tune model
        fine_tuner = ThaiPretrainedFineTuner(
            model_name=args.model,
            output_dir=args.output_dir,
            use_lora=args.use_lora
        )
        
        fine_tuner.fine_tune(max_samples=args.samples)
        fine_tuner.test_generation()
        
        # Optionally merge LoRA weights after training
        if args.use_lora:
            print("\nDo you want to merge LoRA weights? (y/n): ", end="")
            response = input().strip().lower()
            if response in ['y', 'yes']:
                merged_dir = fine_tuner.merge_and_save_lora()
                print(f"Merged model saved to: {merged_dir}")


if __name__ == "__main__":
    main()

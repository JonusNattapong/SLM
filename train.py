import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import wandb
import os
import json
import logging
from datetime import datetime
from tqdm import tqdm
import numpy as np
from typing import Dict, Any, Optional

from model import SLMForCausalLM, SLMConfig
from dataset import ThaiDatasetPreprocessor, prepare_thai_dataset


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextDataset(Dataset):
    """Custom dataset for text data"""
    
    def __init__(self, data):
        self.data = data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]


class SLMTrainer:
    """Trainer class for Small Language Model"""
    
    def __init__(self, 
                 model: SLMForCausalLM,
                 config: SLMConfig,
                 tokenizer,
                 train_dataset: Dataset,
                 val_dataset: Optional[Dataset] = None,
                 output_dir: str = "./slm_thai_model"):
        
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.output_dir = output_dir
        
        # Training parameters
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save config
        with open(os.path.join(output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(config.__dict__, f, indent=2, ensure_ascii=False)
    
    def create_data_collator(self):
        """Create data collator for padding"""
        def collate_fn(batch):
            # Get max length in batch
            max_len = max(len(item['input_ids']) for item in batch)
            
            batch_input_ids = []
            batch_attention_mask = []
            batch_labels = []
            
            for item in batch:
                input_ids = item['input_ids']
                attention_mask = item['attention_mask']
                labels = item['labels']
                
                # Pad to max length
                pad_length = max_len - len(input_ids)
                
                input_ids = torch.cat([
                    input_ids, 
                    torch.full((pad_length,), self.tokenizer.pad_token_id)
                ])
                attention_mask = torch.cat([
                    attention_mask,
                    torch.zeros(pad_length)
                ])
                labels = torch.cat([
                    labels,
                    torch.full((pad_length,), -100)
                ])
                
                batch_input_ids.append(input_ids)
                batch_attention_mask.append(attention_mask)
                batch_labels.append(labels)
            
            return {
                'input_ids': torch.stack(batch_input_ids),
                'attention_mask': torch.stack(batch_attention_mask),
                'labels': torch.stack(batch_labels)
            }
        
        return collate_fn
    
    def train(self,
              num_epochs: int = 3,
              batch_size: int = 8,
              learning_rate: float = 5e-4,
              warmup_steps: int = 1000,
              gradient_accumulation_steps: int = 4,
              max_grad_norm: float = 1.0,
              save_steps: int = 1000,
              eval_steps: int = 500,
              logging_steps: int = 100,
              use_wandb: bool = True):
        """Train the model"""
        
        # Initialize wandb
        if use_wandb:
            wandb.init(
                project="thai-slm-moe",
                config={
                    "model_config": self.config.__dict__,
                    "num_epochs": num_epochs,
                    "batch_size": batch_size,
                    "learning_rate": learning_rate,
                    "gradient_accumulation_steps": gradient_accumulation_steps
                }
            )
        
        # Create data loaders with device-appropriate settings
        device_type = self.device.type
        num_workers = 0 if device_type == "cpu" else 2  # No multiprocessing on CPU to avoid overhead
        
        train_dataloader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.create_data_collator(),
            num_workers=num_workers,
            pin_memory=False if device_type == "cpu" else True
        )
        
        val_dataloader = None
        if self.val_dataset:
            val_dataloader = DataLoader(
                self.val_dataset,
                batch_size=batch_size,
                shuffle=False,
                collate_fn=self.create_data_collator(),
                num_workers=num_workers,
                pin_memory=False if device_type == "cpu" else True
            )
        
        # Setup optimizer and scheduler
        optimizer = AdamW(
            self.model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            weight_decay=0.1
        )
        
        total_steps = len(train_dataloader) * num_epochs // gradient_accumulation_steps
        scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)
        
        # Training loop
        global_step = 0
        total_loss = 0
        
        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Total training steps: {total_steps}")
        logger.info(f"Device: {self.device}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                # Move batch to device
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                # Forward pass
                outputs = self.model(**batch)
                loss = outputs['loss']
                
                # Scale loss for gradient accumulation
                loss = loss / gradient_accumulation_steps
                
                # Backward pass
                loss.backward()
                
                total_loss += loss.item()
                epoch_loss += loss.item()
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    # Clip gradients
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_grad_norm)
                    
                    # Update weights
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                    
                    global_step += 1
                    
                    # Logging
                    if global_step % logging_steps == 0:
                        avg_loss = total_loss / logging_steps
                        lr = scheduler.get_last_lr()[0]
                        
                        logger.info(f"Step {global_step}: loss={avg_loss:.4f}, lr={lr:.2e}")
                        
                        if use_wandb:
                            wandb.log({
                                "train/loss": avg_loss,
                                "train/learning_rate": lr,
                                "train/step": global_step
                            })
                        
                        total_loss = 0
                    
                    # Evaluation
                    if val_dataloader and global_step % eval_steps == 0:
                        eval_loss = self.evaluate(val_dataloader)
                        logger.info(f"Eval loss: {eval_loss:.4f}")
                        
                        if use_wandb:
                            wandb.log({
                                "eval/loss": eval_loss,
                                "train/step": global_step
                            })
                        
                        self.model.train()
                    
                    # Save checkpoint
                    if global_step % save_steps == 0:
                        self.save_checkpoint(global_step)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f"{epoch_loss / (step + 1):.4f}",
                    'lr': f"{scheduler.get_last_lr()[0]:.2e}"
                })
            
            # Save epoch checkpoint
            self.save_checkpoint(f"epoch_{epoch+1}")
        
        # Final save
        self.save_model()
        
        if use_wandb:
            wandb.finish()
    
    def evaluate(self, dataloader):
        """Evaluate the model"""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Evaluating"):
                batch = {k: v.to(self.device) for k, v in batch.items()}
                outputs = self.model(**batch)
                total_loss += outputs['loss'].item()
                num_batches += 1
        
        return total_loss / num_batches
    
    def save_checkpoint(self, step):
        """Save training checkpoint"""
        checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{step}")
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(checkpoint_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(checkpoint_dir)
        
        # Save config
        with open(os.path.join(checkpoint_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Checkpoint saved to {checkpoint_dir}")
    
    def save_model(self):
        """Save final model"""
        # Save model state
        torch.save(self.model.state_dict(), os.path.join(self.output_dir, "pytorch_model.bin"))
        
        # Save tokenizer
        self.tokenizer.save_pretrained(self.output_dir)
        
        # Save config
        with open(os.path.join(self.output_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(self.config.__dict__, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Model saved to {self.output_dir}")


def train_thai_slm():
    """Main training function"""
    
    # Check if GPU is available and adjust config accordingly
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Training on: {device}")
    
    # Prepare dataset first to get tokenizer info
    logger.info("Preparing dataset...")
    
    # Adjust dataset size based on device
    if device.type == "cpu":
        max_samples = 2000  # Smaller dataset for CPU
        max_length = 256   # Shorter sequences for CPU
        vocab_size = 20000  # Smaller vocab for CPU
        logger.info("Using smaller dataset for CPU training")
    else:
        max_samples = 10000  # Full dataset for GPU
        max_length = 512
        vocab_size = 30000  # Full vocab for GPU
        logger.info("Using full dataset for GPU training")
    
    preprocessor, training_data = prepare_thai_dataset(
        max_samples=max_samples,
        vocab_size=vocab_size,
        max_length=max_length
    )
    
    # Get actual tokenizer vocabulary size
    actual_vocab_size = len(preprocessor.tokenizer)
    logger.info(f"Actual tokenizer vocabulary size: {actual_vocab_size}")
    
    # Configuration - Use actual vocab size from tokenizer
    if device.type == "cpu":
        # Smaller model for CPU training with advanced features
        config = SLMConfig(
            vocab_size=actual_vocab_size,  # Use actual tokenizer vocab size
            hidden_size=512,
            num_hidden_layers=8,
            num_attention_heads=8,
            num_kv_heads=4,  # GQA with fewer KV heads
            intermediate_size=2048,
            max_position_embeddings=512,
            num_experts=4,
            num_experts_per_token=2,
            aux_loss_alpha=0.01,
            router_z_loss_alpha=0.001,
            # Advanced features
            use_flash_attention=True,
            use_dynamic_capacity=True,
            layer_drop_prob=0.05,  # Light stochastic depth for CPU
            use_gradient_checkpointing=False,  # Disable for CPU
            rope_scaling_ntk=True,
            rope_scaling_alpha=1.5
        )
        logger.info("Using enhanced CPU-optimized configuration")
    else:
        # Full model for GPU training with all advanced features
        config = SLMConfig(
            vocab_size=actual_vocab_size,  # Use actual tokenizer vocab size
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            num_kv_heads=6,  # GQA with half KV heads
            intermediate_size=3072,
            max_position_embeddings=1024,
            num_experts=8,
            num_experts_per_token=2,
            aux_loss_alpha=0.01,
            router_z_loss_alpha=0.001,
            # Advanced features
            use_flash_attention=True,
            use_dynamic_capacity=True,
            layer_drop_prob=0.1,  # 10% stochastic depth
            use_gradient_checkpointing=True,  # Enable for GPU memory efficiency
            rope_scaling_ntk=True,
            rope_scaling_alpha=2.0
        )
        logger.info("Using enhanced GPU-optimized configuration")
    
    # Split data
    train_size = int(0.9 * len(training_data))
    train_data = training_data[:train_size]
    val_data = training_data[train_size:]
    
    train_dataset = TextDataset(train_data)
    val_dataset = TextDataset(val_data)
    
    logger.info(f"Train size: {len(train_dataset)}, Val size: {len(val_dataset)}")
    
    # Initialize model
    model = SLMForCausalLM(config)
    
    # Validate tokenizer compatibility
    logger.info("Validating tokenizer-model compatibility...")
    test_tokens = preprocessor.tokenizer("ทดสอบ", return_tensors="pt")
    test_input_ids = test_tokens["input_ids"]
    max_token_id = test_input_ids.max().item()
    
    logger.info(f"Model vocab size: {config.vocab_size}")
    logger.info(f"Tokenizer vocab size: {len(preprocessor.tokenizer)}")
    logger.info(f"Sample max token ID: {max_token_id}")
    
    if max_token_id >= config.vocab_size:
        logger.error(f"Token ID {max_token_id} >= model vocab size {config.vocab_size}")
        raise ValueError("Tokenizer produces token IDs outside model vocabulary range")
    
    # Test a forward pass to ensure compatibility
    try:
        with torch.no_grad():
            test_output = model(test_input_ids)
        logger.info("✅ Model-tokenizer compatibility validated")
    except Exception as e:
        logger.error(f"Model-tokenizer compatibility test failed: {e}")
        raise
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"Total parameters: {total_params:,}")
    logger.info(f"Trainable parameters: {trainable_params:,}")
    
    # Initialize trainer
    trainer = SLMTrainer(
        model=model,
        config=config,
        tokenizer=preprocessor.tokenizer,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        output_dir="./thai_slm_moe_model"
    )
    
    # Train model with device-appropriate settings
    if device.type == "cpu":
        # CPU-optimized training parameters
        trainer.train(
            num_epochs=2,  # Fewer epochs for CPU
            batch_size=2,  # Smaller batch size for CPU
            learning_rate=5e-4,  # Slightly higher LR for faster convergence
            gradient_accumulation_steps=16,  # More accumulation for effective batch size
            save_steps=100,  # Save more frequently
            eval_steps=50,   # Evaluate more frequently
            use_wandb=True
        )
        logger.info("Training completed with CPU-optimized settings")
    else:
        # GPU training parameters
        trainer.train(
            num_epochs=3,
            batch_size=4,
            learning_rate=1e-4,
            gradient_accumulation_steps=8,
            save_steps=500,
            eval_steps=250,
            use_wandb=True
        )
        logger.info("Training completed with GPU settings")


if __name__ == "__main__":
    train_thai_slm()

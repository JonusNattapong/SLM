import torch
from datasets import load_dataset
from tokenizers import ByteLevelBPETokenizer
from transformers import PreTrainedTokenizerFast
import os
import json
from typing import List, Dict, Any
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ThaiDatasetPreprocessor:
    """Preprocessor for Thai Wikipedia dataset"""
    
    def __init__(self, dataset_name: str = "ZombitX64/Wikipedia-Thai"):
        self.dataset_name = dataset_name
        self.tokenizer = None
        
    def load_dataset(self, split: str = "train", streaming: bool = False):
        """Load the Thai Wikipedia dataset"""
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            dataset = load_dataset(self.dataset_name, split=split, streaming=streaming)
            logger.info(f"Dataset loaded successfully. Size: {len(dataset) if not streaming else 'streaming'}")
            return dataset
        except Exception as e:
            logger.error(f"Error loading dataset: {e}")
            raise
    
    def create_tokenizer(self, 
                        texts: List[str], 
                        vocab_size: int = 50000,
                        min_frequency: int = 2,
                        special_tokens: List[str] = None) -> PreTrainedTokenizerFast:
        """Create and train a Thai tokenizer"""
        
        if special_tokens is None:
            special_tokens = [
                "<pad>", "<unk>", "<s>", "</s>", "<mask>",
                "<cls>", "<sep>", "<newline>", "<tab>"
            ]
        
        logger.info(f"Training tokenizer with vocab size: {vocab_size}")
        
        # Initialize ByteLevelBPE tokenizer
        tokenizer = ByteLevelBPETokenizer()
        
        # Prepare text iterator with error handling
        def text_iterator():
            for text in texts:
                if text and isinstance(text, str) and len(text.strip()) > 0:
                    yield text.strip()
        
        try:
            # Train tokenizer on texts
            tokenizer.train_from_iterator(
                text_iterator(),
                vocab_size=vocab_size,
                min_frequency=min_frequency,
                special_tokens=special_tokens,
                show_progress=True
            )
        except Exception as e:
            logger.error(f"Error training tokenizer: {e}")
            logger.info("Trying with fallback parameters...")
            
            # Fallback with more conservative settings
            tokenizer.train_from_iterator(
                text_iterator(),
                vocab_size=min(vocab_size, 25000),  # Smaller vocab
                min_frequency=1,  # Lower frequency threshold
                special_tokens=special_tokens,
                show_progress=True
            )
        
        # Convert to HuggingFace tokenizer with better compatibility
        try:
            tokenizer_hf = PreTrainedTokenizerFast(
                tokenizer_object=tokenizer,
                pad_token="<pad>",
                unk_token="<unk>",
                bos_token="<s>",
                eos_token="</s>",
                mask_token="<mask>",
                cls_token="<cls>",
                sep_token="<sep>",
                clean_up_tokenization_spaces=True,
                model_max_length=512
            )
        except Exception as e:
            logger.error(f"Error creating HuggingFace tokenizer: {e}")
            raise
        
        self.tokenizer = tokenizer_hf
        logger.info("Tokenizer training completed")
        return tokenizer_hf
    
    def save_tokenizer(self, tokenizer_path: str):
        """Save the trained tokenizer"""
        if self.tokenizer is None:
            raise ValueError("No tokenizer to save. Train tokenizer first.")
        
        os.makedirs(tokenizer_path, exist_ok=True)
        self.tokenizer.save_pretrained(tokenizer_path)
        logger.info(f"Tokenizer saved to: {tokenizer_path}")
    
    def load_tokenizer(self, tokenizer_path: str):
        """Load a pre-trained tokenizer"""
        self.tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_path)
        logger.info(f"Tokenizer loaded from: {tokenizer_path}")
        return self.tokenizer
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess Thai text"""
        if not text or not isinstance(text, str):
            return ""
        
        # Remove excessive whitespace
        text = " ".join(text.split())
        
        # Remove very short lines (likely noise)
        lines = text.split('\n')
        lines = [line.strip() for line in lines if len(line.strip()) > 10]
        text = '\n'.join(lines)
        
        return text.strip()
    
    def extract_texts_from_dataset(self, dataset, text_column: str = "text", max_samples: int = None):
        """Extract and clean texts from dataset"""
        logger.info("Extracting texts from dataset...")
        
        texts = []
        count = 0
        
        for item in dataset:
            if max_samples and count >= max_samples:
                break
                
            text = item.get(text_column, "")
            cleaned_text = self.clean_text(text)
            
            if cleaned_text:
                texts.append(cleaned_text)
                count += 1
                
                if count % 1000 == 0:
                    logger.info(f"Processed {count} texts")
        
        logger.info(f"Extracted {len(texts)} texts")
        return texts
    
    def prepare_training_data(self, 
                            texts: List[str], 
                            max_length: int = 512,
                            stride: int = 256) -> List[Dict[str, Any]]:
        """Prepare training data with tokenization"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not available. Train or load tokenizer first.")
        
        logger.info("Preparing training data...")
        
        training_data = []
        
        for i, text in enumerate(texts):
            # Split long texts into chunks manually to avoid tokenizer issues
            text_chunks = self._split_text_into_chunks(text, max_length, stride)
            
            for chunk in text_chunks:
                if len(chunk.strip()) < 20:  # Skip very short chunks
                    continue
                
                try:
                    # Tokenize each chunk separately
                    tokens = self.tokenizer(
                        chunk,
                        truncation=True,
                        max_length=max_length,
                        padding=False,
                        return_tensors="pt"
                    )
                    
                    input_ids = tokens['input_ids'].squeeze()
                    attention_mask = tokens['attention_mask'].squeeze()
                    
                    # Ensure we have valid tensors
                    if input_ids.dim() == 0:
                        input_ids = input_ids.unsqueeze(0)
                    if attention_mask.dim() == 0:
                        attention_mask = attention_mask.unsqueeze(0)
                    
                    if len(input_ids) > 10:  # Skip very short sequences
                        training_data.append({
                            'input_ids': input_ids,
                            'attention_mask': attention_mask,
                            'labels': input_ids.clone()  # For causal LM, labels = input_ids
                        })
                
                except Exception as e:
                    logger.warning(f"Error tokenizing chunk: {e}")
                    continue
            
            if (i + 1) % 100 == 0:
                logger.info(f"Processed {i + 1}/{len(texts)} texts")
        
        logger.info(f"Prepared {len(training_data)} training examples")
        return training_data
    
    def _split_text_into_chunks(self, text: str, max_length: int, stride: int) -> List[str]:
        """Split text into overlapping chunks"""
        words = text.split()
        chunks = []
        
        # Estimate words per chunk (rough approximation)
        words_per_chunk = max_length // 2  # Conservative estimate
        
        i = 0
        while i < len(words):
            chunk_words = words[i:i + words_per_chunk]
            chunk = ' '.join(chunk_words)
            chunks.append(chunk)
            
            # Move by stride (in words)
            stride_words = stride // 2
            i += max(stride_words, 1)
            
            # Break if we've covered all words
            if i >= len(words):
                break
        
        return chunks
    
    def create_data_collator(self, tokenizer):
        """Create data collator for training"""
        from torch.nn.utils.rnn import pad_sequence
        
        def collate_fn(batch):
            input_ids = [item['input_ids'] for item in batch]
            attention_masks = [item['attention_mask'] for item in batch]
            labels = [item['labels'] for item in batch]
            
            # Pad sequences
            input_ids = pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
            attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)
            labels = pad_sequence(labels, batch_first=True, padding_value=-100)
            
            return {
                'input_ids': input_ids,
                'attention_mask': attention_masks,
                'labels': labels
            }
        
        return collate_fn


def prepare_thai_dataset(
    dataset_name: str = "ZombitX64/Wikipedia-Thai",
    tokenizer_path: str = "./thai_tokenizer",
    max_samples: int = 10000,
    vocab_size: int = 50000,
    max_length: int = 512
):
    """Main function to prepare Thai dataset"""
    
    # Initialize preprocessor
    preprocessor = ThaiDatasetPreprocessor(dataset_name)
    
    # Check if tokenizer already exists
    if os.path.exists(tokenizer_path) and os.path.exists(os.path.join(tokenizer_path, "tokenizer.json")):
        logger.info(f"Loading existing tokenizer from {tokenizer_path}")
        try:
            preprocessor.load_tokenizer(tokenizer_path)
        except Exception as e:
            logger.warning(f"Failed to load existing tokenizer: {e}")
            logger.info("Creating new tokenizer...")
            preprocessor.tokenizer = None
    
    # Load dataset
    try:
        dataset = preprocessor.load_dataset(streaming=True)
    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.info("Using fallback sample data...")
        # Create some sample Thai text for testing
        sample_texts = [
            "ประเทศไทยมีจังหวัดทั้งหมด 77 จังหวัด แบ่งออกเป็นภูมิภาคต่างๆ",
            "วิทยาศาสตร์และเทคโนโลยีมีบทบาทสำคัญในการพัฒนาประเทศ",
            "การศึกษาเป็นรากฐานที่สำคัญของการพัฒนาคุณภาพชีวิต",
            "อาหารไทยมีเอกลักษณ์เฉพาะตัวและเป็นที่รู้จักทั่วโลก",
            "ประวัติศาสตร์ไทยมีความยาวนานและมีเหตุการณ์สำคัญมากมาย"
        ]
        texts = sample_texts * (max_samples // len(sample_texts) + 1)
        texts = texts[:max_samples]
    else:
        # Extract texts
        texts = preprocessor.extract_texts_from_dataset(dataset, max_samples=max_samples)
    
    # Validate texts
    if not texts:
        raise ValueError("No texts extracted from dataset")
    
    logger.info(f"Working with {len(texts)} texts")
    
    # Create and train tokenizer if not already available
    if preprocessor.tokenizer is None:
        tokenizer = preprocessor.create_tokenizer(texts, vocab_size=vocab_size)
        
        # Save tokenizer
        preprocessor.save_tokenizer(tokenizer_path)
        
        # Test tokenizer
        test_text = texts[0][:100] if texts else "ทดสอบ tokenizer"
        test_tokens = preprocessor.tokenizer(test_text)
        logger.info(f"Tokenizer test - Input: {test_text[:50]}...")
        logger.info(f"Tokenizer test - Tokens: {len(test_tokens['input_ids'])}")
    
    # Validate tokenizer vocabulary size
    actual_vocab_size = len(preprocessor.tokenizer)
    logger.info(f"Requested vocab size: {vocab_size}")
    logger.info(f"Actual vocab size: {actual_vocab_size}")
    
    if abs(actual_vocab_size - vocab_size) > 1000:  # Allow some tolerance
        logger.warning(f"Large vocab size difference: requested {vocab_size}, got {actual_vocab_size}")
    
    # Test for out-of-range tokens
    test_text = texts[0] if texts else "ทดสอบการทำงานของ tokenizer"
    test_tokens = preprocessor.tokenizer(test_text, return_tensors="pt")
    max_token_id = test_tokens["input_ids"].max().item()
    
    if max_token_id >= actual_vocab_size:
        logger.error(f"Found token ID {max_token_id} >= vocab size {actual_vocab_size}")
        raise ValueError("Tokenizer producing invalid token IDs")
    
    logger.info(f"✅ Tokenizer validation passed. Max token ID: {max_token_id}")
    
    # Prepare training data
    training_data = preprocessor.prepare_training_data(texts, max_length=max_length)
    
    # Validate training data
    if not training_data:
        raise ValueError("No training data prepared")
    
    logger.info(f"Training data statistics:")
    logger.info(f"  Total examples: {len(training_data)}")
    
    # Calculate statistics
    lengths = [len(item['input_ids']) for item in training_data]
    if lengths:
        logger.info(f"  Average length: {sum(lengths) / len(lengths):.1f}")
        logger.info(f"  Min length: {min(lengths)}")
        logger.info(f"  Max length: {max(lengths)}")
    
    # Save training data
    try:
        torch.save(training_data, "training_data.pt")
        logger.info("Training data saved to training_data.pt")
    except Exception as e:
        logger.error(f"Failed to save training data: {e}")
        raise
    
    return preprocessor, training_data


if __name__ == "__main__":
    # Example usage with error handling
    try:
        logger.info("Starting dataset preparation...")
        
        preprocessor, training_data = prepare_thai_dataset(
            max_samples=2000,  # Smaller for testing
            vocab_size=20000,  # Smaller vocab for faster training
            max_length=256     # Shorter sequences
        )
        
        print(f"\n✅ Dataset preparation successful!")
        print(f"📊 Number of training examples: {len(training_data)}")
        print(f"🔤 Tokenizer vocab size: {len(preprocessor.tokenizer)}")
        
        # Show example
        if training_data:
            example = training_data[0]
            print(f"📝 Example input shape: {example['input_ids'].shape}")
            try:
                decoded_text = preprocessor.tokenizer.decode(example['input_ids'][:30])
                print(f"📄 Example text: {decoded_text}...")
            except Exception as e:
                print(f"⚠️ Could not decode example: {e}")
        
        print(f"\n🎯 Ready for training!")
        print(f"📁 Files created:")
        print(f"   - training_data.pt ({os.path.getsize('training_data.pt') // (1024*1024)} MB)")
        print(f"   - ./thai_tokenizer/ (tokenizer files)")
        
    except Exception as e:
        logger.error(f"Dataset preparation failed: {e}")
        print(f"\n❌ Dataset preparation failed!")
        print(f"Error: {e}")
        print(f"\n🔧 Troubleshooting:")
        print(f"1. Check internet connection")
        print(f"2. Verify dataset availability")
        print(f"3. Try reducing max_samples")
        print(f"4. Check available disk space")
        
        import traceback
        traceback.print_exc()

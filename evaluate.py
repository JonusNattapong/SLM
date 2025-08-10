import torch
import json
import numpy as np
from typing import List, Dict, Any
import logging
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from inference import ThaiSLMInference
from dataset import ThaiDatasetPreprocessor
import os


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ModelEvaluator:
    """Evaluate Thai SLM model performance"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        try:
            self.inference = ThaiSLMInference(model_path)
            self.model_loaded = True
            logger.info("Model loaded successfully for evaluation")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self.model_loaded = False
    
    def evaluate_perplexity(self, test_texts: List[str], max_length: int = 512) -> float:
        """Calculate perplexity on test texts"""
        if not self.model_loaded:
            return float('inf')
        
        total_loss = 0
        total_tokens = 0
        
        logger.info("Calculating perplexity...")
        
        for text in tqdm(test_texts, desc="Computing perplexity"):
            # Tokenize
            inputs = self.inference.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=max_length
            )
            
            input_ids = inputs["input_ids"].to(self.inference.device)
            
            if input_ids.size(1) < 2:
                continue
            
            with torch.no_grad():
                # Forward pass
                outputs = self.inference.model(input_ids=input_ids, labels=input_ids)
                loss = outputs['loss']
                
                total_loss += loss.item() * input_ids.size(1)
                total_tokens += input_ids.size(1)
        
        if total_tokens == 0:
            return float('inf')
        
        avg_loss = total_loss / total_tokens
        perplexity = torch.exp(torch.tensor(avg_loss)).item()
        
        logger.info(f"Perplexity: {perplexity:.2f}")
        return perplexity
    
    def evaluate_generation_quality(self, prompts: List[str]) -> Dict[str, Any]:
        """Evaluate generation quality metrics"""
        if not self.model_loaded:
            return {}
        
        results = {
            'prompts': [],
            'generations': [],
            'lengths': [],
            'repetition_scores': [],
            'diversity_scores': []
        }
        
        logger.info("Evaluating generation quality...")
        
        for prompt in tqdm(prompts, desc="Generating texts"):
            # Generate text
            generated = self.inference.generate_text(
                prompt=prompt,
                max_length=100,
                temperature=0.8,
                do_sample=True
            )
            
            results['prompts'].append(prompt)
            results['generations'].append(generated)
            results['lengths'].append(len(generated.split()))
            
            # Calculate repetition score
            words = generated.split()
            unique_words = set(words)
            repetition_score = 1 - (len(unique_words) / len(words)) if words else 0
            results['repetition_scores'].append(repetition_score)
        
        # Calculate diversity across all generations
        all_words = []
        for gen in results['generations']:
            all_words.extend(gen.split())
        
        if all_words:
            unique_ratio = len(set(all_words)) / len(all_words)
            results['overall_diversity'] = unique_ratio
        else:
            results['overall_diversity'] = 0
        
        return results
    
    def evaluate_thai_language_understanding(self) -> Dict[str, Any]:
        """Evaluate Thai language specific understanding"""
        if not self.model_loaded:
            return {}
        
        # Thai language test cases
        test_cases = [
            {
                'prompt': 'ประเทศไทยตั้งอยู่ในทวีป',
                'expected_keywords': ['เอเชีย', 'เอเซีย', 'asia'],
                'category': 'geography'
            },
            {
                'prompt': 'เมืองหลวงของประเทศไทยคือ',
                'expected_keywords': ['กรุงเทพ', 'bangkok', 'กทม'],
                'category': 'geography'
            },
            {
                'prompt': 'อาหารไทยที่มีชื่อเสียงได้แก่',
                'expected_keywords': ['ผ้าไทย', 'ต้มยำ', 'แกงเขียวหวาน', 'ข้าวผ่าด'],
                'category': 'culture'
            },
            {
                'prompt': 'ภาษาไทยเป็นภาษา',
                'expected_keywords': ['ราชการ', 'หลัก', 'ประจำชาติ'],
                'category': 'language'
            },
            {
                'prompt': 'วันชาติไทยตรงกับวันที่',
                'expected_keywords': ['5', 'ธันวาคม', 'december'],
                'category': 'history'
            }
        ]
        
        results = {
            'test_cases': [],
            'scores': [],
            'category_scores': {}
        }
        
        logger.info("Evaluating Thai language understanding...")
        
        for test_case in tqdm(test_cases, desc="Testing Thai understanding"):
            generated = self.inference.generate_text(
                prompt=test_case['prompt'],
                max_length=50,
                temperature=0.5
            )
            
            # Check if any expected keywords appear
            score = 0
            found_keywords = []
            for keyword in test_case['expected_keywords']:
                if keyword.lower() in generated.lower():
                    score += 1
                    found_keywords.append(keyword)
            
            score = score / len(test_case['expected_keywords'])
            
            results['test_cases'].append({
                'prompt': test_case['prompt'],
                'generated': generated,
                'expected_keywords': test_case['expected_keywords'],
                'found_keywords': found_keywords,
                'score': score,
                'category': test_case['category']
            })
            results['scores'].append(score)
            
            # Category scores
            category = test_case['category']
            if category not in results['category_scores']:
                results['category_scores'][category] = []
            results['category_scores'][category].append(score)
        
        # Calculate average scores per category
        for category in results['category_scores']:
            scores = results['category_scores'][category]
            results['category_scores'][category] = {
                'scores': scores,
                'average': np.mean(scores),
                'count': len(scores)
            }
        
        results['overall_score'] = np.mean(results['scores'])
        
        return results
    
    def create_evaluation_report(self, output_dir: str = "./evaluation_results"):
        """Create comprehensive evaluation report"""
        os.makedirs(output_dir, exist_ok=True)
        
        report = {
            'model_path': self.model_path,
            'model_loaded': self.model_loaded
        }
        
        if not self.model_loaded:
            logger.error("Cannot create evaluation report - model not loaded")
            return report
        
        # Load test data
        logger.info("Loading test data...")
        try:
            preprocessor = ThaiDatasetPreprocessor()
            dataset = preprocessor.load_dataset(streaming=True)
            test_texts = preprocessor.extract_texts_from_dataset(dataset, max_samples=100)
        except Exception as e:
            logger.warning(f"Could not load test dataset: {e}")
            test_texts = [
                "ประเทศไทยมีประวัติศาสตร์ยาวนาน",
                "วิทยาศาสตร์และเทคโนโลยีมีบทบาทสำคัญ",
                "การศึกษาเป็นรากฐานของการพัฒนา"
            ]
        
        # Evaluate perplexity
        logger.info("Evaluating perplexity...")
        perplexity = self.evaluate_perplexity(test_texts[:20])  # Use subset for speed
        report['perplexity'] = perplexity
        
        # Evaluate generation quality
        test_prompts = [
            "ประเทศไทยมีจังหวัด",
            "วิทยาศาสตร์และเทคโนโลยี",
            "การศึกษาในยุคดิจิทัล",
            "อาหารไทยที่มีชื่อเสียง",
            "ประวัติศาสตร์ของไทย"
        ]
        
        generation_results = self.evaluate_generation_quality(test_prompts)
        report['generation_quality'] = generation_results
        
        # Evaluate Thai understanding
        thai_results = self.evaluate_thai_language_understanding()
        report['thai_understanding'] = thai_results
        
        # Save report
        report_path = os.path.join(output_dir, "evaluation_report.json")
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Create visualizations
        self.create_visualizations(report, output_dir)
        
        # Print summary
        self.print_evaluation_summary(report)
        
        return report
    
    def create_visualizations(self, report: Dict[str, Any], output_dir: str):
        """Create evaluation visualizations"""
        plt.style.use('seaborn-v0_8')
        
        # Generation length distribution
        if 'generation_quality' in report and report['generation_quality']:
            lengths = report['generation_quality']['lengths']
            
            plt.figure(figsize=(10, 6))
            plt.hist(lengths, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            plt.title('Distribution of Generated Text Lengths')
            plt.xlabel('Number of Words')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'generation_lengths.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # Repetition scores
            repetition_scores = report['generation_quality']['repetition_scores']
            plt.figure(figsize=(10, 6))
            plt.hist(repetition_scores, bins=20, alpha=0.7, color='lightcoral', edgecolor='black')
            plt.title('Distribution of Repetition Scores')
            plt.xlabel('Repetition Score (lower is better)')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.savefig(os.path.join(output_dir, 'repetition_scores.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Thai understanding scores by category
        if 'thai_understanding' in report and report['thai_understanding']:
            thai_results = report['thai_understanding']
            categories = list(thai_results['category_scores'].keys())
            scores = [thai_results['category_scores'][cat]['average'] for cat in categories]
            
            plt.figure(figsize=(10, 6))
            bars = plt.bar(categories, scores, color='lightgreen', alpha=0.7, edgecolor='black')
            plt.title('Thai Language Understanding by Category')
            plt.xlabel('Category')
            plt.ylabel('Average Score')
            plt.ylim(0, 1)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Add value labels on bars
            for bar, score in zip(bars, scores):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                        f'{score:.2f}', ha='center', va='bottom')
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, 'thai_understanding.png'), dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizations saved to {output_dir}")
    
    def print_evaluation_summary(self, report: Dict[str, Any]):
        """Print evaluation summary"""
        print("\n" + "="*60)
        print("📊 THAI SLM MODEL EVALUATION REPORT")
        print("="*60)
        
        if not report.get('model_loaded', False):
            print("❌ Model not loaded - evaluation incomplete")
            return
        
        print(f"📂 Model Path: {report['model_path']}")
        
        # Perplexity
        if 'perplexity' in report:
            perplexity = report['perplexity']
            print(f"\n📈 Perplexity: {perplexity:.2f}")
            if perplexity < 50:
                print("   ✅ Good perplexity score")
            elif perplexity < 100:
                print("   ⚠️ Moderate perplexity score")
            else:
                print("   ❌ High perplexity score")
        
        # Generation quality
        if 'generation_quality' in report and report['generation_quality']:
            gen_qual = report['generation_quality']
            avg_length = np.mean(gen_qual['lengths'])
            avg_repetition = np.mean(gen_qual['repetition_scores'])
            diversity = gen_qual.get('overall_diversity', 0)
            
            print(f"\n📝 Generation Quality:")
            print(f"   Average Length: {avg_length:.1f} words")
            print(f"   Average Repetition: {avg_repetition:.3f} (lower is better)")
            print(f"   Overall Diversity: {diversity:.3f} (higher is better)")
        
        # Thai understanding
        if 'thai_understanding' in report and report['thai_understanding']:
            thai_results = report['thai_understanding']
            overall_score = thai_results.get('overall_score', 0)
            
            print(f"\n🇹🇭 Thai Language Understanding: {overall_score:.2f}/1.0")
            
            if overall_score > 0.7:
                print("   ✅ Excellent Thai understanding")
            elif overall_score > 0.5:
                print("   ⚠️ Good Thai understanding")
            elif overall_score > 0.3:
                print("   ⚠️ Moderate Thai understanding")
            else:
                print("   ❌ Poor Thai understanding")
            
            print("   Category Breakdown:")
            for category, data in thai_results['category_scores'].items():
                avg_score = data['average']
                print(f"     {category}: {avg_score:.2f}")
        
        print("\n" + "="*60)


def main():
    """Main evaluation function"""
    model_path = "./thai_slm_moe_model"
    
    if not os.path.exists(model_path):
        logger.error("Model not found. Please train the model first.")
        return
    
    evaluator = ModelEvaluator(model_path)
    evaluator.create_evaluation_report()


if __name__ == "__main__":
    main()

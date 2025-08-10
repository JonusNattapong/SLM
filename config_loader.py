import configparser
import os
from typing import Dict, Any, Union
from dataclasses import dataclass, field


@dataclass
class ModelConfig:
    """Model configuration"""
    vocab_size: int = 30000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    max_position_embeddings: int = 1024
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    aux_loss_alpha: float = 0.01
    router_z_loss_alpha: float = 0.001
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2


@dataclass
class TrainingConfig:
    """Training configuration"""
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 8
    learning_rate: float = 1e-4
    weight_decay: float = 0.1
    max_grad_norm: float = 1.0
    warmup_steps: int = 1000
    scheduler_type: str = "cosine"
    optimizer_type: str = "adamw"
    adam_beta1: float = 0.9
    adam_beta2: float = 0.95
    adam_epsilon: float = 1e-8
    logging_steps: int = 100
    save_steps: int = 500
    eval_steps: int = 250
    save_total_limit: int = 3
    fp16: bool = True
    fp16_opt_level: str = "O1"


@dataclass
class DatasetConfig:
    """Dataset configuration"""
    dataset_name: str = "ZombitX64/Wikipedia-Thai"
    max_samples: int = 10000
    max_length: int = 512
    stride: int = 256
    train_test_split: float = 0.9
    tokenizer_vocab_size: int = 30000
    min_frequency: int = 2
    add_prefix_space: bool = False


@dataclass
class GenerationConfig:
    """Generation configuration"""
    max_new_tokens: int = 150
    temperature: float = 0.8
    top_k: int = 50
    top_p: float = 0.9
    do_sample: bool = True
    repetition_penalty: float = 1.1


@dataclass
class WandbConfig:
    """Weights & Biases configuration"""
    enabled: bool = True
    project: str = "thai-slm-moe"
    entity: str = ""
    tags: list = field(default_factory=lambda: ["thai", "slm", "moe", "language-model"])


@dataclass
class HuggingFaceConfig:
    """Hugging Face configuration"""
    organization: str = ""
    model_name: str = "thai-slm-moe"
    private: bool = False
    license: str = "apache-2.0"


@dataclass
class HardwareConfig:
    """Hardware configuration"""
    device: str = "auto"
    num_workers: int = 2
    pin_memory: bool = True
    use_multiple_gpus: bool = False


@dataclass
class PathsConfig:
    """Paths configuration"""
    output_dir: str = "./thai_slm_moe_model"
    tokenizer_dir: str = "./thai_tokenizer"
    data_dir: str = "./data"
    cache_dir: str = "./cache"
    log_dir: str = "./logs"


@dataclass
class EvaluationConfig:
    """Evaluation configuration"""
    eval_batch_size: int = 8
    eval_max_samples: int = 100
    eval_perplexity: bool = True
    eval_generation_quality: bool = True
    eval_thai_understanding: bool = True


@dataclass
class WebInterfaceConfig:
    """Web interface configuration"""
    port: int = 7860
    share: bool = False
    server_name: str = "127.0.0.1"
    enable_queue: bool = True
    max_size: int = 20


@dataclass
class Config:
    """Main configuration class"""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    wandb: WandbConfig = field(default_factory=WandbConfig)
    huggingface: HuggingFaceConfig = field(default_factory=HuggingFaceConfig)
    hardware: HardwareConfig = field(default_factory=HardwareConfig)
    paths: PathsConfig = field(default_factory=PathsConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    web_interface: WebInterfaceConfig = field(default_factory=WebInterfaceConfig)


class ConfigLoader:
    """Configuration loader from INI file"""
    
    def __init__(self, config_path: str = "config.ini"):
        self.config_path = config_path
        self.parser = configparser.ConfigParser()
        
        if os.path.exists(config_path):
            self.parser.read(config_path, encoding='utf-8')
        else:
            print(f"Warning: Config file {config_path} not found. Using default values.")
    
    def _get_value(self, section: str, key: str, default: Any, value_type: type = str) -> Any:
        """Get value from config with type conversion"""
        try:
            if value_type == bool:
                return self.parser.getboolean(section, key)
            elif value_type == int:
                return self.parser.getint(section, key)
            elif value_type == float:
                return self.parser.getfloat(section, key)
            elif value_type == list:
                value = self.parser.get(section, key)
                return [item.strip().strip('"\'') for item in value.split(',') if item.strip()]
            else:
                return self.parser.get(section, key)
        except (configparser.NoSectionError, configparser.NoOptionError, ValueError):
            return default
    
    def load_config(self) -> Config:
        """Load complete configuration"""
        config = Config()
        
        # Model configuration
        model_defaults = ModelConfig()
        config.model = ModelConfig(
            vocab_size=self._get_value('model', 'vocab_size', model_defaults.vocab_size, int),
            hidden_size=self._get_value('model', 'hidden_size', model_defaults.hidden_size, int),
            num_hidden_layers=self._get_value('model', 'num_hidden_layers', model_defaults.num_hidden_layers, int),
            num_attention_heads=self._get_value('model', 'num_attention_heads', model_defaults.num_attention_heads, int),
            intermediate_size=self._get_value('model', 'intermediate_size', model_defaults.intermediate_size, int),
            max_position_embeddings=self._get_value('model', 'max_position_embeddings', model_defaults.max_position_embeddings, int),
            num_experts=self._get_value('model', 'num_experts', model_defaults.num_experts, int),
            num_experts_per_token=self._get_value('model', 'num_experts_per_token', model_defaults.num_experts_per_token, int),
            expert_capacity_factor=self._get_value('model', 'expert_capacity_factor', model_defaults.expert_capacity_factor, float),
            hidden_dropout_prob=self._get_value('model', 'hidden_dropout_prob', model_defaults.hidden_dropout_prob, float),
            attention_dropout_prob=self._get_value('model', 'attention_dropout_prob', model_defaults.attention_dropout_prob, float),
            layer_norm_eps=self._get_value('model', 'layer_norm_eps', model_defaults.layer_norm_eps, float),
            initializer_range=self._get_value('model', 'initializer_range', model_defaults.initializer_range, float),
            aux_loss_alpha=self._get_value('model', 'aux_loss_alpha', model_defaults.aux_loss_alpha, float),
            router_z_loss_alpha=self._get_value('model', 'router_z_loss_alpha', model_defaults.router_z_loss_alpha, float),
            pad_token_id=self._get_value('model', 'pad_token_id', model_defaults.pad_token_id, int),
            bos_token_id=self._get_value('model', 'bos_token_id', model_defaults.bos_token_id, int),
            eos_token_id=self._get_value('model', 'eos_token_id', model_defaults.eos_token_id, int)
        )
        
        # Training configuration
        training_defaults = TrainingConfig()
        config.training = TrainingConfig(
            num_epochs=self._get_value('training', 'num_epochs', training_defaults.num_epochs, int),
            batch_size=self._get_value('training', 'batch_size', training_defaults.batch_size, int),
            gradient_accumulation_steps=self._get_value('training', 'gradient_accumulation_steps', training_defaults.gradient_accumulation_steps, int),
            learning_rate=self._get_value('training', 'learning_rate', training_defaults.learning_rate, float),
            weight_decay=self._get_value('training', 'weight_decay', training_defaults.weight_decay, float),
            max_grad_norm=self._get_value('training', 'max_grad_norm', training_defaults.max_grad_norm, float),
            warmup_steps=self._get_value('training', 'warmup_steps', training_defaults.warmup_steps, int),
            scheduler_type=self._get_value('training', 'scheduler_type', training_defaults.scheduler_type, str),
            optimizer_type=self._get_value('training', 'optimizer_type', training_defaults.optimizer_type, str),
            adam_beta1=self._get_value('training', 'adam_beta1', training_defaults.adam_beta1, float),
            adam_beta2=self._get_value('training', 'adam_beta2', training_defaults.adam_beta2, float),
            adam_epsilon=self._get_value('training', 'adam_epsilon', training_defaults.adam_epsilon, float),
            logging_steps=self._get_value('training', 'logging_steps', training_defaults.logging_steps, int),
            save_steps=self._get_value('training', 'save_steps', training_defaults.save_steps, int),
            eval_steps=self._get_value('training', 'eval_steps', training_defaults.eval_steps, int),
            save_total_limit=self._get_value('training', 'save_total_limit', training_defaults.save_total_limit, int),
            fp16=self._get_value('training', 'fp16', training_defaults.fp16, bool),
            fp16_opt_level=self._get_value('training', 'fp16_opt_level', training_defaults.fp16_opt_level, str)
        )
        
        # Dataset configuration
        dataset_defaults = DatasetConfig()
        config.dataset = DatasetConfig(
            dataset_name=self._get_value('dataset', 'dataset_name', dataset_defaults.dataset_name, str),
            max_samples=self._get_value('dataset', 'max_samples', dataset_defaults.max_samples, int),
            max_length=self._get_value('dataset', 'max_length', dataset_defaults.max_length, int),
            stride=self._get_value('dataset', 'stride', dataset_defaults.stride, int),
            train_test_split=self._get_value('dataset', 'train_test_split', dataset_defaults.train_test_split, float),
            tokenizer_vocab_size=self._get_value('dataset', 'tokenizer_vocab_size', dataset_defaults.tokenizer_vocab_size, int),
            min_frequency=self._get_value('dataset', 'min_frequency', dataset_defaults.min_frequency, int),
            add_prefix_space=self._get_value('dataset', 'add_prefix_space', dataset_defaults.add_prefix_space, bool)
        )
        
        # Generation configuration
        generation_defaults = GenerationConfig()
        config.generation = GenerationConfig(
            max_new_tokens=self._get_value('generation', 'max_new_tokens', generation_defaults.max_new_tokens, int),
            temperature=self._get_value('generation', 'temperature', generation_defaults.temperature, float),
            top_k=self._get_value('generation', 'top_k', generation_defaults.top_k, int),
            top_p=self._get_value('generation', 'top_p', generation_defaults.top_p, float),
            do_sample=self._get_value('generation', 'do_sample', generation_defaults.do_sample, bool),
            repetition_penalty=self._get_value('generation', 'repetition_penalty', generation_defaults.repetition_penalty, float)
        )
        
        # WandB configuration
        wandb_defaults = WandbConfig()
        config.wandb = WandbConfig(
            enabled=self._get_value('wandb', 'enabled', wandb_defaults.enabled, bool),
            project=self._get_value('wandb', 'project', wandb_defaults.project, str),
            entity=self._get_value('wandb', 'entity', wandb_defaults.entity, str),
            tags=self._get_value('wandb', 'tags', wandb_defaults.tags, list)
        )
        
        # HuggingFace configuration
        hf_defaults = HuggingFaceConfig()
        config.huggingface = HuggingFaceConfig(
            organization=self._get_value('huggingface', 'organization', hf_defaults.organization, str),
            model_name=self._get_value('huggingface', 'model_name', hf_defaults.model_name, str),
            private=self._get_value('huggingface', 'private', hf_defaults.private, bool),
            license=self._get_value('huggingface', 'license', hf_defaults.license, str)
        )
        
        # Hardware configuration
        hardware_defaults = HardwareConfig()
        config.hardware = HardwareConfig(
            device=self._get_value('hardware', 'device', hardware_defaults.device, str),
            num_workers=self._get_value('hardware', 'num_workers', hardware_defaults.num_workers, int),
            pin_memory=self._get_value('hardware', 'pin_memory', hardware_defaults.pin_memory, bool),
            use_multiple_gpus=self._get_value('hardware', 'use_multiple_gpus', hardware_defaults.use_multiple_gpus, bool)
        )
        
        # Paths configuration
        paths_defaults = PathsConfig()
        config.paths = PathsConfig(
            output_dir=self._get_value('paths', 'output_dir', paths_defaults.output_dir, str),
            tokenizer_dir=self._get_value('paths', 'tokenizer_dir', paths_defaults.tokenizer_dir, str),
            data_dir=self._get_value('paths', 'data_dir', paths_defaults.data_dir, str),
            cache_dir=self._get_value('paths', 'cache_dir', paths_defaults.cache_dir, str),
            log_dir=self._get_value('paths', 'log_dir', paths_defaults.log_dir, str)
        )
        
        # Evaluation configuration
        eval_defaults = EvaluationConfig()
        config.evaluation = EvaluationConfig(
            eval_batch_size=self._get_value('evaluation', 'eval_batch_size', eval_defaults.eval_batch_size, int),
            eval_max_samples=self._get_value('evaluation', 'eval_max_samples', eval_defaults.eval_max_samples, int),
            eval_perplexity=self._get_value('evaluation', 'eval_perplexity', eval_defaults.eval_perplexity, bool),
            eval_generation_quality=self._get_value('evaluation', 'eval_generation_quality', eval_defaults.eval_generation_quality, bool),
            eval_thai_understanding=self._get_value('evaluation', 'eval_thai_understanding', eval_defaults.eval_thai_understanding, bool)
        )
        
        # Web interface configuration
        web_defaults = WebInterfaceConfig()
        config.web_interface = WebInterfaceConfig(
            port=self._get_value('web_interface', 'port', web_defaults.port, int),
            share=self._get_value('web_interface', 'share', web_defaults.share, bool),
            server_name=self._get_value('web_interface', 'server_name', web_defaults.server_name, str),
            enable_queue=self._get_value('web_interface', 'enable_queue', web_defaults.enable_queue, bool),
            max_size=self._get_value('web_interface', 'max_size', web_defaults.max_size, int)
        )
        
        return config
    
    def save_config(self, config: Config, output_path: str = "config_generated.ini"):
        """Save configuration to INI file"""
        parser = configparser.ConfigParser()
        
        # Model section
        parser.add_section('model')
        model = config.model
        for field_name, field_value in model.__dict__.items():
            parser.set('model', field_name, str(field_value))
        
        # Training section
        parser.add_section('training')
        training = config.training
        for field_name, field_value in training.__dict__.items():
            parser.set('training', field_name, str(field_value))
        
        # Dataset section
        parser.add_section('dataset')
        dataset = config.dataset
        for field_name, field_value in dataset.__dict__.items():
            parser.set('dataset', field_name, str(field_value))
        
        # Generation section
        parser.add_section('generation')
        generation = config.generation
        for field_name, field_value in generation.__dict__.items():
            parser.set('generation', field_name, str(field_value))
        
        # WandB section
        parser.add_section('wandb')
        wandb_config = config.wandb
        for field_name, field_value in wandb_config.__dict__.items():
            if isinstance(field_value, list):
                field_value = ', '.join(field_value)
            parser.set('wandb', field_name, str(field_value))
        
        # HuggingFace section
        parser.add_section('huggingface')
        hf = config.huggingface
        for field_name, field_value in hf.__dict__.items():
            parser.set('huggingface', field_name, str(field_value))
        
        # Hardware section
        parser.add_section('hardware')
        hardware = config.hardware
        for field_name, field_value in hardware.__dict__.items():
            parser.set('hardware', field_name, str(field_value))
        
        # Paths section
        parser.add_section('paths')
        paths = config.paths
        for field_name, field_value in paths.__dict__.items():
            parser.set('paths', field_name, str(field_value))
        
        # Evaluation section
        parser.add_section('evaluation')
        evaluation = config.evaluation
        for field_name, field_value in evaluation.__dict__.items():
            parser.set('evaluation', field_name, str(field_value))
        
        # Web interface section
        parser.add_section('web_interface')
        web = config.web_interface
        for field_name, field_value in web.__dict__.items():
            parser.set('web_interface', field_name, str(field_value))
        
        # Write to file
        with open(output_path, 'w', encoding='utf-8') as f:
            parser.write(f)


def load_config(config_path: str = "config.ini") -> Config:
    """Load configuration from file"""
    loader = ConfigLoader(config_path)
    return loader.load_config()


if __name__ == "__main__":
    # Test configuration loading
    config = load_config()
    print("Configuration loaded successfully!")
    print(f"Model hidden size: {config.model.hidden_size}")
    print(f"Training epochs: {config.training.num_epochs}")
    print(f"Dataset: {config.dataset.dataset_name}")
    print(f"Output directory: {config.paths.output_dir}")

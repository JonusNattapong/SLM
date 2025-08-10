import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, List, Dict, Any
from dataclasses import dataclass


@dataclass
class SLMConfig:
    """Configuration class for Small Language Model with MoE"""
    vocab_size: int = 50000
    hidden_size: int = 768
    num_hidden_layers: int = 12
    num_attention_heads: int = 12
    intermediate_size: int = 3072
    hidden_dropout_prob: float = 0.1
    attention_dropout_prob: float = 0.1
    max_position_embeddings: int = 2048
    layer_norm_eps: float = 1e-12
    initializer_range: float = 0.02
    use_cache: bool = True
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    # GQA parameters
    num_kv_heads: int = 0  # if 0, defaults to num_attention_heads

    # RoPE scaling parameters
    rope_theta: float = 10000.0
    rope_scaling_type: Optional[str] = None  # e.g., 'linear'
    rope_scaling_factor: float = 1.0
    
    # MoE specific parameters
    num_experts: int = 8
    num_experts_per_token: int = 2
    expert_capacity_factor: float = 1.25
    aux_loss_alpha: float = 0.01
    router_z_loss_alpha: float = 0.001
    
    # Advanced features
    use_flash_attention: bool = True
    use_dynamic_capacity: bool = True
    layer_drop_prob: float = 0.0  # Stochastic depth
    use_gradient_checkpointing: bool = False
    
    # NTK-aware RoPE scaling
    rope_scaling_ntk: bool = False
    rope_scaling_alpha: float = 1.0


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding (RoPE) with optional scaling"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048, base: float = 10000.0,
                 scaling_type: Optional[str] = None, scaling_factor: float = 1.0):
        super().__init__()
        self.dim = dim
        self.max_seq_len = max_seq_len
        self.base = base
        self.scaling_type = scaling_type
        self.scaling_factor = scaling_factor
        
        # Precompute frequencies
        inv_freq = 1.0 / (self.base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
        # Precompute cos and sin for efficiency (unscaled default)
        self._precompute_freqs_cis(max_seq_len)
    
    def _precompute_freqs_cis(self, seq_len: int):
        t = torch.arange(seq_len, dtype=torch.float32)
        freqs = torch.outer(t, self.inv_freq)
        cos = torch.cos(freqs)
        sin = torch.sin(freqs)
        self.register_buffer('cos_cached', cos)
        self.register_buffer('sin_cached', sin)
    
    def apply_rotary_pos_emb(self, x: torch.Tensor, seq_len: int = None):
        if seq_len is None:
            seq_len = x.size(-2)  # Get sequence length from correct dimension
        
        # Ensure cache large enough
        if seq_len > self.max_seq_len:
            self._precompute_freqs_cis(seq_len)
            self.max_seq_len = seq_len
        
        # Handle scaling modes with NTK-aware scaling
        if self.scaling_type == 'linear' and abs(self.scaling_factor - 1.0) > 1e-6:
            # Linear scaling: scale positions by factor
            t = (torch.arange(seq_len, dtype=torch.float32, device=x.device) / self.scaling_factor)
            freqs = torch.outer(t, self.inv_freq.to(x.device))
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
        elif self.scaling_type == 'ntk' and abs(self.scaling_factor - 1.0) > 1e-6:
            # NTK-aware scaling (better for long sequences)
            alpha = self.scaling_factor
            base = self.base * (alpha * seq_len / self.max_seq_len - (alpha - 1)) ** (self.dim / (self.dim - 2))
            inv_freq = 1.0 / (base ** (torch.arange(0, self.dim, 2).float().to(x.device) / self.dim))
            t = torch.arange(seq_len, dtype=torch.float32, device=x.device)
            freqs = torch.outer(t, inv_freq)
            cos = torch.cos(freqs)
            sin = torch.sin(freqs)
        else:
            cos = self.cos_cached[:seq_len, :].to(x.device)
            sin = self.sin_cached[:seq_len, :].to(x.device)
        
        # x shape: [batch_size, num_heads, seq_len, head_dim]
        # We need to reshape cos/sin to broadcast properly
        cos = cos.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        sin = sin.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim//2]
        
        # Apply rotary embedding
        x1, x2 = x[..., ::2], x[..., 1::2]  # Split even/odd dimensions
        rotated_x1 = x1 * cos - x2 * sin
        rotated_x2 = x1 * sin + x2 * cos
        
        # Interleave back together
        rotated_x = torch.zeros_like(x)
        rotated_x[..., ::2] = rotated_x1
        rotated_x[..., 1::2] = rotated_x2
        
        return rotated_x


class SwiGLU(nn.Module):
    """SwiGLU activation function"""
    
    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.gate_proj(x)
        up = self.up_proj(x)
        return self.down_proj(F.silu(gate) * up)


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization"""
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm_x = x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return norm_x * self.weight


class AdvancedRouter(nn.Module):
    """Advanced Router with dynamic capacity for Mixture of Experts"""
    
    def __init__(self, hidden_size: int, num_experts: int, use_dynamic_capacity: bool = False):
        super().__init__()
        self.num_experts = num_experts
        self.use_dynamic_capacity = use_dynamic_capacity
        self.router = nn.Linear(hidden_size, num_experts, bias=False)
        
        # For dynamic capacity
        if use_dynamic_capacity:
            self.capacity_predictor = nn.Linear(hidden_size, 1)
            self.capacity_activation = nn.Sigmoid()
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        # x: [batch_size, seq_len, hidden_size]
        router_logits = self.router(x)  # [batch_size, seq_len, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)
        
        # Dynamic capacity prediction
        dynamic_capacity = None
        if self.use_dynamic_capacity:
            capacity_logits = self.capacity_predictor(x)  # [batch_size, seq_len, 1]
            dynamic_capacity = self.capacity_activation(capacity_logits)  # [0, 1]
        
        return router_logits, router_probs, dynamic_capacity


class MixtureOfExperts(nn.Module):
    """Advanced Mixture of Experts layer with dynamic capacity"""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.expert_capacity_factor = config.expert_capacity_factor
        self.use_dynamic_capacity = getattr(config, 'use_dynamic_capacity', False)
        
        self.router = AdvancedRouter(config.hidden_size, config.num_experts, self.use_dynamic_capacity)
        self.experts = nn.ModuleList([Expert(config) for _ in range(config.num_experts)])
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        batch_size, seq_len, hidden_size = x.shape
        
        # Get router outputs
        router_logits, router_probs, dynamic_capacity = self.router(x)
        
        # Select top-k experts per token
        top_k_probs, top_k_indices = torch.topk(router_probs, self.num_experts_per_token, dim=-1)
        top_k_probs = F.softmax(top_k_probs, dim=-1)
        
        # Apply dynamic capacity if enabled
        if self.use_dynamic_capacity and dynamic_capacity is not None:
            # Modulate expert selection based on predicted capacity
            capacity_weights = dynamic_capacity.expand_as(top_k_probs[:, :, :1])
            top_k_probs = top_k_probs * capacity_weights
        
        # Initialize output
        output = torch.zeros_like(x)
        
        # Route tokens to experts (vectorized approach for better efficiency)
        flat_top_k_indices = top_k_indices.view(-1, self.num_experts_per_token)
        flat_top_k_probs = top_k_probs.view(-1, self.num_experts_per_token)
        flat_x = x.view(-1, hidden_size)
        flat_output = torch.zeros_like(flat_x)
        
        for expert_idx in range(self.num_experts):
            # Create mask for tokens assigned to this expert
            expert_mask = (flat_top_k_indices == expert_idx)
            if expert_mask.any():
                # Get tokens and weights for this expert
                token_indices = expert_mask.any(dim=1)
                if token_indices.any():
                    expert_tokens = flat_x[token_indices]
                    expert_output = self.experts[expert_idx](expert_tokens)
                    
                    # Calculate weights for this expert
                    weights = flat_top_k_probs[token_indices] * expert_mask[token_indices].float()
                    weights = weights.sum(dim=1, keepdim=True)
                    
                    # Add weighted output
                    flat_output[token_indices] += expert_output * weights
        
        output = flat_output.view(batch_size, seq_len, hidden_size)
        
        # Compute auxiliary losses
        aux_losses = self._compute_auxiliary_losses(router_probs, top_k_indices, dynamic_capacity)
        
        return output, aux_losses
    
    def _compute_auxiliary_losses(self, router_probs: torch.Tensor, top_k_indices: torch.Tensor, 
                                  dynamic_capacity: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """Compute auxiliary losses for training stability"""
        
        # Load balancing loss
        expert_counts = torch.zeros(self.num_experts, device=router_probs.device)
        for i in range(self.num_experts):
            expert_counts[i] = (top_k_indices == i).float().sum()
        
        total_tokens = top_k_indices.numel()
        load_balance_loss = torch.var(expert_counts) / (total_tokens / self.num_experts) ** 2
        
        # Router z-loss (encourages router to be decisive)
        router_z_loss = torch.logsumexp(router_probs, dim=-1).mean()
        
        # Expert diversity loss (encourages different experts to specialize)
        expert_probs = router_probs.mean(dim=(0, 1))  # Average probability per expert
        diversity_loss = -torch.sum(expert_probs * torch.log(expert_probs + 1e-8))
        
        losses = {
            'load_balance_loss': load_balance_loss,
            'router_z_loss': router_z_loss,
            'diversity_loss': diversity_loss
        }
        
        # Dynamic capacity loss if enabled
        if self.use_dynamic_capacity and dynamic_capacity is not None:
            # Encourage moderate capacity usage (not too high/low)
            capacity_target = 0.6  # Target capacity usage
            capacity_loss = F.mse_loss(dynamic_capacity.mean(), torch.tensor(capacity_target, device=dynamic_capacity.device))
            losses['capacity_loss'] = capacity_loss
        
        return losses


class Expert(nn.Module):
    """Single expert in MoE"""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.swiglu = SwiGLU(config.hidden_size, config.intermediate_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.swiglu(x))


class MultiHeadAttention(nn.Module):
    """Multi-Head Attention with RoPE, GQA, and Flash Attention support"""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.num_attention_heads = config.num_attention_heads
        self.num_kv_heads = config.num_kv_heads if getattr(config, 'num_kv_heads', 0) > 0 else config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_attention_heads
        self.use_flash_attention = getattr(config, 'use_flash_attention', True)
        
        assert self.hidden_size % self.num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
        assert self.num_attention_heads % self.num_kv_heads == 0, "num_attention_heads must be multiple of num_kv_heads"
        
        # Projections: Q has H*D, K/V have KV*D
        self.q_proj = nn.Linear(self.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Initialize RoPE with config values
        rope_scaling_type = getattr(config, 'rope_scaling_type', None)
        if getattr(config, 'rope_scaling_ntk', False):
            rope_scaling_type = 'ntk'
        
        self.rotary_emb = RotaryPositionalEmbedding(
            self.head_dim, 
            config.max_position_embeddings,
            base=config.rope_theta,
            scaling_type=rope_scaling_type,
            scaling_factor=getattr(config, 'rope_scaling_factor', 1.0)
        )
        
        self.dropout = nn.Dropout(config.attention_dropout_prob)
        self.scale = 1.0 / math.sqrt(self.head_dim)
    
    def forward(
        self, 
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        
        batch_size, seq_len, _ = hidden_states.shape
        
        # Project to Q, K, V
        q = self.q_proj(hidden_states)
        k = self.k_proj(hidden_states)
        v = self.v_proj(hidden_states)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.num_attention_heads, self.head_dim).transpose(1, 2)          # [B, H, S, D]
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)                 # [B, KV, S, D]
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)                 # [B, KV, S, D]
        
        # Apply rotary positional embedding to q and k
        q = self.rotary_emb.apply_rotary_pos_emb(q, seq_len)
        k = self.rotary_emb.apply_rotary_pos_emb(k, seq_len)
        
        # Expand K/V heads to match Q heads if using GQA
        if self.num_kv_heads != self.num_attention_heads:
            repeat_factor = self.num_attention_heads // self.num_kv_heads
            k = k.repeat_interleave(repeat_factor, dim=1)
            v = v.repeat_interleave(repeat_factor, dim=1)
        
        # Handle past key values for generation
        if past_key_value is not None:
            past_key, past_value = past_key_value
            k = torch.cat([past_key, k], dim=2)
            v = torch.cat([past_value, v], dim=2)
        
        # Store key/value for next step if using cache
        present_key_value = (k, v) if use_cache else None
        
        # Use Flash Attention if available and enabled
        if self.use_flash_attention and hasattr(F, 'scaled_dot_product_attention'):
            # For SDPA, we don't need custom attention mask - just use is_causal=True
            # This is more efficient and avoids tensor size mismatches
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.config.attention_dropout_prob if self.training else 0.0,
                is_causal=True  # Always use causal attention for language modeling
            )
        else:
            # Fallback to manual attention computation
            attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, S, S]
            
            if attention_mask is not None:
                attn_scores = attn_scores + attention_mask  # additive mask as bias
            
            attn_probs = F.softmax(attn_scores, dim=-1)
            attn_probs = self.dropout(attn_probs)
            
            # Apply attention to values
            attn_output = torch.matmul(attn_probs, v)  # [B, H, S, D]
        
        # Reshape and project output
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.hidden_size)
        attn_output = self.o_proj(attn_output)
        
        return attn_output, present_key_value


class TransformerBlock(nn.Module):
    """Transformer block with MoE and Stochastic Depth"""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.attention = MultiHeadAttention(config)
        self.moe = MixtureOfExperts(config)
        
        # Replace LayerNorm with RMSNorm
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Stochastic depth (layer drop)
        self.layer_drop_prob = getattr(config, 'layer_drop_prob', 0.0)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]], Dict[str, torch.Tensor]]:
        
        # Apply stochastic depth during training
        if self.training and self.layer_drop_prob > 0:
            if torch.rand(1).item() < self.layer_drop_prob:
                # Skip this layer but return consistent aux_losses structure
                empty_aux_losses = {
                    'load_balance_loss': torch.tensor(0.0, device=hidden_states.device),
                    'router_z_loss': torch.tensor(0.0, device=hidden_states.device),
                    'diversity_loss': torch.tensor(0.0, device=hidden_states.device)
                }
                return hidden_states, past_key_value, empty_aux_losses
        
        # Self-attention with residual connection
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        attention_output, present_key_value = self.attention(
            hidden_states, attention_mask, past_key_value, use_cache
        )
        hidden_states = residual + self.dropout(attention_output)
        
        # MoE with residual connection
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        moe_output, aux_losses = self.moe(hidden_states)
        hidden_states = residual + self.dropout(moe_output)
        
        return hidden_states, present_key_value, aux_losses


class SLMModel(nn.Module):
    """Small Language Model with Mixture of Experts and advanced features"""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        ])
        # Replace LayerNorm with RMSNorm
        self.norm = RMSNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Gradient checkpointing support
        self.use_gradient_checkpointing = getattr(config, 'use_gradient_checkpointing', False)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, RMSNorm):
            torch.nn.init.ones_(module.weight)
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        
        batch_size, seq_len = input_ids.shape
        
        # Create attention mask if not provided
        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_len), device=input_ids.device)
        
        # Convert attention mask to bias
        if attention_mask is not None:
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = (1.0 - attention_mask) * torch.finfo(torch.float32).min
        
        # Embed tokens
        hidden_states = self.embed_tokens(input_ids)
        hidden_states = self.dropout(hidden_states)
        
        # Initialize past key values if not provided
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        
        present_key_values = []
        all_aux_losses = []
        
        # Pass through transformer layers
        for i, (layer, past_key_value) in enumerate(zip(self.layers, past_key_values)):
            if self.use_gradient_checkpointing and self.training:
                # Use gradient checkpointing
                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs)
                    return custom_forward
                
                hidden_states, present_key_value, aux_losses = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer),
                    hidden_states,
                    attention_mask,
                    past_key_value,
                    use_cache
                )
            else:
                hidden_states, present_key_value, aux_losses = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_key_value,
                    use_cache=use_cache
                )
            
            if use_cache:
                present_key_values.append(present_key_value)
            
            if aux_losses:  # Only add if not empty (stochastic depth might skip layers)
                all_aux_losses.append(aux_losses)
        
        # Final layer norm
        hidden_states = self.norm(hidden_states)
        
        # Aggregate auxiliary losses
        total_aux_losses = {}
        if all_aux_losses:
            # Get all unique loss keys from all layers
            all_loss_keys = set()
            for aux_losses in all_aux_losses:
                all_loss_keys.update(aux_losses.keys())
            
            # Aggregate each loss type
            for loss_key in all_loss_keys:
                losses_for_key = [
                    layer_losses[loss_key] for layer_losses in all_aux_losses 
                    if loss_key in layer_losses
                ]
                if losses_for_key:
                    total_aux_losses[loss_key] = sum(losses_for_key) / len(losses_for_key)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': present_key_values if use_cache else None,
            'aux_losses': total_aux_losses
        }


class SLMForCausalLM(nn.Module):
    """SLM for Causal Language Modeling"""
    
    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # Tie weights
        self.lm_head.weight = self.model.embed_tokens.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[Tuple[torch.Tensor]]] = None,
        use_cache: bool = False
    ) -> Dict[str, Any]:
        
        # Forward pass through the model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache
        )
        
        hidden_states = outputs['last_hidden_state']
        
        # Compute logits
        logits = self.lm_head(hidden_states)
        
        loss = None
        if labels is not None:
            # Shift labels for causal LM
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            
            # Compute cross-entropy loss
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            
            # Add auxiliary losses with advanced loss types
            aux_losses = outputs['aux_losses']
            total_aux_loss = (
                self.config.aux_loss_alpha * aux_losses.get('load_balance_loss', 0.0) +
                self.config.router_z_loss_alpha * aux_losses.get('router_z_loss', 0.0) +
                0.001 * aux_losses.get('diversity_loss', 0.0) +  # Expert diversity
                0.01 * aux_losses.get('capacity_loss', 0.0)     # Dynamic capacity
            )
            loss = loss + total_aux_loss
        
        return {
            'loss': loss,
            'logits': logits,
            'past_key_values': outputs['past_key_values'],
            'aux_losses': outputs['aux_losses']
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.9,
        do_sample: bool = True,
        pad_token_id: int = None,
        eos_token_id: int = None,
        repetition_penalty: float = 1.0,
        frequency_penalty: float = 0.0,
        presence_penalty: float = 0.0,
        penalty_window: int = 64,
        bad_token_ids: Optional[List[int]] = None,
    ) -> torch.Tensor:
        """Generate text using the model with improved sampling and penalties"""
        
        if pad_token_id is None:
            pad_token_id = self.config.pad_token_id
        if eos_token_id is None:
            eos_token_id = self.config.eos_token_id
        
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        vocab_size = self.config.vocab_size
        
        # Initialize generation
        generated = input_ids.clone()
        past_key_values = None
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
        
        for _ in range(max_length - seq_len):
            # Forward pass
            with torch.no_grad():
                outputs = self.forward(
                    input_ids=generated[:, -1:] if past_key_values is not None else generated,
                    past_key_values=past_key_values,
                    use_cache=True
                )
            
            logits = outputs['logits'][:, -1, :]  # [B, V]
            past_key_values = outputs['past_key_values']
            
            # Apply basic masks (avoid generating pad unless finished)
            if pad_token_id is not None:
                logits[:, pad_token_id] = float('-inf')
            if bad_token_ids:
                logits[:, bad_token_ids] = float('-inf')
            
            # Temperature
            if temperature and temperature != 1.0:
                logits = logits / temperature
            
            # Repetition / frequency / presence penalties using a recent window
            if repetition_penalty != 1.0 or frequency_penalty > 0.0 or presence_penalty > 0.0:
                ctx = generated[:, -penalty_window:] if penalty_window > 0 else generated
                # counts: [B, V]
                counts = torch.zeros((batch_size, vocab_size), device=device, dtype=logits.dtype)
                ones = torch.ones_like(ctx, dtype=logits.dtype)
                counts.scatter_add_(1, ctx, ones)
                presence = (counts > 0).to(logits.dtype)
                
                # Frequency & presence penalties (OpenAI-style)
                if frequency_penalty > 0.0:
                    logits = logits - frequency_penalty * counts
                if presence_penalty > 0.0:
                    logits = logits - presence_penalty * presence
                
                # Repetition penalty (HF-style)
                if repetition_penalty != 1.0:
                    rep_mask = presence.bool()
                    if rep_mask.any():
                        pos_mask = logits > 0
                        neg_mask = ~pos_mask
                        # For tokens that appeared: divide positives, multiply negatives
                        mask = rep_mask & pos_mask
                        logits[mask] = logits[mask] / repetition_penalty
                        mask = rep_mask & neg_mask
                        logits[mask] = logits[mask] * repetition_penalty
            
            # Top-k filtering
            if top_k and top_k > 0:
                top_k = min(top_k, vocab_size)
                top_k_values, top_k_indices = torch.topk(logits, top_k, dim=-1)
                mask = torch.full_like(logits, float('-inf'))
                mask.scatter_(1, top_k_indices, top_k_values)
                logits = mask
            
            # Top-p (nucleus) filtering
            if top_p and top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True, dim=-1)
                probs = F.softmax(sorted_logits, dim=-1)
                cumulative_probs = torch.cumsum(probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                # Keep at least 1 token
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                # Set removed indices to -inf in logits
                indices_to_remove = torch.zeros_like(logits, dtype=torch.bool)
                indices_to_remove.scatter_(1, sorted_indices, sorted_indices_to_remove)
                logits = logits.masked_fill(indices_to_remove, float('-inf'))
            
            # Sample or greedy
            if do_sample:
                probs = F.softmax(logits, dim=-1)
                next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1)
            else:
                next_tokens = torch.argmax(logits, dim=-1)
            
            # Finish handling
            finished = finished | (next_tokens == eos_token_id)
            if pad_token_id is not None:
                next_tokens = torch.where(finished, torch.tensor(pad_token_id, device=device), next_tokens)
            
            # Append
            generated = torch.cat([generated, next_tokens.unsqueeze(1)], dim=1)
            
            if finished.all():
                break
        
        return generated


if __name__ == "__main__":
    # Test the enhanced model
    config = SLMConfig(
        vocab_size=30000,
        hidden_size=512,
        num_hidden_layers=8,
        num_attention_heads=8,
        num_kv_heads=4,  # GQA with 4 KV heads
        intermediate_size=2048,
        num_experts=8,
        num_experts_per_token=2,
        # Advanced features
        use_flash_attention=True,
        use_dynamic_capacity=True,
        layer_drop_prob=0.1,  # 10% stochastic depth
        rope_scaling_ntk=True,
        rope_scaling_alpha=2.0
    )
    
    model = SLMForCausalLM(config)
    
    # Test forward pass
    batch_size, seq_len = 2, 256
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    outputs = model(input_ids=input_ids)
    print(f"Enhanced Model output shape: {outputs['logits'].shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Aux losses: {list(outputs['aux_losses'].keys())}")
    print("✅ Enhanced model test passed!")

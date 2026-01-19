import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig, TaskType
import math
from capfl_variational import VariationalLoRAEstimator

# ============================================================================
# Integrated Complete Model
# ============================================================================
class LlamaWithCAFLHead(nn.Module):
    """
    Llama Model Integrated with CA-PFL Mechanisms
    Includes:
    1. LoRA-adapted base model
    2. RFF personalized strategy generator
    3. Variational LoRA estimator
    4. Classification head
    """
    def __init__(self, model_path, llm_last_size, num_lora_layers, lora_rank,
                 mlp_hidden_dim, rff_dim_out, dtype=torch.bfloat16):
        super().__init__()
        
        self.num_lora_layers = num_lora_layers
        self.lora_rank = lora_rank
        
        # 1. 加载并配置LoRA基础模型
        base_model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=dtype,
            device_map={"": "cuda:0"}
        )
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=128,
            target_modules=["q_proj", "v_proj"],
            lora_dropout=0.05,
            bias="none",
        )
        
        self.base_model = get_peft_model(base_model, lora_config)
        self.base_model.enable_input_require_grads()
        
        print("PEFT Model created. Trainable parameters:")
        self.base_model.print_trainable_parameters()
        
        # Determine output device
        self.output_device = self.base_model.device
        print(f"Model output device: {self.output_device}")
        
        # 2. RFF Personalized Strategy Generator
        # Input dimension = num_lora_layers * 2 (mean and std for each layer)
        rff_input_dim = num_lora_layers * 2
        
        self.input_norm = nn.LayerNorm(rff_input_dim, elementwise_affine=False)
        self.rff_projector = RFFProjection(rff_input_dim, rff_dim_out)
        
        # MLP generates layer-wise variational parameters: output num_lora_layers * lora_rank * 2
        mlp_output_dim = num_lora_layers * lora_rank * 2
        
        self.strategy_mlp = nn.Sequential(
            nn.Linear(rff_dim_out, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, mlp_output_dim),
            nn.Softplus()
        )
        
        # 4. Variational LoRA Estimator (for training, maintains learnable variational parameters)
        self.variational_estimator = VariationalLoRAEstimator(
            num_lora_layers, lora_rank, device=self.output_device
        )
        
        # Move all components to output device
        self.input_norm.to(device=self.output_device, dtype=dtype)
        self.rff_projector.to(device=self.output_device, dtype=dtype)
        self.strategy_mlp.to(device=self.output_device, dtype=dtype)
        
    def extract_hidden_statistics(self, hidden_states, attention_mask):
        """Extract statistical features from hidden layers"""
        batch_size = hidden_states[0].shape[0]
        device = hidden_states[0].device
        
        # Find last valid token
        sequence_lengths = attention_mask.sum(dim=1)
        last_token_indices = sequence_lengths - 1
        batch_indices = torch.arange(batch_size, device=device)
        
        layer_stats = []
        for layer_hidden in hidden_states:
            last_token_hidden = layer_hidden[batch_indices, last_token_indices, :]
            mean_val = torch.mean(last_token_hidden, dim=-1, keepdim=True)
            std_val = torch.std(last_token_hidden, dim=-1, keepdim=True)
            layer_stats.append(torch.cat([mean_val, std_val], dim=-1))
        
        features = torch.cat(layer_stats, dim=-1)  # [batch, num_layers*2]
        return features
    
    def generate_personalized_params(self, hidden_features):
        """
        Generate personalized variational parameters via RFF
        
        Returns:
            alpha: [batch, num_layers, rank]
            beta: [batch, num_layers, rank]
        """
        batch_size = hidden_features.shape[0]
        
        # Normalization
        h_normed = self.input_norm(hidden_features)
        
        # RFF projection
        rff_feature = self.rff_projector(h_normed)
        
        # MLP parameter generation
        params = self.strategy_mlp(rff_feature)
        
        # Reshape
        params = params.view(batch_size, self.num_lora_layers, self.lora_rank, 2)
        
        alpha = torch.clamp(params[..., 0], min=0.01, max=100.0)
        beta = torch.clamp(params[..., 1], min=0.01, max=100.0)
        
        return alpha, beta
    
    def forward(self, input_ids, attention_mask,labels=None, return_variational_params=False):
        """
        Forward Propagation
        
        Args:
            input_ids: [batch, seq_len]
            attention_mask: [batch, seq_len]
            return_variational_params: Whether to return variational parameters
        
        Returns:
            logits: [batch, num_classes]
            variational_params: (alpha, beta) if return_variational_params
        """
        # Pass through base model
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels = labels,
            return_dict=True,
            output_hidden_states=True
        )
        
        logits = outputs.logits
        lm_loss = outputs.loss if labels is not None else None
        variational_params = None
        # Extract features
        hidden_states = outputs.hidden_states[1:]  # Exclude embedding layer

        if return_variational_params:
            # Extract statistical features
            hidden_features = self.extract_hidden_statistics(
                hidden_states, attention_mask
            )
            
            # Generate personalized parameters
            alpha, beta = self.generate_personalized_params(
                hidden_features.to(self.output_device)
            )

            variational_params = (alpha, beta)
        
        output_dict = {"logits":logits}
        if lm_loss is not None:
            output_dict["loss"] = lm_loss
        if variational_params is not None:
            output_dict["variational_params"] = variational_params

        return output_dict
    
class RFFProjection(nn.Module):
    """Random Fourier Features Projection"""
    def __init__(self, dim_in, dim_out=256):
        super().__init__()
        self.omega = nn.Parameter(torch.randn(dim_in, dim_out // 2) * 0.1)
        self.b = nn.Parameter(torch.rand(dim_out // 2) * 2 * math.pi)
        
    def forward(self, x):
        proj = torch.matmul(x, self.omega) + self.b
        cos_proj = torch.cos(proj)
        sin_proj = torch.sin(proj)
        rff_feature = torch.cat([cos_proj, sin_proj], dim=-1)
        return rff_feature * math.sqrt(2.0 / self.omega.shape[1])


# ============================================================================
# Helper Functions
# ============================================================================
def extract_lora_B_matrices(model):
    """Extract LoRA B Matrices"""
    lora_B_list = []
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            lora_B_list.append(param)
    return lora_B_list


def compute_smooth_l1_loss(lora_B_params, kappa, delta=1.0):
    """Smooth L1 Regularization"""
    total_loss = 0.0
    
    for layer_idx, B_matrix in enumerate(lora_B_params):
        B_l1_norm = torch.norm(B_matrix, p=1, dim=0)
        kappa_l = kappa[layer_idx]
        
        abs_B = B_l1_norm
        loss_l = torch.where(
            abs_B < delta,
            0.5 * (abs_B ** 2) / (kappa_l + 1e-8),
            (abs_B - 0.5 * delta) / (kappa_l + 1e-8)
        )
        
        total_loss += loss_l.sum()
    
    return total_loss
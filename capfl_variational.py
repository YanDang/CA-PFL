import torch
import torch.nn as nn
import numpy as np

# ============================================================================
# 1. Variational Bayesian LoRA Parameter Estimator
# ============================================================================
class VariationalLoRAEstimator(nn.Module):
    """
    Maintains independent variational papameters (alpha', beta') for each LoRA dimension in each layer
    """
    def __init__(self, num_layers, rank, device='cuda'):
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        self.device = device
        
        # Initialize variational parameters for each rank dimension in each layer
        # alpha': shape parameter of variational Gamma distribution
        # Initialized close to prior to avoid excessively large KL divergence at the beginning
        self.alpha_var = nn.Parameter(
            torch.ones(num_layers, rank, device=device) * 1.0
        )
        # beta': rate parameter of variational Gamma distribution
        self.beta_var = nn.Parameter(
            torch.ones(num_layers, rank, device=device) * 1.0
        )
        
    def forward(self):
        """Returns current variational distribution parameters"""
        # Ensure parameters are positive
        alpha = torch.clamp(self.alpha_var, min=0.01, max=100.0)
        beta = torch.clamp(self.beta_var, min=0.01, max=100.0)
        return alpha, beta
    
    def sample_kappa(self, alpha, beta):
        """
        Samples κ (sparsity parameter) from variational distribution
        Uses reparameterization trick
        """
        # Gamma分布采样: kappa ~ Gamma(alpha, beta)
        dist = torch.distributions.Gamma(alpha, beta)
        kappa = dist.rsample()  # 使用rsample支持梯度回传
        kappa = torch.clamp(kappa, min=1e-6, max=100.0)
        return kappa
    
    def compute_kl_divergence(self, alpha_var, beta_var, alpha_prior, beta_prior):
        """
        Computes KL divergence between variational distribution q(κ|α',β') and prior distribution p(κ|α_g,β_g)
        
        Correct KL divergence formula for Gamma distribution:
        KL(Gamma(α',β') || Gamma(α_g,β_g)) = 
            (α' - α_g)ψ(α') - log(Γ(α')/Γ(α_g)) + α_g*log(β'/β_g) + α'*(β_g - β')/β'
        """
        # Ensure all parameters are positive and restrict ranges to avoid numerical issues
        alpha_var = torch.clamp(alpha_var, min=0.1, max=50.0)
        beta_var = torch.clamp(beta_var, min=0.1, max=50.0)
        alpha_prior = torch.clamp(alpha_prior, min=0.1, max=50.0)
        beta_prior = torch.clamp(beta_prior, min=0.1, max=50.0)
        
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        
        # Calculate each term of KL divergence
        # term1: (α' - α_g)ψ(α')
        term1 = (alpha_var - alpha_prior) * torch.digamma(alpha_var)
        
        # term2: -log(Γ(α')/Γ(α_g)) = log(Γ(α_g)) - log(Γ(α'))
        term2 = torch.lgamma(alpha_prior) - torch.lgamma(alpha_var)
        
        # term3: α_g * log(β'/β_g)
        term3 = alpha_prior * torch.log((beta_var + eps) / (beta_prior + eps))
        
        # term4: α' * (β_g - β')/β'
        term4 = alpha_var * (beta_prior - beta_var) / (beta_var + eps)
        
        kl = term1 + term2 + term3 + term4
        
        # Sum over all layers and ranks
        return kl.sum()


# ============================================================================
# 2. Smooth-L1 Regularization
# ============================================================================
def compute_smooth_l1_loss(lora_B_params, kappa, delta=1.0):
    """
    Computes Smooth-L1 regularization loss
    Args:
        lora_B_params: list of LoRA B matrices, each with shape [d, r]
        kappa: sparsity parameter, with shape [num_layers, rank]
        delta: Huber loss threshold
    """
    total_loss = 0.0
    
    # Calculate LoRA matrices for each layer
    num_lora_matrices = len(lora_B_params)
    num_kappa_layers = kappa.shape[0]

    if num_kappa_layers == 0:
        return torch.tensor(0.0).to(kappa.device)
    matrices_per_layer = num_lora_matrices // num_kappa_layers

    for idx, B_matrix in enumerate(lora_B_params):
        
        # B_matrix shape: [d, r]
        # Calculate L1 norm for each column
        B_l1_norm = torch.norm(B_matrix, p=1, dim=0)  # shape: [r]
        kappa_idx = idx // 2
        # Get kappa values for this layer
        kappa_l = kappa[kappa_idx]  # shape: [r]
        
        # Smooth L1 loss
        abs_B = B_l1_norm
        loss_l = torch.where(
            abs_B < delta,
            0.5 * (abs_B ** 2) / (kappa_l + 1e-8),
            (abs_B - 0.5 * delta) / (kappa_l + 1e-8)
        )
        
        total_loss += loss_l.sum()
    
    return total_loss


# ============================================================================
# 3. Complete Variational Optimization Loss
# ============================================================================
def compute_variational_loss(lora_B_params, variational_estimator, 
                            global_alpha, global_beta, 
                            alpha_s=0.1, alpha_p=0.01):
    """
    Computes complete variational optimization loss: Smooth-L1 + KL divergence
    
    Args:
        lora_B_params: list of LoRA B matrices
        variational_estimator: Instance of VariationalLoRAEstimator
        global_alpha, global_beta: Global prior parameters from server
        alpha_s: Weight for Smooth-L1 loss
        alpha_p: Weight for KL divergence loss
    """
    # Get current variational parameters
    alpha_var, beta_var = variational_estimator()
    
    # Sample kappa
    kappa = variational_estimator.sample_kappa(alpha_var, beta_var)
    
    # 1. Smooth-L1 regularization loss
    smooth_l1_loss = compute_smooth_l1_loss(lora_B_params, kappa)
    
    # 2. KL divergence loss
    kl_loss = variational_estimator.compute_kl_divergence(
        alpha_var, beta_var, global_alpha, global_beta
    )
    
    # Total variational loss
    total_var_loss = alpha_s * smooth_l1_loss + alpha_p * kl_loss
    
    return total_var_loss, kappa, (alpha_var, beta_var)


# ============================================================================
# 4. Helper Function: Extract LoRA B Matrices
# ============================================================================
def extract_lora_B_matrices(model):
    """
    Extracts all LoRA B matrices from PEFT model
    
    Returns:
        list of tensors, each with shape [d, r]
    """
    lora_B_list = []
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            lora_B_list.append(param)
    return lora_B_list


# ============================================================================
# Test Code
# ============================================================================
if __name__ == "__main__":
    # Simulate parameters
    num_layers = 16
    rank = 8
    d = 3072  # hidden dimension
    
    # Create variational estimator
    var_estimator = VariationalLoRAEstimator(num_layers, rank)
    
    # Simulate LoRA B matrices
    lora_B_params = [torch.randn(d, rank, device='cuda', requires_grad=True) 
                     for _ in range(num_layers)]
    
    # Global prior (obtained from server)
    global_alpha = torch.ones(num_layers, rank, device='cuda') * 0.5
    global_beta = torch.ones(num_layers, rank, device='cuda') * 100.0
    
    # Calculate variational loss
    var_loss, kappa, (alpha_var, beta_var) = compute_variational_loss(
        lora_B_params, var_estimator, global_alpha, global_beta
    )
    
    print(f"Variational Loss: {var_loss.item():.4f}")
    print(f"Kappa shape: {kappa.shape}")
    print(f"Kappa sample values (layer 0): {kappa[0, :5]}")
    print(f"Alpha var (layer 0): {alpha_var[0, :5]}")
    print(f"Beta var (layer 0): {beta_var[0, :5]}")
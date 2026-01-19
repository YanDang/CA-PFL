import torch
import torch.nn as nn
import math

# ============================================================================
# 1. RFF Projection Layer (Unchanged, with added comments)
# ============================================================================
class RFFProjection(nn.Module):
    """
    Random Fourier Features Projection
    Maps high-dimensional inputs to a low-dimensional feature space based on Bochner's theorem
    """
    def __init__(self, dim_in, dim_out=256):
        super().__init__()
        # Random projection matrix ω
        self.omega = nn.Parameter(torch.randn(dim_in, dim_out // 2) * 0.1)
        # Phase shift b
        self.b = nn.Parameter(torch.rand(dim_out // 2) * 2 * math.pi)
        
    def forward(self, x):
        """
        x: [batch_size, dim_in]
        Returns: [batch_size, dim_out]
        """
        # Linear projection
        proj = torch.matmul(x, self.omega) + self.b  # [batch, dim_out//2]
        
        # Trigonometric nonlinear transformation
        cos_proj = torch.cos(proj)
        sin_proj = torch.sin(proj)
        
        # Concatenate and scale
        rff_feature = torch.cat([cos_proj, sin_proj], dim=-1)  # [batch, dim_out]
        rff_feature = rff_feature * math.sqrt(2.0 / self.omega.shape[1])
        
        return rff_feature


# ============================================================================
# 2. Layer-wise Personalization Strategy Generator (Key Modification)
# ============================================================================
class PersonalizationStrategyGenerator(nn.Module):
    """
    Generates layer-wise variational parameters (α, β) based on RFF features
    This is one of the core innovations of the paper
    
    Input: Statistical features extracted from model hidden layers
    Output: (α, β) parameters for each rank in each layer
    """
    def __init__(self, input_dim, num_layers, rank, 
                 rff_dim=256, mlp_hidden_dim=256):
        super().__init__()
        self.num_layers = num_layers
        self.rank = rank
        
        # Input normalization
        self.input_norm = nn.LayerNorm(input_dim, elementwise_affine=False)
        
        # RFF projection
        self.rff_projector = RFFProjection(input_dim, rff_dim)
        
        # MLP network: generates layer-wise parameters
        # Output dimension: num_layers * rank * 2 (each rank requires α and β)
        output_dim = num_layers * rank * 2
        
        self.mlp = nn.Sequential(
            nn.Linear(rff_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(mlp_hidden_dim, output_dim),
            nn.Softplus()  # 确保输出为正
        )
        
    def forward(self, hidden_features):
        """
        Args:
            hidden_features: [batch_size, input_dim] 
                           Statistical features extracted from model hidden layers
        
        Returns:
            alpha: [batch_size, num_layers, rank]
            beta: [batch_size, num_layers, rank]
        """
        batch_size = hidden_features.shape[0]
        
        # Normalize input
        h_normed = self.input_norm(hidden_features)
        
        # RFF projection
        rff_feature = self.rff_projector(h_normed)
        
        # MLP parameter generation
        params = self.mlp(rff_feature)  # [batch, num_layers * rank * 2]
        
        # Reshape to layer-wise structure
        params = params.view(batch_size, self.num_layers, self.rank, 2)
        
        # Separate α and β
        alpha = params[..., 0]  # [batch, num_layers, rank]
        beta = params[..., 1]   # [batch, num_layers, rank]
        
        # Restrict range
        alpha = torch.clamp(alpha, min=0.01, max=100.0)
        beta = torch.clamp(beta, min=0.01, max=100.0)
        
        return alpha, beta


# ============================================================================
# 3. Extract Statistical Features from Hidden Layers
# ============================================================================
def extract_hidden_statistics(hidden_states, attention_mask):
    """
    Extract statistical features from model hidden layers
    The paper uses mean and standard deviation for each layer
    
    Args:
        hidden_states: tuple of tensors, each with shape [batch, seq_len, hidden_size]
        attention_mask: [batch, seq_len]
    
    Returns:
        features: [batch, num_layers * 2]  (mean + std for each layer)
    """
    batch_size = hidden_states[0].shape[0]
    device = hidden_states[0].device
    
    # Calculate positions of valid tokens (last token)
    sequence_lengths = attention_mask.sum(dim=1)
    last_token_indices = sequence_lengths - 1
    batch_indices = torch.arange(batch_size, device=device)
    
    layer_stats = []
    
    # Calculate statistics for each layer
    for layer_hidden in hidden_states:
        # Extract hidden states of the last valid token
        last_token_hidden = layer_hidden[batch_indices, last_token_indices, :]
        # [batch, hidden_size]
        
        # Calculate mean and standard deviation
        mean_val = torch.mean(last_token_hidden, dim=-1, keepdim=True)  # [batch, 1]
        std_val = torch.std(last_token_hidden, dim=-1, keepdim=True)    # [batch, 1]
        
        layer_stats.append(torch.cat([mean_val, std_val], dim=-1))  # [batch, 2]
    
    # Concatenate statistics from all layers
    features = torch.cat(layer_stats, dim=-1)  # [batch, num_layers * 2]
    
    return features


# ============================================================================
# 4. Complete Personalized Parameter Generation Process
# ============================================================================
class ClientPersonalizationModule(nn.Module):
    """
    Client Personalization Module
    Integrates RFF and strategy generator to produce personalized variational parameters for each batch
    """
    def __init__(self, num_layers, rank, hidden_size, 
                 rff_dim=256, mlp_hidden_dim=256, device='cuda'):
        super().__init__()
        
        # Input dimension = 2 statistics (mean + std) per layer
        input_dim = num_layers * 2
        
        self.strategy_generator = PersonalizationStrategyGenerator(
            input_dim=input_dim,
            num_layers=num_layers,
            rank=rank,
            rff_dim=rff_dim,
            mlp_hidden_dim=mlp_hidden_dim
        )
        
        self.strategy_generator.to(device)
        
    def forward(self, hidden_states, attention_mask):
        """
        Args:
            hidden_states: All hidden layer outputs of the model
            attention_mask: attention mask
        
        Returns:
            alpha: [batch, num_layers, rank]
            beta: [batch, num_layers, rank]
        """
        # Extract statistical features
        features = extract_hidden_statistics(hidden_states, attention_mask)
        
        # Generate personalized parameters
        alpha, beta = self.strategy_generator(features)
        
        return alpha, beta


# ============================================================================
# Test Code
# ============================================================================
if __name__ == "__main__":
    # Simulate Llama-3.2 3B configuration
    num_layers = 16
    rank = 8
    hidden_size = 3072
    batch_size = 4
    seq_len = 512
    
    # Create personalization module
    personalization_module = ClientPersonalizationModule(
        num_layers=num_layers,
        rank=rank,
        hidden_size=hidden_size,
        device='cuda'
    )
    
    # Simulate hidden layer outputs
    hidden_states = tuple([
        torch.randn(batch_size, seq_len, hidden_size, device='cuda')
        for _ in range(num_layers)
    ])
    
    attention_mask = torch.ones(batch_size, seq_len, device='cuda')
    
    # Generate personalized parameters
    alpha, beta = personalization_module(hidden_states, attention_mask)
    
    print(f"Alpha shape: {alpha.shape}")  # [4, 16, 8]
    print(f"Beta shape: {beta.shape}")    # [4, 16, 8]
    print(f"\nSample alpha values (batch 0, layer 0): {alpha[0, 0, :]}")
    print(f"Sample beta values (batch 0, layer 0): {beta[0, 0, :]}")
    
    # Verify parameter range
    print(f"\nAlpha range: [{alpha.min():.4f}, {alpha.max():.4f}]")
    print(f"Beta range: [{beta.min():.4f}, {beta.max():.4f}]")
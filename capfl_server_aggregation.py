import torch
import torch.nn as nn
import copy

# ============================================================================
# 1. Improved Federated Server
# ============================================================================
class CAFLServer:
    """
    Federated Server for CA-PFL
    Maintains global model and global variational prior parameters
    """
    def __init__(self, global_model, num_layers, rank, device='cuda'):
        self.global_model = global_model
        self.num_layers = num_layers
        self.rank = rank
        self.device = device
        
        # Global variational prior parameters (α_g, β_g)
        # Uses more reasonable initialization: weakly informative prior
        # α=1.0 (shape parameter), β=1.0 (rate parameter) represents an exponential distribution with mean 1
        self.global_alpha = torch.ones(num_layers, rank, device=device) * 1.0
        self.global_beta = torch.ones(num_layers, rank, device=device) * 1.0
        
        # Records training history
        self.round_history = []
    
    def get_global_prior(self):
        """Returns global prior parameters"""
        return self.global_alpha.clone(), self.global_beta.clone()
    
    def aggregate_model_updates(self, client_updates, client_weights=None, 
                               lr_global=1.0):
        """
        Aggregates sparse model updates from clients
        
        Args:
            client_updates: list of dict, each dict is a client's sparse updates
            client_weights: list of float, client weights (based on data volume)
            lr_global: global learning rate
        
        Returns:
            updated_state_dict: updated global model state
        """
        num_clients = len(client_updates)
        
        # If no weights are provided, use uniform weights
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Get current global model state
        current_state = self.global_model.state_dict()
        
        # Create accumulator
        aggregated_delta = {}
        
        # Aggregate sparse updates
        for name in client_updates[0].keys():
            # Skip parameters that do not need aggregation
            if name not in current_state:
                continue
            
            # Weighted average of client updates
            weighted_sum = torch.zeros_like(
                current_state[name], 
                device=self.device
            )
            
            for client_idx, update in enumerate(client_updates):
                if name in update:
                    delta = update[name].to(self.device)
                    weighted_sum += delta * client_weights[client_idx]
            
            aggregated_delta[name] = weighted_sum
        
        # Update global model
        for name, delta in aggregated_delta.items():
            current_state[name] = current_state[name] + delta * lr_global
        
        self.global_model.load_state_dict(current_state)
        
        return current_state
    
    def aggregate_variational_parameters(self, client_alphas, client_betas, 
                                        client_weights=None):
        """
        Aggregates variational parameters from clients
        
        Args:
            client_alphas: list of tensors, each with shape [num_layers, rank]
            client_betas: list of tensors, each with shape [num_layers, rank]
            client_weights: list of float, client weights
        
        Returns:
            updated global_alpha, global_beta
        """
        num_clients = len(client_alphas)
        
        if client_weights is None:
            client_weights = [1.0 / num_clients] * num_clients
        
        # Normalize weights
        total_weight = sum(client_weights)
        client_weights = [w / total_weight for w in client_weights]
        
        # Weighted average
        new_alpha = torch.zeros_like(self.global_alpha)
        new_beta = torch.zeros_like(self.global_beta)
        
        for idx, (alpha, beta) in enumerate(zip(client_alphas, client_betas)):
            alpha = alpha.to(self.device)
            beta = beta.to(self.device)
            
            # Ensure correct shape: [num_layers, rank]
            if alpha.dim() == 3:
                # If batch dimension exists, take average
                alpha = alpha.mean(dim=0)
                beta = beta.mean(dim=0)
            
            # Verify shape
            assert alpha.shape == self.global_alpha.shape, \
                f"Alpha shape mismatch: expected {self.global_alpha.shape}, got {alpha.shape}"
            assert beta.shape == self.global_beta.shape, \
                f"Beta shape mismatch: expected {self.global_beta.shape}, got {beta.shape}"
            
            new_alpha += alpha * client_weights[idx]
            new_beta += beta * client_weights[idx]
        
        # Update with momentum (smooth transition)
        momentum = 0.9
        self.global_alpha = momentum * self.global_alpha + (1 - momentum) * new_alpha
        self.global_beta = momentum * self.global_beta + (1 - momentum) * new_beta
        
        # Ensure parameters are within reasonable range
        self.global_alpha = torch.clamp(self.global_alpha, min=0.1, max=50.0)
        self.global_beta = torch.clamp(self.global_beta, min=0.1, max=50.0)
        
        return self.global_alpha.clone(), self.global_beta.clone()
    
    def aggregate(self, client_updates, client_alphas, client_betas, 
                  client_weights=None, lr_global=1.0):
        """
        Complete aggregation process: model parameters + variational parameters
        
        Args:
            client_updates: list of dict, sparse model updates
            client_alphas: list of tensors, variational α parameters
            client_betas: list of tensors, variational β parameters
            client_weights: list of float, client weights
            lr_global: global learning rate
        
        Returns:
            global_state: updated global model state
            global_alpha, global_beta: updated global priors
        """
        # 1. Aggregate model updates
        global_state = self.aggregate_model_updates(
            client_updates, client_weights, lr_global
        )
        
        # 2. Aggregate variational parameters
        global_alpha, global_beta = self.aggregate_variational_parameters(
            client_alphas, client_betas, client_weights
        )
        
        # 3. Record statistics for this round
        round_info = {
            'global_alpha_mean': global_alpha.mean().item(),
            'global_alpha_std': global_alpha.std().item(),
            'global_beta_mean': global_beta.mean().item(),
            'global_beta_std': global_beta.std().item(),
        }
        self.round_history.append(round_info)
        
        print(f"[Server] Global variational parameters updated:")
        print(f"  Alpha: mean={round_info['global_alpha_mean']:.4f}, "
              f"std={round_info['global_alpha_std']:.4f}")
        print(f"  Beta:  mean={round_info['global_beta_mean']:.4f}, "
              f"std={round_info['global_beta_std']:.4f}")
        
        return global_state, global_alpha, global_beta
    
    def compute_client_divergence(self, client_alphas, client_betas):
        """
        Computes divergence between client variational distributions and global prior
        Used to monitor personalization level
        """
        divergences = []
        
        for alpha, beta in zip(client_alphas, client_betas):
            if len(alpha.shape) == 3:
                alpha = alpha.mean(dim=0)
                beta = beta.mean(dim=0)
            
            alpha = alpha.to(self.device)
            beta = beta.to(self.device)
            
            # Simplified KL divergence calculation
            kl = (self.global_alpha / alpha - torch.log(self.global_beta / beta) - 
                  torch.digamma(alpha) + 
                  torch.lgamma(self.global_alpha) - torch.lgamma(alpha) + 
                  alpha * (beta / self.global_beta - 1.0))
            
            divergences.append(kl.sum().item())
        
        return divergences


# ============================================================================
# 2. Communication Cost Calculation
# ============================================================================
def compute_communication_cost(sparse_updates, rff_params=None):
    """
    Calculates communication cost (MB)
    
    Args:
        sparse_updates: dict, sparse model updates
        rff_params: dict, RFF parameters (if need to transmit)
    
    Returns:
        cost_mb: communication cost (MB)
        breakdown: detailed breakdown
    """
    total_bytes = 0
    breakdown = {}
    
    # Calculate cost of sparse updates
    for name, param in sparse_updates.items():
        # Only transmit non-zero elements + indices
        nonzero_count = param.count_nonzero().item()
        
        # Data: float32 = 4 bytes
        data_bytes = nonzero_count * 4
        
        # Indices: int64 = 8 bytes (assuming COO format)
        index_bytes = nonzero_count * 8
        
        param_bytes = data_bytes + index_bytes
        total_bytes += param_bytes
        
        breakdown[name] = param_bytes / (1024 * 1024)  # Convert to MB
    
    # RFF parameters
    if rff_params is not None:
        rff_bytes = sum(p.numel() * 4 for p in rff_params.values())
        total_bytes += rff_bytes
        breakdown['rff_params'] = rff_bytes / (1024 * 1024)
    
    cost_mb = total_bytes / (1024 * 1024)
    
    return cost_mb, breakdown


# ============================================================================
# Test Code
# ============================================================================
if __name__ == "__main__":
    # Simulate global model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Linear(100, 50)
            self.layer2 = nn.Linear(50, 10)
        
        def forward(self, x):
            return self.layer2(self.layer1(x))
    
    global_model = SimpleModel()
    num_layers = 3
    rank = 8
    
    # Create server
    server = CAFLServer(global_model, num_layers, rank, device='cpu')
    
    # Simulate updates from 3 clients
    num_clients = 3
    client_updates = []
    client_alphas = []
    client_betas = []
    client_weights = [0.3, 0.5, 0.2]  # Weights based on data volume
    
    for i in range(num_clients):
        # Sparse updates
        update = {
            'layer1.weight': torch.randn(50, 100) * 0.01,
            'layer1.bias': torch.randn(50) * 0.01,
        }
        client_updates.append(update)
        
        # Variational parameters
        alpha = torch.rand(num_layers, rank) * 2.0
        beta = torch.rand(num_layers, rank) * 100.0
        client_alphas.append(alpha)
        client_betas.append(beta)
    
    # Execute federated aggregation
    print("Executing federated aggregation...")
    global_state, global_alpha, global_beta = server.aggregate(
        client_updates, client_alphas, client_betas, 
        client_weights, lr_global=1.0
    )
    
    # Calculate divergence
    divergences = server.compute_client_divergence(client_alphas, client_betas)
    print(f"\nClient divergences: {[f'{d:.4f}' for d in divergences]}")
    
    # 计算通信成本
    cost_mb, breakdown = compute_communication_cost(client_updates[0])
    print(f"\nSingle client communication cost: {cost_mb:.2f} MB")
    print("Detailed breakdown:")
    for name, cost in breakdown.items():
        print(f"  {name}: {cost:.2f} MB")
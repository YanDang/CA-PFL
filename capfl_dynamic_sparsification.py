import torch
import torch.nn as nn

# ============================================================================
# 1. Dynamic Sparsification based on κ
# ============================================================================
def dynamic_sparsification_with_kappa(delta_params, kappa, base_ratio=0.25):
    """
    Dynamically sparsify parameter updates based on learned κ values
    
    Larger κ value -> Less important dimension -> Easier to be sparsified
    Smaller κ value -> More important dimension -> Easier to be retained
    
    Args:
        delta_params: dict, {param_name: delta_tensor}
        kappa: tensor, [num_layers, rank] variational parameter
        base_ratio: float, base retention ratio
    
    Returns:
        sparse_delta: dict, sparsified parameter updates
        masks: dict, corresponding masks
    """
    sparse_delta = {}
    masks = {}
    
    layer_idx = 0
    
    for name, delta in delta_params.items():
        if "lora_B" in name:
            # LoRA B matrix: [d, r]
            # Use κ values of the corresponding layer
            kappa_l = kappa[layer_idx]  # [r]
            
            # Calculate importance score for each column (reciprocal of κ)
            # importance_scores = 1.0 / (kappa_l + 1e-8)  # [r]
            importance_scores = kappa_l
            
            # Dynamically adjust retention ratio based on importance
            # More important columns retain more elements
            column_sparse_delta = []
            column_masks = []
            
            for col_idx in range(delta.shape[1]):
                col = delta[:, col_idx]  # [d]
                
                # Retention ratio of this column is proportional to its importance
                col_importance = importance_scores[col_idx].item()
                
                # Normalize importance to [0.1, 0.5] range
                max_importance = importance_scores.max().item()
                min_importance = importance_scores.min().item()
                normalized_importance = (col_importance - min_importance) / \
                                       (max_importance - min_importance + 1e-8)
                
                # Dynamic retention ratio
                keep_ratio = base_ratio + normalized_importance * 0.25  # [0.25, 0.5]
                
                # Top-k selection
                k = max(int(col.numel() * keep_ratio), 1)
                _, top_indices = torch.abs(col).topk(k)
                
                # Create mask
                mask = torch.zeros_like(col)
                mask[top_indices] = 1.0
                
                column_sparse_delta.append(col * mask)
                column_masks.append(mask)
            
            # Reconstruct sparsified matrix
            sparse_matrix = torch.stack(column_sparse_delta, dim=1)
            mask_matrix = torch.stack(column_masks, dim=1)
            
            sparse_delta[name] = sparse_matrix
            masks[name] = mask_matrix
            
            layer_idx += 1
            
        elif "lora_A" in name:
            # LoRA A marix: [r, e]
            # Use κ values of the previous B matrix
            kappa_l = kappa[layer_idx - 1]  # [r]
            
            # Apply the same strategy to each row
            row_sparse_delta = []
            row_masks = []
            
            for row_idx in range(delta.shape[0]):
                row = delta[row_idx, :]  # [e]
                
                # Importance of this row
                row_importance = 1.0 / (kappa_l[row_idx] + 1e-8)
                
                # Normalization
                max_importance = (1.0 / (kappa_l + 1e-8)).max().item()
                min_importance = (1.0 / (kappa_l + 1e-8)).min().item()
                normalized_importance = (row_importance.item() - min_importance) / \
                                       (max_importance - min_importance + 1e-8)
                
                keep_ratio = base_ratio + normalized_importance * 0.25
                
                # Top-k selection
                k = max(int(row.numel() * keep_ratio), 1)
                _, top_indices = torch.abs(row).topk(k)
                
                mask = torch.zeros_like(row)
                mask[top_indices] = 1.0
                
                row_sparse_delta.append(row * mask)
                row_masks.append(mask)
            
            sparse_matrix = torch.stack(row_sparse_delta, dim=0)
            mask_matrix = torch.stack(row_masks, dim=0)
            
            sparse_delta[name] = sparse_matrix
            masks[name] = mask_matrix
            
        else:
            # Fixed strategy for other parameters (e.g., RFF-related)
            k = max(int(delta.numel() * base_ratio), 1)
            flat_delta = delta.flatten()
            _, top_indices = torch.abs(flat_delta).topk(k)
            
            mask = torch.zeros_like(flat_delta)
            mask[top_indices] = 1.0
            mask = mask.view_as(delta)
            
            sparse_delta[name] = delta * mask
            masks[name] = mask
    
    return sparse_delta, masks


# ============================================================================
# 2. Error Feedback Mechanism
# ============================================================================
class ErrorCompensationBuffer:
    """
    Maintain residual buffer for each parameter
    Implements the error feedback mechanism from the paper
    """
    def __init__(self):
        self.buffer = {}
    
    def update(self, param_name, error):
        """Update residual error"""
        self.buffer[param_name] = error.cpu()
    
    def get(self, param_name, device):
        """Get residual error"""
        if param_name in self.buffer:
            return self.buffer[param_name].to(device)
        else:
            return None
    
    def clear(self):
        """Clear buffer"""
        self.buffer.clear()


# ============================================================================
# 3. Complete Sparse Update Process
# ============================================================================
def compute_sparse_update_with_error_feedback(
    model, initial_state, kappa, error_buffer, base_ratio=0.25
):
    """
    Compute sparsified parameter updates combined with error compensation
    
    Args:
        model: Current model
        initial_state: Initial state before training
        kappa: [num_layers, rank] sparsity parameter
        error_buffer: Instance of ErrorCompensationBuffer
        base_ratio: Base retention ratio
    
    Returns:
        sparse_updates: Sparsified updates (for upload)
        error_buffer: Updated error compensation buffer
    """
    sparse_updates = {}
    
    with torch.no_grad():
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            
            # 1. Compute full update amount
            delta = param.data - initial_state[name].to(param.device)
            
            # 2. Add residual error from previous round (Error Compensation)
            prev_error = error_buffer.get(name, param.device)
            if prev_error is not None:
                delta += prev_error
            
            # Collect delta for all LoRA parameters
            delta_dict = {name: delta}
            
            # 3. Dynamic sparsification
            if "lora" in name:
                sparse_dict, _ = dynamic_sparsification_with_kappa(
                    delta_dict, kappa, base_ratio
                )
                sparse_delta = sparse_dict[name]
            else:
                # Fixed strategy for non-LoRA parameters
                k = max(int(delta.numel() * base_ratio), 1)
                flat_delta = delta.flatten()
                _, top_indices = torch.abs(flat_delta).topk(k)
                mask = torch.zeros_like(flat_delta)
                mask[top_indices] = 1.0
                sparse_delta = (delta.flatten() * mask).view_as(delta)
            
            # 4. Compute new residual error
            residual = delta - sparse_delta
            error_buffer.update(name, residual)
            
            # 5. Save sparse updates
            if sparse_delta.count_nonzero() > 0:
                sparse_updates[name] = sparse_delta.cpu()
    
    return sparse_updates, error_buffer


# ============================================================================
# 4. Sparsity Statistics Calculation
# ============================================================================
def compute_sparsity_statistics(sparse_updates):
    """
    Calculate statistics of sparse updates
    """
    total_params = 0
    nonzero_params = 0
    
    for name, param in sparse_updates.items():
        total_params += param.numel()
        nonzero_params += param.count_nonzero().item()
    
    sparsity_ratio = 1.0 - (nonzero_params / total_params)
    
    stats = {
        'total_params': total_params,
        'nonzero_params': nonzero_params,
        'sparsity_ratio': sparsity_ratio,
        'compression_ratio': total_params / max(nonzero_params, 1)
    }
    
    return stats


if __name__ == "__main__":
    # Simulate parameters
    num_layers = 3
    rank = 8
    d = 3072
    e = 3072
    
    # Simulate LoRA parameter updates
    delta_params = {
        'lora_B.0': torch.randn(d, rank, device='cuda'),
        'lora_A.0': torch.randn(rank, e, device='cuda'),
        'lora_B.1': torch.randn(d, rank, device='cuda'),
        'lora_A.1': torch.randn(rank, e, device='cuda'),
        'lora_B.2': torch.randn(d, rank, device='cuda'),
        'lora_A.2': torch.randn(rank, e, device='cuda'),
    }
    
    # Simulate κ values (different layers have different importance)
    kappa = torch.tensor([
        [0.5, 0.8, 1.2, 0.3, 1.5, 0.7, 0.9, 1.0],  # Layer 0
        [1.0, 0.4, 0.6, 1.8, 0.5, 1.1, 0.8, 0.9],  # Layer 1
        [0.7, 1.3, 0.4, 0.9, 1.6, 0.6, 1.0, 0.8],  # Layer 2
    ], device='cuda')
    
    # Execute dynamic sparsification
    sparse_delta, masks = dynamic_sparsification_with_kappa(
        delta_params, kappa, base_ratio=0.25
    )
    
    # Statistics
    print("Sparsification Results:")
    for name in sparse_delta.keys():
        original = delta_params[name]
        sparse = sparse_delta[name]
        sparsity = 1.0 - sparse.count_nonzero().item() / sparse.numel()
        print(f"{name}: Sparsity = {sparsity:.2%}, "
              f"Non-zero elements = {sparse.count_nonzero()}/{sparse.numel()}")
    
    # Test error compensation
    print("\nTesting Error Compensation Mechanism:")
    error_buffer = ErrorCompensationBuffer()
    
    # First round
    for name, delta in delta_params.items():
        residual = delta - sparse_delta[name]
        error_buffer.update(name, residual)
        print(f"First round residual {name}: norm = {residual.norm().item():.4f}")
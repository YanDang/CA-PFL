import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import get_linear_schedule_with_warmup
from capfl_dynamic_sparsification import dynamic_sparsification_with_kappa, ErrorCompensationBuffer, compute_sparse_update_with_error_feedback
from capfl_variational import compute_smooth_l1_loss
# ============================================================================
# Complete Client Training Class
# ============================================================================
class CAFLClient:
    """
    CA-PFL Client
    Implements the complete training process, including:
    1. variational parameter learning
    2. Dynamic sparsification
    3. Error compensation
    """
    def __init__(self, client_id, dataset, collate_fn, device='cuda'):
        self.client_id = client_id
        self.dataset = dataset
        self.data_collator = collate_fn
        self.device = device
        self.global_step = 0
        
        # Error compensation buffer
        self.error_buffer = ErrorCompensationBuffer()
        
    def train_local(self, model, global_weights, server_alpha, server_beta,
                   epochs, batch_size, learning_rate, alpha_s=0.1, alpha_p=0.01, 
                   alpha_f=0.01,warmup_steps=100):
        """
        Local training
        
        Args:
            model: CA-PFL model
            global_weights: Global model weights
            server_alpha, server_beta: Global prior parameters from server
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate
            alpha_s: Weight for Smooth-L1 loss
            alpha_p: Weight for KL divergence loss
            alpha_f: Weight for federated constraint loss
        
        Returns:
            sparse_updates: Sparsified parameter updates
            avg_alpha, avg_beta: Average variational parameters
        """
        # Load global weights
        model.load_state_dict(global_weights)
        model.train()
        
        # Create data loader
        train_loader = DataLoader(
            self.dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self.data_collator
        )
        
        # Optimizer (only optimize trainable parameters)
        trainable_params = filter(lambda p: p.requires_grad, model.parameters())
        optimizer = optim.AdamW(trainable_params, lr=learning_rate, weight_decay=0.01)
        
        num_update_steps_per_epoch = len(train_loader)
        max_train_steps = epochs * num_update_steps_per_epoch

        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=max_train_steps
        )

        # Save initial state for computing updates
        initial_state = {k: v.clone().cpu() for k, v in model.state_dict().items()}
        
        # Move server priors to correct device
        server_alpha = server_alpha.to(self.device)
        server_beta = server_beta.to(self.device)
        
        # For accumulating variational parameters (sum first, average later)
        sum_alpha = None
        sum_beta = None
        total_samples = 0
        
        print(f"\n=== Client {self.client_id} Starting Training ===")
        
        for epoch in range(epochs):
            total_loss = 0
            total_ce_loss = 0
            total_var_loss = 0
            total_fed_loss = 0
            
            progress_bar = tqdm(
                train_loader, 
                desc=f"Client {self.client_id} Epoch {epoch+1}/{epochs}"
            )
            
            for batch in progress_bar:
                self.global_step += 1
                
                # Prepare inputs
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                optimizer.zero_grad()
                
                # Forward pass (get logits and variational parameters)
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                    return_variational_params=True
                )
                logits = outputs["logits"]
                alpha_batch,beta_batch = outputs["variational_params"]
                
                # 1. Clssification
                ce_loss = outputs["loss"]
                # 2. Variational loss
                # Extract LoRA B matrices
                lora_B_list = extract_lora_B_matrices(model)
                
                # Get parameters from vari
                var_alpha, var_beta = model.variational_estimator()
                
                # Sample kappa for Smooth-L1
                kappa = model.variational_estimator.sample_kappa(var_alpha, var_beta)
                
                # Smooth-L1 regularization
                smooth_l1_loss = compute_smooth_l1_loss(lora_B_list, kappa)
                
                # KL divergence (ensure correct parameters are used)
                kl_loss = model.variational_estimator.compute_kl_divergence(
                    var_alpha, var_beta, server_alpha, server_beta
                )
                if self.global_step > warmup_steps:
                    var_loss = alpha_s * smooth_l1_loss + alpha_p * kl_loss
                else:
                    var_loss = torch.tensor(0.0,device=self.device)
                # 3. Federated constraint loss (FedProx)
                proximal_term = 0.0
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        initial_param = initial_state[name].to(param.device)
                        diff = param - initial_param
                        proximal_term += (diff ** 2).sum()
                
                fed_loss = 0.5 * alpha_f * proximal_term
                
                # Total loss
                loss = ce_loss + var_loss + fed_loss
                
                # Backward pass
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                lr_scheduler.step()
                
                # Accumulate statistics
                total_loss += loss.item()
                total_ce_loss += ce_loss.item()
                total_var_loss += var_loss.item()
                total_fed_loss += fed_loss.item()
                
                # Accumulate variational parameters (average over batch then sum)
                batch_avg_alpha = alpha_batch.mean(dim=0).detach().cpu()  # [num_layers, rank]
                batch_avg_beta = beta_batch.mean(dim=0).detach().cpu()    # [num_layers, rank]
                
                if sum_alpha is None:
                    sum_alpha = batch_avg_alpha * input_ids.size(0)
                    sum_beta = batch_avg_beta * input_ids.size(0)
                else:
                    sum_alpha += batch_avg_alpha * input_ids.size(0)
                    sum_beta += batch_avg_beta * input_ids.size(0)
                
                total_samples += input_ids.size(0)
                
                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'ce': f'{ce_loss.item():.4f}',
                    'var': f'{var_loss.item():.4f}',
                    'fed': f'{fed_loss.item():.4f}'
                })
            
            # Print epoch statistics
            n_batches = len(train_loader)
            print(f"Epoch {epoch+1} Average Losses:")
            print(f"  Total: {total_loss/n_batches:.4f}")
            print(f"  CE: {total_ce_loss/n_batches:.4f}")
            print(f"  Var: {total_var_loss/n_batches:.4f}")
            print(f"  Fed: {total_fed_loss/n_batches:.4f}")
        
        # Calculate average variational parameters (weighted average)
        avg_alpha = sum_alpha / total_samples  # [num_layers, rank]
        avg_beta = sum_beta / total_samples    # [num_layers, rank]
        
        print(f"\nClient {self.client_id} Average Variational Parameters:")
        print(f"  Shape: alpha={avg_alpha.shape}, beta={avg_beta.shape}")
        print(f"  Alpha range: [{avg_alpha.min():.4f}, {avg_alpha.max():.4f}]")
        print(f"  Beta range: [{avg_beta.min():.4f}, {avg_beta.max():.4f}]")
        
        # Sparsification and error compensation
        print(f"\nClient {self.client_id} Performing sparsification...")
        sparse_updates = self._compute_sparse_updates(
            model, initial_state, kappa.cpu()
        )
        
        # Calculate communication cost
        comm_cost = self._compute_communication_cost(sparse_updates)
        print(f"Communication Cost: {comm_cost:.2f} MB")
        
        return sparse_updates, (avg_alpha, avg_beta)
    
    def _compute_sparse_updates(self, model, initial_state, kappa, base_ratio=0.25):
        """
        Compute sparsified parameter updates (combined with error compensation)
        """
        sparse_updates,self.error_buffer = compute_sparse_update_with_error_feedback(
            model=model,
            initial_state=initial_state,
            kappa=kappa,
            error_buffer=self.error_buffer,
            base_ratio=base_ratio
        )
        return sparse_updates
    
    def _sparsify_lora_B(self, delta, kappa_layer, base_ratio):
        """Sparsify LoRA B matrix"""
        # delta: [d, r], kappa_layer: [r]
        sparse_cols = []
        
        for col_idx in range(delta.shape[1]):
            col = delta[:, col_idx]
            
            # Calculate keep ratio based on κ
            importance = 1.0 / (kappa_layer[col_idx].item() + 1e-8)
            keep_ratio = base_ratio + importance * 0.25
            keep_ratio = min(keep_ratio, 0.5)
            
            # Top-k selection
            k = max(int(col.numel() * keep_ratio), 1)
            _, top_idx = torch.abs(col).topk(k)
            
            mask = torch.zeros_like(col)
            mask[top_idx] = 1.0
            sparse_cols.append(col * mask)
        
        return torch.stack(sparse_cols, dim=1)
    
    def _sparsify_lora_A(self, delta, kappa_layer, base_ratio):
        """Sparsify LoRA A matrix"""
        # delta: [r, e], kappa_layer: [r]
        sparse_rows = []
        
        for row_idx in range(delta.shape[0]):
            row = delta[row_idx, :]
            
            importance = 1.0 / (kappa_layer[row_idx].item() + 1e-8)
            keep_ratio = base_ratio + importance * 0.25
            keep_ratio = min(keep_ratio, 0.5)
            
            k = max(int(row.numel() * keep_ratio), 1)
            _, top_idx = torch.abs(row).topk(k)
            
            mask = torch.zeros_like(row)
            mask[top_idx] = 1.0
            sparse_rows.append(row * mask)
        
        return torch.stack(sparse_rows, dim=0)
    
    def _sparsify_fixed(self, delta, ratio):
        """Fixed ratio sparsification"""
        k = max(int(delta.numel() * ratio), 1)
        flat_delta = delta.flatten()
        _, top_idx = torch.abs(flat_delta).topk(k)
        
        mask = torch.zeros_like(flat_delta)
        mask[top_idx] = 1.0
        
        return (flat_delta * mask).view_as(delta)
    
    def _compute_communication_cost(self, sparse_updates):
        """Calculate communication cos (MB)"""
        total_bytes = 0
        
        for param in sparse_updates.values():
            nonzero_count = param.count_nonzero().item()
            # Data + indices
            total_bytes += nonzero_count * (4 + 8)
        
        return total_bytes / (1024 * 1024)


def extract_lora_B_matrices(model):
    """Extract LoRA B matrices"""
    lora_B_list = []
    for name, param in model.named_parameters():
        if "lora_B" in name and param.requires_grad:
            lora_B_list.append(param)
    return lora_B_list
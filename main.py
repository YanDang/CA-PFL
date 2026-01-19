"""
CA-PFL: Client-adaptive Parameter-efficient Fine-tuning

Integrates all core mechanisms of the paper:
1. Variational Bayesian LoRA rank estimation
2. RFF personalized strategy generation
3. Dynamic sparsification
4. Error compensation
5. Federated aggregation
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader,Subset
from datasets import load_dataset,concatenate_datasets 
from transformers import AutoTokenizer, AutoConfig, DataCollatorForSeq2Seq
from tqdm import tqdm
import numpy as np

# Import tools
from capfl_server_aggregation import CAFLServer
from capfl_integrated_model import LlamaWithCAFLHead
from capfl_client_training import CAFLClient
from capfl_final_test import run_evaluation
import os
import argparse
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# Random seed
RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
TORCH_DTYPE = torch.bfloat16

def parse_args():
    """
    Parse command line arguments for CA-PFL training configuration
    """
    parser = argparse.ArgumentParser(description='CA-PFL Training Configuration')
    
    # ============================
    # Base path settings
    # ============================
    parser.add_argument('--model_path', 
                        type=str, 
                        default="meta-llama/Llama-3.2-3B",
                        help='Path to the base Llama model')
    parser.add_argument('--csqa_data_path', 
                        type=str, 
                        default="tasksource/commonsense_qa_2.0",
                        help='Path to CommonsenseQA dataset')
    parser.add_argument('--obqa_data_path', 
                        type=str, 
                        default="allenai/openbookqa",
                        help='Path to OpenBookQA dataset')
    
    # ============================
    # Model settings
    # ============================
    parser.add_argument('--mlp_hidden_dim', 
                        type=int, 
                        default=256,
                        help='Hidden dimension of MLP for personalization strategy')
    parser.add_argument('--rff_output_dim', 
                        type=int, 
                        default=256,
                        help='Output dimension of RFF projection layer')
    parser.add_argument('--lora_rank', 
                        type=int, 
                        default=8,
                        help='Rank of LoRA adaptation')
    parser.add_argument('--batch_size', 
                        type=int, 
                        default=8,
                        help='Training batch size per client')
    parser.add_argument('--max_length', 
                        type=int, 
                        default=512,
                        help='Maximum sequence length for tokenization')
    
    # ============================
    # Training settings
    # ============================
    parser.add_argument('--client_epochs', 
                        type=int, 
                        default=2,
                        help='Number of training epochs per client')
    parser.add_argument('--learning_rate', 
                        type=float, 
                        default=1e-5,
                        help='Learning rate for client training')
    parser.add_argument('--warmup_steps', 
                        type=int, 
                        default=500,
                        help='Number of warmup steps for learning rate scheduler')
    
    # ============================
    # Loss weight settings
    # ============================
    parser.add_argument('--alpha_s', 
                        type=float, 
                        default=0.01,
                        help='Weight for Smooth-L1 regularization loss')
    parser.add_argument('--alpha_p', 
                        type=float, 
                        default=0.01,
                        help='Weight for KL divergence loss')
    parser.add_argument('--alpha_f', 
                        type=float, 
                        default=0.1,
                        help='Weight for FedProx loss')
    
    # ============================
    # Federated learning settings
    # ============================
    parser.add_argument('--num_clients', 
                        type=int, 
                        default=10,
                        help='Total number of clients in federated learning')
    parser.add_argument('--server_epochs', 
                        type=int, 
                        default=10,
                        help='Number of federated rounds (server epochs)')
    parser.add_argument('--fraction', 
                        type=float, 
                        default=0.5,
                        help='Fraction of clients to sample per round')
    
    # ============================
    # Sparsification settings
    # ============================
    parser.add_argument('--base_sparsity_ratio', 
                        type=float, 
                        default=0.15,
                        help='Base sparsity ratio for model updates')
    
    # Parse arguments
    args = parser.parse_args()
    
    # Derived parameter (clients per round)
    args.clients_per_round = int(args.num_clients * args.fraction)
    
    return args

# ============================================================================
# Data Preparation
# ============================================================================
def partition_data(dataset, num_clients, alpha=0.5):
    """
    Non-IID data partitioning
    """
    num_samples = len(dataset)
    indices = np.arange(num_samples)
    
    shard_size = num_samples // num_clients
    client_datasets = []
    random_indices = np.random.permutation(indices)
    
    for i in range(num_clients):
        start_idx = i * shard_size
        end_idx = (i + 1) * shard_size if i < num_clients - 1 else num_samples
        subset_indices = random_indices[start_idx:end_idx]
        
        client_datasets.append(Subset(dataset, subset_indices))
    
    return client_datasets

class EarlyStopping:
    """
    Early stopping mechanism: Stop training when validation loss does not decrease for 'patience' epochs
    """
    def __init__(self, patience=3, verbose=False, delta=0, path='checkpoint.pt', trace_func=print):
        """
        Args:
            patience (int): Number of epochs to wait after last validation loss improvement
            verbose (bool): If True, print validation information at each step
            delta (float): Minimum change in the monitored quantity to qualify as an improvement
            path (str): Path to save the best model checkpoint
            trace_func (function): Function to output log information
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta
        self.path = path
        self.trace_func = trace_func

    def __call__(self, val_loss, model):
        # Score is negative loss because we want to find the minimum loss
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            # Current score is not better than best_score (plus delta threshold)
            self.counter += 1
            if self.verbose:
                self.trace_func(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            # Improvement detected
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Save model when validation loss decreases'''
        if self.verbose:
            self.trace_func(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        
        # For your LlamaWithCAFLHead, saving state_dict is the safest approach
        # If your model contains base_model(LoRA) + other layers, state_dict will save everything
        torch.save(model.state_dict(), self.path)
        self.val_loss_min = val_loss

class DatasetTemplate:
    def __init__(self, data, tokenizer, max_length,dataset_name):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.dataset_name = dataset_name
    def preprocess_function(self, examples):
        tokenizer = self.tokenizer
        first_key = list(examples.keys())[0] 
        batch_size = len(examples[first_key])
        input_ids_list = []
        attention_mask_list = []
        labels_list = []
        
        for i in range(batch_size):
            # -------------------------------------------------------
            # Branch A: Process OpenBookQA (multiple choice questions with choices)
            # -------------------------------------------------------
            if self.dataset_name == "obqa":
                q = examples["question_stem"][i]
                choices = examples["choices"][i]
                fact1 = examples["fact1"][i]
                answer_key = examples["answerKey"][i]
                
                # Construct Prompt: Question + Options
                prompt = f"Question: {q}\n"
                for label, text in zip(choices["label"], choices["text"]):
                    prompt += f"{label}. {text}\n"
                prompt += f"Because {fact1}\n"
                prompt += "Answer: "
                
                # Construct Response: e.g. " A"
                response = f"{answer_key}" + tokenizer.eos_token

            # -------------------------------------------------------
            # Branch B: Process CSQA 2.0 (Yes/No questions without choices)
            # -------------------------------------------------------
            elif self.dataset_name == "csqa": # CSQA uses 'question' field
                q = examples["question"][i]
                ans = examples["answer"][i] # ans is usually 'yes' or 'no'
                
                # Construct Prompt: Only question
                # We can add a hint to let the model know to answer yes/no
                prompt = f"Question: {q}\nAnswer: "
                
                # Construct Response: Directly " yes" or " no"
                response = f"{ans}" + tokenizer.eos_token
                
            else:
                raise KeyError(f"Unknown data format. Keys: {examples.keys()}")

            # -------------------------------------------------------
            # Universal Tokenize and Mask Logic (shared by both branches)
            # -------------------------------------------------------
            prompt_ids = tokenizer(prompt, add_special_tokens=False)["input_ids"]
            response_ids = tokenizer(response, add_special_tokens=False)["input_ids"]
            
            input_ids = [tokenizer.bos_token_id] + prompt_ids + response_ids
            attention_mask = [1] * len(input_ids)
            
            # Only calculate Loss for Response, mask Prompt part (-100)
            labels = [-100] * (len(prompt_ids) + 1) + response_ids
            
            # Truncation handling
            max_len = 256
            if len(input_ids) > max_len:
                input_ids = input_ids[:max_len]
                attention_mask = attention_mask[:max_len]
                labels = labels[:max_len]
                
            input_ids_list.append(input_ids)
            attention_mask_list.append(attention_mask)
            labels_list.append(labels)
        
        return {
            "input_ids": input_ids_list,
            "attention_mask": attention_mask_list,
            "labels": labels_list
        }
    def get_dataset(self):
        tokenized_data = self.data.map(
            self.preprocess_function,
            batched=True,
            remove_columns=self.data.column_names
        )
        return tokenized_data

def preprocess_datasets(tokenizer, data_path_dict:dict, max_length=512):
    """Load and preprocess datasets"""
    train_data_list = []
    val_data_list = []
    test_data_dict = {}
    for dataset_name,data_path in data_path_dict.items():
        dataset = load_dataset(data_path)
        # obqa and csqa as training sets
        if dataset_name in ["obqa","csqa"]:
            train_data = dataset["train"]
            train_processor = DatasetTemplate(train_data, tokenizer, max_length,dataset_name)
            train_data_list.append(train_processor.get_dataset())
        val_data = dataset["validation"]
        val_processor = DatasetTemplate(val_data,tokenizer,max_length,dataset_name)
        val_data_list.append(val_processor.get_dataset())
        if "test" in dataset.keys():
            test_data_dict[dataset_name] = dataset["test"]
        else:
            test_data_dict[dataset_name] = dataset["validation"]
    # Merge training sets and shuffle
    combined_train_dataset = concatenate_datasets(train_data_list).shuffle(seed=RANDOM_SEED)
    combined_val_dataset = concatenate_datasets(val_data_list).shuffle(seed=RANDOM_SEED)

    return combined_train_dataset, combined_val_dataset,test_data_dict

# ============================================================================
# Main Training Process
# ============================================================================
def main(args):
    print("="*80)
    print("CA-PFL: Client-adaptive Parameter-efficient Fine-tuning")
    print("="*80)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")
    
    # 1. Load tokenizer
    print("\n[1/6] Loading Tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, padding=True)
    
    # 2. Load datasets
    print("\n[2/6] Loading datasets...")
    dataset_train, dataset_val,dataset_test_dict = preprocess_datasets(
        tokenizer, {"obqa":args.obqa_data_path,"csqa":args.csqa_data_path}, args.max_length
    )
    # Partition data clients
    client_datasets = partition_data(dataset_train, num_clients=args.num_clients)
    print(f"Data partitioned for {args.num_clients} clients")
    print(f"Training set size: {len(dataset_train)}, Validation set size: {len(dataset_val)}")
    
    # Create validation data loader
    val_loader = DataLoader(
        dataset_val,
        batch_size=args.batch_size,
        collate_fn=data_collator
    )
    
    # 3. Create global model
    print("\n[3/6] Creating global model...")
    config = AutoConfig.from_pretrained(args.model_path)
    num_lora_layers = config.num_hidden_layers
    
    # Need to use modified model class here
    # Provide pseudocode framework due to large model size
    print(f"Model configuration:")
    print(f"  Number of hidden layers: {num_lora_layers}")
    print(f"  Hidden dimension: {config.hidden_size}")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  RFF dimension: {args.rff_output_dim}")
    
    global_model = LlamaWithCAFLHead(
        model_path=args.model_path,
        llm_last_size=config.hidden_size,
        num_lora_layers=num_lora_layers,
        lora_rank=args.lora_rank,
        mlp_hidden_dim=args.mlp_hidden_dim,
        rff_dim_out=args.rff_output_dim,
        dtype=TORCH_DTYPE
    )
    
    # 4. Create server
    print("\n[4/6] Initializing federated server...")
    server = CAFLServer(
        global_model=global_model,
        num_layers=num_lora_layers,
        rank=args.lora_rank,
        device=device
    )
    print(f"Server initialization completed")
    print(f"  Global prior parameters: alpha_g=0.5, beta_g=100.0")
    
    # 5. Create clients
    print("\n[5/6] Creating clients...")
    clients = []
    for i in range(args.num_clients):
        client = CAFLClient(
            client_id=i,
            dataset=client_datasets[i],
            collate_fn=data_collator,
            device=device
        )
        clients.append(client)
        print(f"  Client {i}: {len(client_datasets[i])} samples")
    
    # 6. Federated training loop
    print("\n[6/6] Starting federated training...")
    print("="*80)
    
    # Initialize EarlyStopping
    early_stopping = EarlyStopping(patience=3, verbose=True, path='best_model.pt')

    for round_idx in range(args.server_epochs):
        print(f"\n{'='*80}")
        print(f"Federated training round {round_idx + 1}/{args.server_epochs}")
        print(f"{'='*80}")
        current_alpha_s = min(args.alpha_s, args.alpha_s * (round_idx / 5.0)) # 5 rounds warm-up
        current_alpha_f = args.alpha_f # FedProx can be kept constant or introduced later
        # Get global weights and prior parameters
        global_weights = server.global_model.state_dict()
        global_alpha, global_beta = server.get_global_prior()
        
        # Randomly select clients
        selected_indices = np.random.choice(
            args.num_clients, args.clients_per_round, replace=False
        )
        print(f"\nSelected clients: {selected_indices.tolist()}")
        
        # Collect client updates
        collected_updates = []
        collected_alphas = []
        collected_betas = []
        collected_weights = []
        
        for idx in selected_indices:
            client = clients[idx]
            client_data_size = len(client.dataset)
            
            print(f"\n--- 客户端 {idx} 训练中 ---")
            
            # Local training
            sparse_update, (avg_alpha, avg_beta) = client.train_local(
                model=server.global_model,
                global_weights=global_weights,
                server_alpha=global_alpha,
                server_beta=global_beta,
                epochs=args.client_epochs,
                batch_size=args.batch_size,
                learning_rate=args.learning_rate,
                alpha_s=current_alpha_s,
                alpha_p=args.alpha_p,
                alpha_f=current_alpha_f,
                warmup_steps=args.warmup_steps
            )
            
            collected_updates.append(sparse_update)
            collected_alphas.append(avg_alpha)
            collected_betas.append(avg_beta)
            collected_weights.append(client_data_size)
            
            print(f"Client {idx} training completed")
        
        # Server aggregation
        print(f"\n{'='*80}")
        print("Server aggregating updates...")
        global_state, global_alpha, global_beta = server.aggregate(
            client_updates=collected_updates,
            client_alphas=collected_alphas,
            client_betas=collected_betas,
            client_weights=collected_weights,
            lr_global=1.0
        )
        
        # Calculate client divergence (monitor personalization level)
        divergences = server.compute_client_divergence(
            collected_alphas, collected_betas
        )
        print(f"Client divergences: {[f'{d:.4f}' for d in divergences]}")
        
        # Validation
        print(f"\n{'='*80}")
        print("Global model validation...")
        
        server.global_model.eval()
        total_val_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = server.global_model(input_ids, attention_mask,labels=labels)
                logits = outputs["logits"]
                loss = outputs["loss"]
                
                total_val_loss += loss.item()
                # 2. Get predicted Token ID (take maximum value in last dimension)
                # Shape: [Batch, Seq_Len]
                preds = torch.argmax(logits, dim=-1) 
                
                # 3. Shift processing
                # Predictions: remove last token (since it predicts future unknown token)
                shift_preds = preds[..., :-1] 
                # True labels: remove first token (since first token cannot be predicted from previous)
                shift_labels = labels[..., 1:]
                
                # 4. Create mask, only calculate parts where label is not -100 (i.e., only Answer part)
                mask = shift_labels != -100
                
                # 5. Calculate number of correct predictions
                # Only compare positions where mask is True
                correct_tokens = (shift_preds[mask] == shift_labels[mask]).sum().item()
                total_tokens = mask.sum().item() # Total number of valid tokens
                
                correct += correct_tokens
                total += total_tokens

        avg_val_loss = total_val_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        
        print(f"\nRound {round_idx + 1} validation results:")
        print(f"  Validation loss: {avg_val_loss:.4f}")
        print(f"  Validation accuracy: {accuracy:.2f}%")
        early_stopping(avg_val_loss, server.global_model)
        
        if early_stopping.early_stop:
            print("\n" + "!"*30)
            print("Early Stopping Triggered")
            print("!"*30)
            break
        print(f"\nRound {round_idx + 1} completed!")
    
    print("\n" + "="*80)
    print("Federated training completed!")
    print("Loading best model for testing...")
    server.global_model.load_state_dict(torch.load('best_model.pt'))
    print("="*80)
    print("\n" + "="*80)
    print("Starting final test set evaluation...")
    run_evaluation(server.global_model.base_model, tokenizer, dataset_test_dict["obqa"],"obqa")

if __name__ == "__main__":
    args = parse_args()
    main(args)
# CA-PFL: Client-Adaptive Parameter-Efficient Fine-Tuning for Personalized Federated Learning

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Framework](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org/)
[![PEFT](https://img.shields.io/badge/PEFT-LoRA-green)](https://github.com/huggingface/peft)

> **Context-Aware Personalized Federated Learning framework for LLMs via Variational Bayesian LoRA, RFF-based personalization, and Sparse Optimization.**

This repository contains the official PyTorch implementation of **CA-PFL**. CA-PFL is a novel framework designed to address the challenges of deploying Large Language Models (LLMs) in heterogeneous federated environments. It dynamically generates personalization strategies and optimizes communication efficiency through variational sparse control and random Fourier features.

## 🚀 Key Features

* **Client-Adaptive Sparsity**: Dynamically assigns LoRA rank based on local data feature distributions to establish sparsity.
* **Variational Bayesian LoRA**: Utilizes variational Bayesian priors to estimate optimal rank configurations compatible with federated aggregation.
* **RFF-based Personalization**: Reduces communication cost and enables efficient personalization through Random Fourier Feature (RFF) projection.
* **Dynamic Sparsification & Error Compensation**: Implements dynamic pruning based on learned $\kappa$ values and maintains an error buffer to ensure convergence.
* **Communication Efficiency**: Reduces communication overhead by approximately **78%** compared to baselines while maintaining high accuracy.

## 📂 Repository Structure

The codebase is organized as follows:

```text
.
├── main.py                       # Entry point: Argument parsing, data partitioning, and federated training loop
├── capfl_client_training.py      # Client-side logic: Local training, loss calculation, and sparse updates
├── capfl_server_aggregation.py   # Server-side logic: Aggregating sparse model updates and variational priors
├── capfl_integrated_model.py     # Model definition: Wraps Llama with RFF and Variational Heads
├── capfl_variational.py          # Variational Inference: KL Divergence calculation and Kappa sampling
├── capfl_dynamic_sparsification.py # Sparsification logic: Dynamic pruning and error compensation buffer
├── capfl_rff_personalization.py  # Personalization: Random Fourier Features (RFF) projection module
└── capfl_final_test.py           # Evaluation script for Accuracy (Logits Method)
```
## 🛠️ Installation

1. Clone the repository
```text
git clone https://github.com/YanDang/CA-PFL.git
cd CA-PFL
```
2. Install dependencies It is recommended to use a virtual environment.
```text
pip install -r requirements.txt
```
## 🏃 Usage
### Quick Start
To run the federated learning simulation with default settings (Llama-3.2-3B on OpenBookQA):
```text
python main.py \
    --model_path "meta-llama/Llama-3.2-3B" \
    --num_clients 10 \
    --server_epochs 10 \
    --lora_rank 8 \
    --alpha_s 0.01 \
    --alpha_p 0.01 \
    --rff_output_dim 256
```
### Key Arguments configuration (main.py)
|Argument|Default|Description|
|:-:|:-:|:-:|
|--model_path|meta-llama/Llama-3.2-3B|Path to the base LLM|
|--alpha_s|0.01|Weight for Smooth-L1 regularization (controls sparsity)|
|--alpha_p|0.01|Weight for KL divergence loss (variational constraint)|
|--alpha_f|0.1|Weight for FedProx constraint term|
|--rff_output_dim|256|Dimension for Random Fourier Features projection|
|--base_sparsity_ratio|0.15|Base retention ratio for dynamic sparsification|
|--num_clients|10|Total number of clients in the federated system|
## 📊 Performance
CA-PFL demonstrates superior performance on heterogeneous non-IID data setups compared to state-of-the-art baselines like FedEx-LoRA and LoRA-FAIR.
### Accuracy vs. Communication Cost
<table>
  <thead>
    <tr>
      <th style="text-align: left;">Model</th>
      <th style="text-align: left;">Method</th>
      <th style="text-align: left;">Accuracy (Avg)</th>
      <th style="text-align: left;">Comm. Cost (MB/round)</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td style="text-align: left; vertical-align: top;" rowspan="2">Llama-3.2 3B</td>
      <td style="text-align: left;">FedEx-LoRA</td>
      <td style="text-align: left;">76.59%</td>
      <td style="text-align: left;">41.29</td>
    </tr>
    <tr>
      <td style="text-align: left;"><strong>CA-PFL (Ours)</strong></td>
      <td style="text-align: left;">77.77%</td>
      <td style="text-align: left;">7.19</td>
    </tr>
    <tr>
      <td style="text-align: left; vertical-align: top;" rowspan="2">Mistral-7B</td>
      <td style="text-align: left;">FedEx-LoRA</td>
      <td style="text-align: left;">78.66%</td>
      <td style="text-align: left;">95.6</td>
    </tr>
    <tr>
      <td style="text-align: left;"><strong>CA-PFL (Ours)</strong></td>
      <td style="text-align: left;">79.89%</td>
      <td style="text-align: left;">21.4</td>
    </tr>
  </tbody>
</table>
Results based on experiments on OBQA, BoolQ, WinoG, and ARC datasets.

## 🧩 Methodology Highlights
### 1. Variational LoRA Estimator  
We model the LoRA rank selection as a variational inference problem. The model learns a distribution over the sparsity parameter $\kappa$ using a Gamma hyper-prior:  

$$ p(\kappa|\alpha, \beta) = \prod \mathcal{G}(\kappa_i | \alpha, \beta) $$  

### 2. RFF Personalization  
To handle the "cold start" and personalization on edge devices, we use Random Fourier Features to project hidden states into a low-dimensional manifold:  

$$ z_i = \sqrt{\frac{2}{D}} [\cos(\omega^T h_i + b), \sin(\omega^T h_i + b)] $$  

This acts as a lightweight "Personalized Modulator" that generates the specific $\alpha$ and $\beta$ parameters for each client.  

### 3. Dynamic Sparsification  
Parameters are updated based on the learned importance score $\kappa$. **Smaller $\kappa$ implies less importance** of the corresponding LoRA parameter, allowing for aggressive sparsification (i.e., pruning low-importance parameters). An Error Compensation Buffer (ECB) is introduced to carry over residual errors of pruned parameters to the next round, preventing irreversible information loss during federated updates.
## 📜 Citation  
If you find this code useful for your research, please cite our paper
```

import torch
from tqdm import tqdm

# logits打分
import torch
from tqdm import tqdm

import torch
from tqdm import tqdm

def run_evaluation(model, tokenizer, dataset, dataset_name="obqa"):
    print(f"——Accuracy Evaluation [{dataset_name}] (Logits Method)——\n")
    model.eval()
    
    # ============================
    # 1. Prepare Token ID
    # ============================
    # Llama is sensitive to spaces, usually followed by a space after Answer:
    
    # For OBQA (multiple choice)
    choices_map = ["A", "B", "C", "D"]
    choice_ids = [tokenizer.encode(c, add_special_tokens=False)[0] for c in choices_map]
    idx_to_label = {0: "A", 1: "B", 2: "C", 3: "D"}
    
    correct_count = 0
    total_count = 0

    for sample in tqdm(dataset, desc="Evaluating"):
        try:
            # ============================
            # 2. Construct Prompt and Get Ground Truth
            # ============================
            if dataset_name == "obqa":
                # OBQA logic (unchanged)
                q = sample["question_stem"]
                choices = sample["choices"]
                fact1 = sample["fact1"]
                ground_truth = sample["answerKey"].strip() # "A", "B"...
                
                prompt = f"Question: {q}\n"
                for label, text in zip(choices["label"], choices["text"]):
                    prompt += f"{label}. {text}\n"
                prompt += f"Because: {fact1}\n"
                prompt += "Answer: "
                
            else:
                continue

            # ============================
            # 3. Forward Propagation
            # ============================
            inputs = tokenizer(prompt, return_tensors="pt",padding=False).to(model.device)
            
            with torch.no_grad():
                # Compatibility handling
                if hasattr(model, "base_model"):
                    outputs = model.base_model(**inputs)
                else:
                    outputs = model(**inputs)
                
                # Get logits of the last token
                # [1, vocab_size]
                next_token_logits = outputs.logits[0, -1, :]


            # ============================
            # 4. Compare Probabilities and Make Prediction
            # ============================
            predicted_answer = None

            if dataset_name == "obqa":
                # Compare scores of A, B, C, D
                candidate_scores = [next_token_logits[tid].item() for tid in choice_ids]
                best_idx = candidate_scores.index(max(candidate_scores))
                predicted_answer = idx_to_label[best_idx] # "A", "B"...
                
                # Judge correctness (case-)
                if predicted_answer.upper() == ground_truth.upper():
                    correct_count += 1

            total_count += 1

        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
            
    if total_count == 0:
        return 0.0

    accuracy = correct_count / total_count
    print(f"\n✅ Final Validation Set Accuracy [{dataset_name}]: {accuracy:.2%}")
    print(f"Correct: {correct_count} / {total_count}")
    
    return accuracy
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, GPT2Tokenizer
from peft import PeftModel

from src.config import Config
from src.models import build_model_and_head
from src.utils import load_checkpoint

def run_evaluation():
    print("--- Starting Inference / Evaluation Session ---")
    config = Config()
    
    # 1. Load tokenizer robustly
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
    except Exception as e:
        print(f"AutoTokenizer failed ({e}). Falling back to GPT2Tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_ID)
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. Build model and load checkpoints
    # Check if checkpoint exists
    if not os.path.exists(config.OUTPUT_DIR):
        print(f"Error: Checkpoint directory {config.OUTPUT_DIR} does not exist. Train the model first.")
        return
        
    print("Building base model and head...")
    model, classification_head = build_model_and_head(config)
    
    print("Loading fine-tuned LoRA adapters and head parameters...")
    # Load LoRA weights
    model.load_adapter(config.OUTPUT_DIR, adapter_name="default")
    # Load classification head
    load_checkpoint(model, classification_head, config.OUTPUT_DIR)
    
    model.eval()
    classification_head.eval()
    
    # 3. Test transcripts
    test_transcripts = [
        # Example 1: Passed QA
        "Agent: Hello! Thank you for contacting customer support. How can I help you today?\n"
        "Customer: Can you help me check if my order #94821 is delayed?\n"
        "Agent: Yes, I checked order #94821. It was shipped yesterday and is on track to arrive by Friday.\n"
        "Agent: Is there anything else I can assist you with today? Have a wonderful day!",
        
        # Example 2: Failed QA - Rude greeting
        "Agent: What do you want?\n"
        "Customer: Can you help me check if my order #94821 is delayed?\n"
        "Agent: It was shipped yesterday.\n"
        "Agent: Bye."
    ]
    
    print("\n--- Evaluating Test Cases ---")
    for idx, transcript in enumerate(test_transcripts):
        print(f"\n[Test Case {idx+1}]")
        print(transcript)
        print("-" * 50)
        
        prompt = (
            "<|start_of_role|>user<|end_of_role|>"
            "Analyze the following customer service transcript and evaluate it.\n\n"
            f"Transcript:\n{transcript}\n\n"
            "Evaluate if this transcript passes or fails QA standards and explain why."
            "<|start_of_role|>assistant<|end_of_role|>"
        )
        
        inputs = tokenizer(prompt, return_tensors="pt").to(config.DEVICE)
        prompt_len = inputs["input_ids"].shape[1]
        
        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                output_hidden_states=True
            )
            
            last_hidden_states = outputs.hidden_states[-1]
            last_token_hidden = last_hidden_states[:, -1, :]
            last_token_hidden = last_token_hidden.to(classification_head.dense.weight.dtype)
            
            class_logits = classification_head(last_token_hidden)
            probs = torch.softmax(class_logits, dim=-1)[0]
            pred_class = torch.argmax(class_logits, dim=-1).item()
            
            # Generate rationale
            generation = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=60,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False
            )
            generated_ids = generation[0][prompt_len:]
            generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
            
        print("[Inference Output]")
        print(f"  QA Status prediction: {'PASSED' if pred_class == 1 else 'FAILED'}")
        print(f"  Confidence: PASSED ({probs[1]:.2%}) | FAILED ({probs[0]:.2%})")
        print(f"  Generated Rationale: {generated_text.strip()}")
        print("=" * 60)

if __name__ == "__main__":
    run_evaluation()

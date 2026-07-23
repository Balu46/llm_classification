import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
from transformers import AutoTokenizer, GPT2Tokenizer
from torch.utils.data import DataLoader

from src.config import Config
from src.data import prepare_autoqa_dataset, AutoQADataset, DualHeadCollate
from src.models import build_model_and_head
from src.training import train_one_epoch, evaluate_model
from src.utils import save_checkpoint

def main():
    print("--- Starting AutoQA PEFT Dual-Head Train Process ---")
    config = Config()
    
    # 1. Initialize tokenizer robustly
    print(f"Loading tokenizer: {config.MODEL_ID}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config.MODEL_ID)
        print("Successfully loaded tokenizer via AutoTokenizer.")
    except Exception as e:
        print(f"AutoTokenizer failed ({e}). Falling back to GPT2Tokenizer...")
        tokenizer = GPT2Tokenizer.from_pretrained(config.MODEL_ID)
        print("Successfully loaded tokenizer via GPT2Tokenizer fallback.")
        
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
        
    # 2. Fetch and augment representative Hugging Face dataset
    raw_data = prepare_autoqa_dataset(num_samples=config.NUM_SAMPLES)
    
    # Train / Val Split
    val_size = int(len(raw_data) * config.TEST_SPLIT)
    train_raw = raw_data[:-val_size]
    val_raw = raw_data[-val_size:]
    
    train_dataset = AutoQADataset(train_raw, tokenizer)
    val_dataset = AutoQADataset(val_raw, tokenizer)
    
    collator = DualHeadCollate(tokenizer.pad_token_id)
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, collate_fn=collator)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, collate_fn=collator)
    
    # 3. Model & classification head creation
    model, classification_head = build_model_and_head(config)
    
    # 4. Group parameters and build optimizer
    trainable_lora_params = [p for p in model.parameters() if p.requires_grad]
    head_params = list(classification_head.parameters())
    
    optimizer_grouped_parameters = [
        {"params": trainable_lora_params, "lr": config.BACKBONE_LR},
        {"params": head_params, "lr": config.HEAD_LR}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, weight_decay=config.WEIGHT_DECAY)
    
    print("\n--- Training Loop ---")
    for epoch in range(config.EPOCHS):
        # Train epoch
        train_metrics = train_one_epoch(model, classification_head, train_loader, optimizer, config)
        # Validation epoch
        val_metrics = evaluate_model(model, classification_head, val_loader, config)
        
        print(f"Epoch {epoch+1}/{config.EPOCHS}")
        print(f"  [Train] Loss: {train_metrics['total_loss']:.4f} (Class Accuracy: {train_metrics['accuracy']:.2%})")
        print(f"          Class Loss: {train_metrics['class_loss']:.4f} | Gen Loss: {train_metrics['gen_loss']:.4f}")
        print(f"  [Val]   Loss: {val_metrics['total_loss']:.4f} (Class Accuracy: {val_metrics['accuracy']:.2%})")
        print(f"          Class Loss: {val_metrics['class_loss']:.4f} | Gen Loss: {val_metrics['gen_loss']:.4f}")
        print("-" * 50)
        
    # 5. Save model checkpoint
    print("Saving checkpoints...")
    save_checkpoint(model, classification_head, config.OUTPUT_DIR)
    print("Done!")

if __name__ == "__main__":
    main()

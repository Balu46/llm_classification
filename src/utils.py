import os
import torch

def save_checkpoint(model, classification_head, output_dir):
    """
    Saves LoRA adapter weights and custom classification head parameters.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Save PEFT LoRA adapters
    model.save_pretrained(output_dir)
    # Save Custom Classification Head
    head_path = os.path.join(output_dir, "classification_head.pt")
    torch.save(classification_head.state_dict(), head_path)
    print(f"Checkpoint successfully saved to: {output_dir}")

def load_checkpoint(model, classification_head, checkpoint_dir):
    """
    Loads custom classification head weights. LoRA adapters should be loaded
    via model.load_adapter() or PeftModel.from_pretrained().
    """
    head_path = os.path.join(checkpoint_dir, "classification_head.pt")
    if os.path.exists(head_path):
        # We match map_location so it loads on CPU or CUDA correctly
        device = next(classification_head.parameters()).device
        state_dict = torch.load(head_path, map_location=device)
        classification_head.load_state_dict(state_dict)
        print("Classification head weights loaded successfully!")
    else:
        print("Warning: Classification head weights file not found.")

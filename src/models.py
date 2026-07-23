import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, prepare_model_for_kbit_training

class ClassificationHead(nn.Module):
    def __init__(self, hidden_size, num_classes=2):
        super().__init__()
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(hidden_size, num_classes)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        x = self.dense(hidden_states)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x

def build_model_and_head(config):
    """
    Loads base Granite Causal LM in 4-bit, configures LoRA,
    and constructs the classification head.
    """
    print("Configuring 4-bit Quantization (bitsandbytes)...")
    q_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    print(f"Loading pre-trained base model: {config.MODEL_ID}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        config.MODEL_ID,
        quantization_config=q_config,
        device_map="auto"
    )
    
    base_model = prepare_model_for_kbit_training(base_model)
    
    print("Applying LoRA Config...")
    peft_config = LoraConfig(
        r=config.LORA_R,
        lora_alpha=config.LORA_ALPHA,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=config.LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    model = get_peft_model(base_model, peft_config)
    model.gradient_checkpointing_enable()
    hidden_size = model.config.hidden_size
    
    print(f"Initializing Classification Head (Hidden Size = {hidden_size})...")
    classification_head = ClassificationHead(hidden_size, num_classes=2)
    classification_head = classification_head.to(config.DEVICE).to(torch.float32)
    
    return model, classification_head

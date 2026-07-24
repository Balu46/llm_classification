import os
import yaml
import torch

class Config:
    def __init__(self, config_path="config.yaml"):
        # Default fallback values
        self.MODEL_ID = "ibm-granite/granite-3.0-3b-a800m-instruct"
        self.LORA_R = 8
        self.LORA_ALPHA = 16
        self.LORA_DROPOUT = 0.05
        
        self.DATASET_ID = "bitext/Bitext-customer-support-llm-chatbot-training-dataset"
        self.NUM_SAMPLES = 200
        self.TEST_SPLIT = 0.15
        self.CLIP_DIALOGUE = True
        
        self.EPOCHS = 3
        self.BATCH_SIZE = 2
        self.BACKBONE_LR = 2e-4
        self.HEAD_LR = 1e-3
        self.WEIGHT_DECAY = 0.01
        self.ALPHA = 1.0
        self.BETA = 1.0
        self.STOP_GRADIENT_FOR_CLASSIFICATION = True
        
        self.OUTPUT_DIR = "./checkpoints/autoqa_peft_dual_head"
        
        # Load from YAML if exists
        if os.path.exists(config_path):
            try:
                with open(config_path, "r") as f:
                    yaml_config = yaml.safe_load(f)
                    self._parse_yaml(yaml_config)
                print(f"Loaded configuration from {config_path}")
            except Exception as e:
                print(f"Warning: Failed to load config from {config_path} ({e}). Using default values.")
                
        # Device configuration
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def _parse_yaml(self, yaml_data):
        if not yaml_data:
            return
            
        model_cfg = yaml_data.get("model", {})
        self.MODEL_ID = model_cfg.get("model_id", self.MODEL_ID)
        self.LORA_R = model_cfg.get("lora_r", self.LORA_R)
        self.LORA_ALPHA = model_cfg.get("lora_alpha", self.LORA_ALPHA)
        self.LORA_DROPOUT = model_cfg.get("lora_dropout", self.LORA_DROPOUT)
        
        data_cfg = yaml_data.get("dataset", {})
        self.DATASET_ID = data_cfg.get("dataset_id", self.DATASET_ID)
        self.NUM_SAMPLES = data_cfg.get("num_samples", self.NUM_SAMPLES)
        self.TEST_SPLIT = data_cfg.get("test_split", self.TEST_SPLIT)
        self.CLIP_DIALOGUE = data_cfg.get("clip_dialogue", self.CLIP_DIALOGUE)
        
        train_cfg = yaml_data.get("training", {})
        self.EPOCHS = train_cfg.get("epochs", self.EPOCHS)
        self.BATCH_SIZE = train_cfg.get("batch_size", self.BATCH_SIZE)
        self.BACKBONE_LR = float(train_cfg.get("backbone_lr", self.BACKBONE_LR))
        self.HEAD_LR = float(train_cfg.get("head_lr", self.HEAD_LR))
        self.WEIGHT_DECAY = float(train_cfg.get("weight_decay", self.WEIGHT_DECAY))
        self.ALPHA = float(train_cfg.get("alpha", self.ALPHA))
        self.BETA = float(train_cfg.get("beta", self.BETA))
        self.STOP_GRADIENT_FOR_CLASSIFICATION = train_cfg.get(
            "stop_gradient_for_classification", self.STOP_GRADIENT_FOR_CLASSIFICATION
        )
        
        out_cfg = yaml_data.get("output", {})
        self.OUTPUT_DIR = out_cfg.get("output_dir", self.OUTPUT_DIR)

import time
import torch
import torch.nn as nn

def train_one_epoch(model, classification_head, dataloader, optimizer, config):
    model.train()
    classification_head.train()
    
    class_criterion = nn.CrossEntropyLoss()
    lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    epoch_class_loss = 0.0
    epoch_gen_loss = 0.0
    epoch_total_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    for step, batch in enumerate(dataloader):
        optimizer.zero_grad()
        
        input_ids = batch["input_ids"].to(config.DEVICE)
        attention_mask = batch["attention_mask"].to(config.DEVICE)
        clf_indices = batch["clf_indices"].to(config.DEVICE)
        class_labels = batch["class_labels"].to(config.DEVICE)
        lm_labels = batch["labels"].to(config.DEVICE)
        
        # Get raw backbone and lm_head to avoid computing massive logit tensors for prompt tokens
        raw_backbone = model.base_model.model.model
        lm_head = model.base_model.model.lm_head
        
        outputs = raw_backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Last Layer outputs: [batch, seq, hidden]
        last_hidden_states = outputs.last_hidden_state
        
        # Extract representations of the last token in the prompt (before generation)
        batch_size = input_ids.shape[0]
        batch_indices = torch.arange(batch_size, device=config.DEVICE)
        clf_hidden = last_hidden_states[batch_indices, clf_indices]
        
        # Apply Stop-Gradient if configured
        if config.STOP_GRADIENT_FOR_CLASSIFICATION:
            clf_hidden = clf_hidden.detach()
            
        clf_hidden = clf_hidden.to(classification_head.dense.weight.dtype)
        
        # Classification Logits and Loss
        class_logits = classification_head(clf_hidden)
        loss_class = class_criterion(class_logits, class_labels)
        
        # Language Modeling Logits and Loss - Sliced to save VRAM
        shift_hidden = last_hidden_states[..., :-1, :]
        shift_labels = lm_labels[..., 1:]
        
        # Optimize memory by masking out prompt tokens (-100) before passing to lm_head
        mask = (shift_labels != -100)
        active_hidden = shift_hidden[mask]
        active_labels = shift_labels[mask]
        
        if active_labels.numel() > 0:
            active_logits = lm_head(active_hidden)
            loss_gen = lm_criterion(active_logits, active_labels)
        else:
            loss_gen = torch.tensor(0.0, device=config.DEVICE, requires_grad=True)
        
        # Multi-task loss
        total_loss = (config.ALPHA * loss_class) + (config.BETA * loss_gen)
        
        total_loss.backward()
        optimizer.step()
        
        # Metrics
        preds = torch.argmax(class_logits, dim=-1)
        correct_preds += (preds == class_labels).sum().item()
        total_preds += class_labels.size(0)
        
        epoch_class_loss += loss_class.item()
        epoch_gen_loss += loss_gen.item()
        epoch_total_loss += total_loss.item()
        
        # Log progress every 100 steps
        if (step + 1) % 100 == 0:
            print(f"  Step {step + 1}/{len(dataloader)} | Loss: {total_loss.item():.4f} (Class: {loss_class.item():.4f}, Gen: {loss_gen.item():.4f})", flush=True)
            
    return {
        "class_loss": epoch_class_loss / len(dataloader),
        "gen_loss": epoch_gen_loss / len(dataloader),
        "total_loss": epoch_total_loss / len(dataloader),
        "accuracy": correct_preds / total_preds
    }

def evaluate_model(model, classification_head, dataloader, config):
    model.eval()
    classification_head.eval()
    
    class_criterion = nn.CrossEntropyLoss()
    lm_criterion = nn.CrossEntropyLoss(ignore_index=-100)
    
    epoch_class_loss = 0.0
    epoch_gen_loss = 0.0
    epoch_total_loss = 0.0
    correct_preds = 0
    total_preds = 0
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(config.DEVICE)
            attention_mask = batch["attention_mask"].to(config.DEVICE)
            clf_indices = batch["clf_indices"].to(config.DEVICE)
            class_labels = batch["class_labels"].to(config.DEVICE)
            lm_labels = batch["labels"].to(config.DEVICE)
            
            # Get raw backbone and lm_head
            raw_backbone = model.base_model.model.model
            lm_head = model.base_model.model.lm_head
            
            outputs = raw_backbone(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            last_hidden_states = outputs.last_hidden_state
            batch_size = input_ids.shape[0]
            batch_indices = torch.arange(batch_size, device=config.DEVICE)
            clf_hidden = last_hidden_states[batch_indices, clf_indices]
            clf_hidden = clf_hidden.to(classification_head.dense.weight.dtype)
            
            class_logits = classification_head(clf_hidden)
            loss_class = class_criterion(class_logits, class_labels)
            
            # Language Modeling Logits and Loss - Sliced to save VRAM
            shift_hidden = last_hidden_states[..., :-1, :]
            shift_labels = lm_labels[..., 1:]
            
            # Optimize memory by masking out prompt tokens (-100) before passing to lm_head
            mask = (shift_labels != -100)
            active_hidden = shift_hidden[mask]
            active_labels = shift_labels[mask]
            
            if active_labels.numel() > 0:
                active_logits = lm_head(active_hidden)
                loss_gen = lm_criterion(active_logits, active_labels)
            else:
                loss_gen = torch.tensor(0.0, device=config.DEVICE)
            
            total_loss = (config.ALPHA * loss_class) + (config.BETA * loss_gen)
            
            preds = torch.argmax(class_logits, dim=-1)
            correct_preds += (preds == class_labels).sum().item()
            total_preds += class_labels.size(0)
            
            epoch_class_loss += loss_class.item()
            epoch_gen_loss += loss_gen.item()
            epoch_total_loss += total_loss.item()
            
    return {
        "class_loss": epoch_class_loss / len(dataloader),
        "gen_loss": epoch_gen_loss / len(dataloader),
        "total_loss": epoch_total_loss / len(dataloader),
        "accuracy": correct_preds / total_preds
    }

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
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        
        # Last Layer outputs: [batch, seq, hidden]
        last_hidden_states = outputs.hidden_states[-1]
        
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
        
        # Language Modeling Logits and Loss
        lm_logits = outputs.logits
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = lm_labels[..., 1:].contiguous()
        
        loss_gen = lm_criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        
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
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True
            )
            
            last_hidden_states = outputs.hidden_states[-1]
            batch_size = input_ids.shape[0]
            batch_indices = torch.arange(batch_size, device=config.DEVICE)
            clf_hidden = last_hidden_states[batch_indices, clf_indices]
            clf_hidden = clf_hidden.to(classification_head.dense.weight.dtype)
            
            class_logits = classification_head(clf_hidden)
            loss_class = class_criterion(class_logits, class_labels)
            
            lm_logits = outputs.logits
            shift_logits = lm_logits[..., :-1, :].contiguous()
            shift_labels = lm_labels[..., 1:].contiguous()
            
            loss_gen = lm_criterion(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
            
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

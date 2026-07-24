import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset
from huggingface_hub import hf_hub_download
from transformers import GPT2Tokenizer

# Set seed for repeatability
random.seed(42)

def prepare_autoqa_dataset(num_samples=200, clip_dialogue=True):
    """
    Downloads the Strova Customer Support Conversations dataset using huggingface_hub,
    groups turns by conversation, formats dialogue transcripts, and assigns QA labels.
    Uses vectorized pandas grouping to efficiently parse all conversations.
    """
    print("Loading HF dataset 'strova-ai/customer_support_conversations_dataset' via hf_hub_download...")
    try:
        csv_path = hf_hub_download(
            repo_id="strova-ai/customer_support_conversations_dataset",
            filename="customer_support_data.csv",
            repo_type="dataset"
        )
        
        # Read the entire dataset
        df = pd.read_csv(csv_path)
        print(f"CSV loaded successfully! Total turns: {len(df)}")
    except Exception as e:
        print(f"Failed to load dataset ({e}). Generating fallback synthetic data...")
        return generate_fallback_synthetic_data(num_samples)

    df["role_label"] = df["role"].str.lower().map({"customer": "Customer", "agent": "Agent"}).fillna("Agent")
    df["formatted_text"] = df["role_label"] + ": " + df["text"].astype(str)
    
    # Sort and group by conv_id, aggregating turns into a list
    df = df.sort_values(by=["conv_id", "turn_index"])
    grouped = df.groupby("conv_id").agg({
        "formatted_text": list,
        "outcome": "first",
        "primary_intent": "first"
    }).reset_index()

    # Convert to list of dicts for native iteration
    records = grouped.to_dict("records")
    qa_data = []
    
    for row in records:
        turns = row["formatted_text"]
        
        # Apply dialogue clipping if enabled and conversation is long
        if clip_dialogue and len(turns) > 6:
            clipped_turns = turns[:3] + ["Agent: [... dialogue turns omitted for brevity ...]"] + turns[-3:]
        else:
            clipped_turns = turns
            
        transcript = "\n".join(clipped_turns)
        outcome = str(row["outcome"]).strip()
        intent = str(row["primary_intent"]).replace("_", " ").strip()
        
        # Map outcome to Pass/Fail label and generate a custom rationale
        if outcome == "Resolved":
            label = 1
            rationale = f"The agent successfully resolved the customer's query regarding {intent} without escalation."
        elif outcome == "Escalated":
            label = 0
            rationale = f"The agent failed the QA audit because the query regarding {intent} had to be escalated to a specialist or supervisor."
        elif outcome == "Closed - No Action":
            label = 0
            rationale = f"The agent failed the QA audit because the conversation was closed without taking action on the customer's {intent} query."
        elif outcome == "Pending Vendor":
            label = 0
            rationale = f"The agent failed the QA audit because the resolution for {intent} is pending vendor response, leaving it unresolved."
        elif outcome == "Pending Customer":
            label = 0
            rationale = f"The agent failed the QA audit because the ticket remains pending customer feedback and has not been resolved."
        else:
            label = 0
            rationale = f"The agent failed the QA audit because the conversation outcome was marked as {outcome} and did not reach resolution."
            
        qa_data.append({
            "transcript": transcript,
            "label": label,
            "rationale": rationale
        })

    # Shuffle and select num_samples if positive
    random.shuffle(qa_data)
    if num_samples is not None and num_samples > 0:
        qa_data = qa_data[:num_samples]
        print(f"Prepared {len(qa_data)} conversations (limited by num_samples) from Strova dataset for training/validation.")
    else:
        print(f"Prepared all {len(qa_data)} conversations from Strova dataset for training/validation.")
    return qa_data

def generate_fallback_synthetic_data(num_samples=100):
    greetings = [
        "Hello! Thank you for calling Customer Support. My name is Alex. How can I help you today?",
        "Hi there, thanks for reaching out. I'm Taylor. How can I assist you?"
    ]
    resolutions = [
        ("I checked your order #{} and it was shipped yesterday.", True, "resolved the order tracking issue"),
        ("I'm sorry, system is down. Call tomorrow.", False, "failed to resolve the issue")
    ]
    farewells = [
        "Is there anything else I can help you with today? Thank you!",
        "Bye."
    ]
    data = []
    for i in range(num_samples):
        order_id = random.randint(10000, 99999)
        greeting = random.choice(greetings)
        res_template, res_success, res_desc = random.choice(resolutions)
        resolution = res_template.format(order_id)
        farewell = random.choice(farewells)
        
        transcript_parts = [f"Agent: {greeting}", "Customer: I need help.", f"Agent: {resolution}"]
        if res_success:
            transcript_parts.append("Customer: Thank you!")
        transcript_parts.append(f"Agent: {farewell}")
        transcript = "\n".join(transcript_parts)
        
        if res_success and farewell == farewells[0]:
            label = 1
            rationale = f"The agent successfully resolved the issue ({res_desc}), greeted the customer, and closed with a polite farewell."
        else:
            label = 0
            rationale = f"The agent failed the QA audit because they either did not resolve the issue ({res_desc}) or closed abruptly."
            
        data.append({"transcript": transcript, "label": label, "rationale": rationale})
    return data

class AutoQADataset(Dataset):
    def __init__(self, raw_data, tokenizer):
        self.data = raw_data
        self.tokenizer = tokenizer
        self.samples = []
        self._preprocess()

    def _preprocess(self):
        for item in self.data:
            prompt = (
                "<|start_of_role|>user<|end_of_role|>"
                "Analyze the following customer service transcript and evaluate it.\n\n"
                f"Transcript:\n{item['transcript']}\n\n"
                "Evaluate if this transcript passes or fails QA standards and explain why."
                "<|start_of_role|>assistant<|end_of_role|>"
            )
            rationale = f"Rationale: {item['rationale']}"
            
            prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
            rationale_ids = self.tokenizer.encode(rationale, add_special_tokens=False)
            rationale_ids.append(self.tokenizer.eos_token_id)
            
            self.samples.append({
                "prompt_ids": prompt_ids,
                "rationale_ids": rationale_ids,
                "label": item["label"],
                "raw": item
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]

class DualHeadCollate:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, batch):
        input_ids_list = []
        labels_list = []
        clf_indices = []
        class_labels = []

        for item in batch:
            p_ids = item["prompt_ids"]
            r_ids = item["rationale_ids"]
            
            input_ids_list.append(p_ids + r_ids)
            labels_list.append([-100] * len(p_ids) + r_ids)
            clf_indices.append(len(p_ids) - 1)
            class_labels.append(item["label"])

        max_len = max(len(x) for x in input_ids_list)
        padded_input_ids = []
        padded_labels = []
        attention_masks = []

        for i_ids, labs in zip(input_ids_list, labels_list):
            pad_len = max_len - len(i_ids)
            padded_input_ids.append(i_ids + [self.pad_token_id] * pad_len)
            padded_labels.append(labs + [-100] * pad_len)
            attention_masks.append([1] * len(i_ids) + [0] * pad_len)

        return {
            "input_ids": torch.tensor(padded_input_ids, dtype=torch.long),
            "labels": torch.tensor(padded_labels, dtype=torch.long),
            "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
            "clf_indices": torch.tensor(clf_indices, dtype=torch.long),
            "class_labels": torch.tensor(class_labels, dtype=torch.long),
        }

# src/model.py
from transformers import AutoModelForImageClassification
from peft import LoraConfig, get_peft_model
from .config import Config

def get_model(use_lora=False):
    """
    Loads the model. If use_lora is True, wraps it in a PEFT LoRA config.
    """
    model = AutoModelForImageClassification.from_pretrained(
        Config.MODEL_CHECKPOINT,
        num_labels=10,
        ignore_mismatched_sizes=True
    )

    if use_lora:
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            inference_mode=False, 
            r=16,           
            lora_alpha=16, 
            lora_dropout=0.1,
            # CHANGED: ViT uses 'query' and 'value' for attention layers
            target_modules=["query", "value"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model
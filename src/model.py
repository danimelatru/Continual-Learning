from transformers import AutoModelForImageClassification
from peft import LoraConfig, get_peft_model
from omegaconf import DictConfig

def get_model(cfg: DictConfig, use_lora=False):
    """
    Loads the model. If use_lora is True, wraps it in a PEFT LoRA config.
    """
    model = AutoModelForImageClassification.from_pretrained(
        cfg.model.checkpoint,
        num_labels=cfg.model.num_labels,
        ignore_mismatched_sizes=cfg.model.ignore_mismatched_sizes
    )

    if use_lora:
        print("Applying LoRA configuration...")
        peft_config = LoraConfig(
            inference_mode=False, 
            r=cfg.lora.r,           
            lora_alpha=cfg.lora.alpha, 
            lora_dropout=cfg.lora.dropout,
            target_modules=cfg.lora.target_modules
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters()
    
    return model
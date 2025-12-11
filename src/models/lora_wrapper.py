from peft import LoraConfig, get_peft_model

def add_lora(model):
    config = LoraConfig(
        inference_mode=False,
        r=16,
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj"]
    )
    return get_peft_model(model, config)
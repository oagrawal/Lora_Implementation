import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .lora import LoRALinear

def get_model(model_name="TinyLlama/TinyLlama_v1.1"):
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map="auto",
    )
    model.config.use_cache = False
    model.config.pretraining_tp = 1
    return model

def get_tokenizer(model_name="TinyLlama/TinyLlama_v1.1"):
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return tokenizer

def apply_lora_to_model(model, rank, alpha):
    # Freeze all parameters
    for parameter in model.parameters():
        parameter.requires_grad = False

    # Apply LoRA
    for layer in model.model.layers:
        layer.mlp.up_proj = LoRALinear(layer.mlp.up_proj, rank=rank, alpha=alpha)
        layer.mlp.down_proj = LoRALinear(layer.mlp.down_proj, rank=rank, alpha=alpha)
        layer.mlp.gate_proj = LoRALinear(layer.mlp.gate_proj, rank=rank, alpha=alpha)
            
    return model

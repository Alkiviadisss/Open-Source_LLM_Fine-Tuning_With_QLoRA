
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


MODEL_NAME = "Qwen/Qwen3-4B"

def load_qlora_model():
    if not torch.cuda.is_available():
        raise RuntimeError("QLoRA requires a compatible CUDA GPU for this setup.")

    compute_dtype = torch.float16

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype)
    
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quantization_config, torch_dtype=compute_dtype, device_map="auto")
    model.config.use_cache = False
    
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none", task_type="CAUSAL_LM", target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])
    
    model = get_peft_model(model, lora_config)
    for name, param in model.named_parameters():
        if param.requires_grad:
            param.data = param.data.to(torch.float32)
    model.print_trainable_parameters()
    
    return model

if __name__ == "__main__":
    model = load_qlora_model()
    print("QLoRA model loaded successfully.")
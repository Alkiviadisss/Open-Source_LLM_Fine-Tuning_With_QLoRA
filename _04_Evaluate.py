
import gc
import json
import math
import re
from decimal import Decimal
from pathlib import Path
import pandas as pd
import torch
from datasets import load_from_disk
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from tqdm.auto import tqdm

MODEL_NAME = "Qwen/Qwen3-4B"
ADAPTER_DIR = Path("results/qlora_adapter")
TEST_DIR = Path("data/processed/test")
OUTPUT_DIR = Path("results/evaluation")
MAX_SAMPLES = None
MAX_NEW_TOKENS = 256
SEED = 42
if not torch.cuda.is_available():
    raise RuntimeError("Evaluation requires a compatible CUDA GPU.")

if not (ADAPTER_DIR / "adapter_config.json").exists():
    raise FileNotFoundError(f"Trained adapter not found in {ADAPTER_DIR}. " "Run _03_Train.py successfully first.")

if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test dataset not found: {TEST_DIR}. " "Run _01_Preprocess.py first.")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("GPU:", torch.cuda.get_device_name(0))


tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

test_dataset = load_from_disk(str(TEST_DIR))

if MAX_SAMPLES is not None:
    sample_count = min(MAX_SAMPLES, len(test_dataset))
    test_dataset = (test_dataset.shuffle(seed=SEED).select(range(sample_count)))

print("Test examples:", len(test_dataset))

def load_quantized_base():
    compute_dtype = (torch.bfloat16
        if torch.cuda.is_bf16_supported()
        else torch.float16)

    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype,)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, quantization_config=quantization_config, torch_dtype=compute_dtype, device_map="auto",)
    model.config.use_cache = True
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()
    return model

def get_target(example):
    completion = example["completion"]

    if isinstance(completion, list):
        for message in reversed(completion):
            if (isinstance(message, dict) and message.get("role") == "assistant"):
                return str(message.get("content", "")).strip()
    return str(completion).strip()

def generate_answer(model, prompt_messages):
    inputs = tokenizer.apply_chat_template(prompt_messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt", enable_thinking=False)
    input_device = model.get_input_embeddings().weight.device
    inputs = {key: value.to(input_device)
        for key, value in inputs.items()}

    prompt_length = inputs["input_ids"].shape[-1]

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True)

    generated_tokens = outputs[0, prompt_length:]
    prediction = tokenizer.decode(generated_tokens, skip_special_tokens=False,)
    return clean_prediction(prediction)

def clean_prediction(text):
    if "</think>" in text:
        text = text.rsplit("</think>", 1)[-1]
    elif "<think>" in text:
        return ""
    
    text = re.sub(r"<\|[^>]+\|>", "", text)
    text = text.strip()

    match = re.search(r"(?:final answer|answer)\s*(?:is|:|-)\s*(.+)", text, flags=re.IGNORECASE | re.DOTALL)

    if match:
        text = match.group(1).strip()
    else:
        lines = [line.strip()
            for line in text.splitlines()
            if line.strip()]
        if lines:
            text = lines[-1]
    return text.strip().strip("\"'` \t\r\n.;")

def normalize_answer(text):
    text = clean_prediction(str(text)).lower().strip()
    text = text.replace(",", "")
    text = re.sub(r"[$€£]", "", text)
    text = re.sub(r"\s+", "", text)
    text = text.rstrip(".;:")

    match = re.fullmatch(
        r"([-+]?\d+(?:\.\d+)?)(%)?", text)
    if match:
        number = Decimal(match.group(1)).normalize()
        text = format(number, "f") + (match.group(2) or "")
    return text

NUMBER_PATTERN = re.compile(
    r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)"
    r"(?:\.\d+)?(?:[eE][-+]?\d+)?\s*%?")


def extract_numeric_answer(text):
    text = clean_prediction(str(text))
    matches = list(NUMBER_PATTERN.finditer(text))

    if not matches:
        return None

    token = matches[-1].group().replace(",", "").strip()
    is_percentage = token.endswith("%")
    if is_percentage:
        token = token[:-1].strip()
    try:
        value = Decimal(token)
    except Exception:
        return None
    return value, is_percentage

def calculate_metrics(predictions, targets):
    exact_matches = [normalize_answer(prediction)
        == normalize_answer(target)
        for prediction, target in zip(predictions, targets)]

    numeric_correct = 0
    numeric_examples = 0
    for prediction, target in zip(predictions, targets):
        gold_number = extract_numeric_answer(target)
        if gold_number is None:
            continue
        numeric_examples += 1
        predicted_number = extract_numeric_answer(prediction)
        if predicted_number is None:
            continue
        pred_value, pred_percentage = predicted_number
        gold_value, gold_percentage = gold_number
        if pred_percentage != gold_percentage:
            continue
        if math.isclose(float(pred_value), float(gold_value), rel_tol=1e-3, abs_tol=1e-3,):
            numeric_correct += 1
    return {"num_examples": len(targets), "exact_match_correct": int(sum(exact_matches)), "normalized_exact_match_accuracy": (sum(exact_matches) / len(exact_matches) if exact_matches else None), "numeric_examples": numeric_examples, "numeric_correct": numeric_correct, "numeric_answer_accuracy": (numeric_correct / numeric_examples if numeric_examples else None)}

def evaluate_model(model, dataset, model_label):
    predictions = []
    targets = [get_target(example) for example in dataset]
    predictions_path = (OUTPUT_DIR / f"{model_label}_predictions.csv")
    for index, example in enumerate(
        tqdm(dataset, desc=f"Evaluating {model_label}")):
        prediction = generate_answer(model=model, prompt_messages=example["prompt"])
        predictions.append(prediction)
        if (index + 1) % 10 == 0:
            pd.DataFrame({"target": targets[:len(predictions)], "prediction": predictions}).to_csv(predictions_path, index=False)

    pd.DataFrame({"target": targets, "prediction": predictions}).to_csv(predictions_path, index=False)
    return predictions, targets

print("\nLoading base model...")
base_model = load_quantized_base()
base_predictions, targets = evaluate_model(base_model, test_dataset, "base")
base_metrics = calculate_metrics(base_predictions, targets)
print("\nBase model metrics:")
print(json.dumps(base_metrics, indent=2))

del base_model
gc.collect()
torch.cuda.empty_cache()

print("\nLoading trained QLoRA adapter...")
base_model = load_quantized_base()
qlora_model = PeftModel.from_pretrained(base_model, str(ADAPTER_DIR), is_trainable=False,)
qlora_model.config.use_cache = True
qlora_model.eval()
qlora_predictions, _ = evaluate_model(qlora_model, test_dataset, "qlora")
qlora_metrics = calculate_metrics(qlora_predictions, targets)
print("\nQLoRA model metrics:")
print(json.dumps(qlora_metrics, indent=2))

del qlora_model
del base_model
gc.collect()
torch.cuda.empty_cache()

comparison_df = pd.DataFrame({"target": targets, "base_prediction": base_predictions, "qlora_prediction": qlora_predictions})
comparison_df["base_exact_match"] = [normalize_answer(prediction) == normalize_answer(target)
    for prediction, target in zip(base_predictions, targets)]
comparison_df["qlora_exact_match"] = [normalize_answer(prediction) == normalize_answer(target)
    for prediction, target in zip(qlora_predictions, targets)]
comparison_df.to_csv(OUTPUT_DIR / "model_comparison.csv", index=False)
metrics = {"model_name": MODEL_NAME, "adapter_path": str(ADAPTER_DIR), "num_test_examples": len(test_dataset), "max_new_tokens": MAX_NEW_TOKENS, "base_model": base_metrics, "qlora_model": qlora_metrics, "exact_match_improvement": (qlora_metrics["normalized_exact_match_accuracy"] - base_metrics["normalized_exact_match_accuracy"]), "numeric_accuracy_improvement": (qlora_metrics["numeric_answer_accuracy"] - base_metrics["numeric_answer_accuracy"]
        if (qlora_metrics["numeric_answer_accuracy"] is not None and base_metrics["numeric_answer_accuracy"] is not None)
        else None)}

with open(OUTPUT_DIR / "metrics.json", "w", encoding="utf-8",) as file:
    json.dump(metrics, file, indent=2, ensure_ascii=False)

print("\nFINAL COMPARISON: ")
print("Base normalized exact match:", base_metrics["normalized_exact_match_accuracy"])
print("QLoRA normalized exact match:", qlora_metrics["normalized_exact_match_accuracy"])
print("Base numeric answer accuracy:", base_metrics["numeric_answer_accuracy"])
print("QLoRA numeric answer accuracy:", qlora_metrics["numeric_answer_accuracy"])
print("\nSaved results to:", OUTPUT_DIR)
import torch
from pathlib import Path
from datasets import load_from_disk
from transformers import AutoTokenizer
from trl import SFTTrainer, SFTConfig
from _02_Model import load_qlora_model

MODEL_NAME = "Qwen/Qwen3-4B"
DATA_DIR = Path("data/processed")
OUTPUT_DIR = Path("results/qlora_adapter")
SMOKE_TEST = False

if not torch.cuda.is_available():
    raise RuntimeError("No CUDA GPU detected.")

train_dataset = load_from_disk(str(DATA_DIR / "train"))
val_dataset = load_from_disk(str(DATA_DIR / "validation"))
print("Training examples:", len(train_dataset))
print("Validation examples:", len(val_dataset))

if SMOKE_TEST:
    train_dataset = train_dataset.select(range(min(32, len(train_dataset))))
    val_dataset = val_dataset.select(range(min(16, len(val_dataset))))

    max_steps = 3
    eval_steps = 3
    save_steps = 3
    max_length = 1024
    logging_steps = 1
else:
    max_steps = -1
    eval_steps = 200
    save_steps = 200
    max_length = 2048
    logging_steps = 10

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = load_qlora_model()
model.config.pad_token_id = tokenizer.pad_token_id

training_args = SFTConfig(
    output_dir=str(OUTPUT_DIR),
    num_train_epochs=1,
    max_steps=max_steps,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=8,
    gradient_checkpointing=True,
    learning_rate=1e-4,
    lr_scheduler_type="cosine",
    warmup_steps=15,
    max_grad_norm=0.3,
    bf16=False,
    fp16=True,
    max_length=max_length,
    completion_only_loss=True,
    eos_token=tokenizer.eos_token,
    eval_strategy="steps",
    eval_steps=eval_steps,
    save_strategy="steps",
    save_steps=save_steps,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=logging_steps,
    seed=42,
    report_to="none")

trainer = SFTTrainer(model=model, args=training_args, train_dataset=train_dataset, eval_dataset=val_dataset, processing_class=tokenizer)
train_result = trainer.train()
trainer.save_model(str(OUTPUT_DIR))
trainer.save_state()

trainer.log_metrics("train", train_result.metrics)
trainer.save_metrics("train", train_result.metrics)

eval_metrics = trainer.evaluate()
trainer.log_metrics("eval", eval_metrics)
trainer.save_metrics("eval", eval_metrics)

print("Training completed.")
print("Adapter saved to:", OUTPUT_DIR)
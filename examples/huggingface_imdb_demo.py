from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset
import torch
from gradlense import GradLense
from gradlense.integrations.huggingface import GradLenseTrainerCallback

# Load IMDb dataset
raw_datasets = load_dataset("imdb")
tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=128)

tokenized_datasets = raw_datasets.map(tokenize_function, batched=True)
tokenized_datasets.set_format("torch", columns=["input_ids", "attention_mask", "label"])

train_dataset = tokenized_datasets["train"].shuffle(seed=42).select(range(2000))
eval_dataset = tokenized_datasets["test"].select(range(500))

# Load model and GradLense
model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
gradlense = GradLense(model)

# Trainer
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=1,
    logging_steps=10,
    logging_dir="./logs",
    report_to=[],
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[GradLenseTrainerCallback(gradlense)]
)

trainer.train()
gradlense.plot_line()
gradlense.plot_heatmap(top_k=20)  # cleaner for large models
gradlense.summarize_alerts()


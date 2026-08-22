import argparse
import torch
import numpy as np
from datasets import load_dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from model import AITextDetector, AITextDetectorConfig


DATASET_NAME = "training_dataset"
MAX_LENGTH = 768


def tokenize(examples, tokenizer):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length",
    )


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    probabilities = 1 / (1 + np.exp(-logits))
    predictions = (probabilities > 0.5).astype(int)

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions),
        "precision": precision_score(labels, predictions),
        "recall": recall_score(labels, predictions),
        "roc_auc": roc_auc_score(labels, probabilities),
        "tpr": tpr,
        "fpr": fpr,
        "fnr": fnr,
    }


class DetectorTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels").float()
        logits = model(**inputs)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
        return (loss, logits) if return_outputs else loss


def main():
    parser = argparse.ArgumentParser(description="Train AI text detector")
    parser.add_argument("--base_model", default="microsoft/deberta-v3-large")
    parser.add_argument("--dataset", default=DATASET_NAME)
    parser.add_argument("--output_dir", default="checkpoints/ai-detector")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--learning_rate", type=float, default=2e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=MAX_LENGTH)
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.base_model)

    dataset = load_dataset(args.dataset)
    tokenized = dataset.map(
        lambda examples: tokenize(examples, tokenizer),
        batched=True,
        remove_columns=[c for c in dataset["train"].column_names if c != "label"],
    )
    tokenized.set_format("torch")

    config = AITextDetectorConfig(pretrained_model_name=args.base_model)
    model = AITextDetector(config)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_steps=100,
        bf16=torch.cuda.is_bf16_supported() if torch.cuda.is_available() else False,
        report_to="none",
    )

    trainer = DetectorTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        compute_metrics=compute_metrics,
    )

    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Model saved to {args.output_dir}")


if __name__ == "__main__":
    main()

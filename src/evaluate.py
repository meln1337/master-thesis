import argparse
import os
from abc import ABC, abstractmethod

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

from model import load_detector


MAX_LENGTH = 768
RESULTS_DIR = "results"


def get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


class BaseDetector(ABC):
    @abstractmethod
    def predict(self, texts):
        raise NotImplementedError


class OurDetector(BaseDetector):
    def __init__(self, checkpoint="weights/ai-text-detector"):
        self.model, self.tokenizer, self.device = load_detector(checkpoint)

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="Ours"):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True,
                max_length=MAX_LENGTH, padding=True,
            ).to(self.device)
            with torch.no_grad():
                logit = self.model(**inputs)
            scores.append(torch.sigmoid(logit).item())
        return scores


class GLTRDetector(BaseDetector):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.lm.eval()
        self.device = get_device()
        self.lm.to(self.device)

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="GLTR"):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024,
            ).to(self.device)
            with torch.no_grad():
                outputs = self.lm(**inputs)

            logits = outputs.logits[:, :-1, :]
            target_ids = inputs["input_ids"][:, 1:]
            sorted_indices = logits.argsort(dim=-1, descending=True)

            ranks = (sorted_indices == target_ids.unsqueeze(-1)).nonzero(as_tuple=True)[-1]
            ranks = ranks.float()

            top10 = (ranks < 10).float().mean().item()
            top100 = (ranks < 100).float().mean().item()
            top1000 = (ranks < 1000).float().mean().item()

            score = 1.0 - (0.5 * top10 + 0.3 * top100 + 0.2 * top1000)
            scores.append(score)
        return scores


class RADARDetector(BaseDetector):
    def __init__(self):
        from transformers import pipeline
        dev = get_device()
        self.paraphrase = pipeline(
            "text2text-generation",
            model="Vamsi/T5_Paraphrase_Paws",
            device=dev,
        )
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.lm.eval()
        self.device = dev
        self.lm.to(self.device)

    def log_likelihood(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.lm(**inputs)
        logits = outputs.logits[:, :-1, :]
        target_ids = inputs["input_ids"][:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.mean().item()

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="RADAR"):
            original_ll = self.log_likelihood(text)
            paraphrased = self.paraphrase(
                f"paraphrase: {text[:512]}", max_length=512, num_return_sequences=1,
            )[0]["generated_text"]
            paraphrased_ll = self.log_likelihood(paraphrased)
            score = 1 / (1 + np.exp(-(original_ll - paraphrased_ll)))
            scores.append(score)
        return scores


class FastDetectGPTDetector(BaseDetector):
    def __init__(self, num_perturbations=5):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.lm.eval()
        self.device = get_device()
        self.lm.to(self.device)
        self.num_perturbations = num_perturbations

    def token_log_probs(self, text):
        inputs = self.tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.device)
        with torch.no_grad():
            outputs = self.lm(**inputs)
        logits = outputs.logits[:, :-1, :]
        target_ids = inputs["input_ids"][:, 1:]
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        token_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
        return token_lp.squeeze(0), logits.squeeze(0), target_ids.squeeze(0)

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="Fast-DetectGPT"):
            token_lp, logits, target_ids = self.token_log_probs(text)
            original_mean = token_lp.mean().item()

            probs = torch.softmax(logits, dim=-1)
            perturbed_means = []
            for _ in range(self.num_perturbations):
                sampled = torch.multinomial(probs, num_samples=1).squeeze(-1)
                sampled_text = self.tokenizer.decode(sampled, skip_special_tokens=True)
                p_lp, _, _ = self.token_log_probs(sampled_text)
                perturbed_means.append(p_lp.mean().item())

            mu = np.mean(perturbed_means)
            sigma = max(np.std(perturbed_means), 1e-10)
            curvature = (original_mean - mu) / sigma
            score = 1 / (1 + np.exp(-curvature))
            scores.append(score)
        return scores


class LLMDetDetector(BaseDetector):
    def __init__(self):
        model_names = [
            "openai-community/gpt2",
            "facebook/opt-125m",
            "EleutherAI/pythia-160m",
        ]
        self.models = []
        self.device = get_device()
        for name in model_names:
            tok = AutoTokenizer.from_pretrained(name)
            lm = AutoModelForCausalLM.from_pretrained(name)
            lm.eval()
            lm.to(self.device)
            if tok.pad_token is None:
                tok.pad_token = tok.eos_token
            self.models.append((tok, lm))

    def perplexity(self, text, tokenizer, lm):
        inputs = tokenizer(
            text, return_tensors="pt", truncation=True, max_length=1024,
        ).to(self.device)
        with torch.no_grad():
            outputs = lm(**inputs)
        logits = outputs.logits[:, :-1, :]
        target_ids = inputs["input_ids"][:, 1:]
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.size(-1)), target_ids.reshape(-1),
        )
        return loss.exp().item()

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="LLMDet"):
            perplexities = []
            for tok, lm in self.models:
                ppl = self.perplexity(text, tok, lm)
                perplexities.append(ppl)

            mean_ppl = np.mean(perplexities)
            std_ppl = np.std(perplexities)
            proxy = 1 / (1 + mean_ppl / 100)
            variance_signal = 1 / (1 + std_ppl / 50)
            score = 0.7 * proxy + 0.3 * variance_signal
            scores.append(score)
        return scores


SHUFFLE_SEED = 42


def load_raid_dataset(split="extra", limit=500):
    ds = load_dataset("liamdugan/raid", split=split).shuffle(seed=SHUFFLE_SEED)
    texts, labels, sources = [], [], []
    for row in ds.select(range(min(limit, len(ds)))):
        texts.append(row["generation"])
        labels.append(0 if row["model"] == "human" else 1)
        sources.append(row["model"])
    return texts, labels, sources


def load_hc3_dataset(limit=500):
    ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train").shuffle(seed=SHUFFLE_SEED)
    texts, labels, sources = [], [], []
    count = 0
    for row in ds:
        if count >= limit:
            break
        for answer in row.get("human_answers", []):
            if count >= limit:
                break
            texts.append(answer)
            labels.append(0)
            sources.append("human")
            count += 1
        for answer in row.get("chatgpt_answers", []):
            if count >= limit:
                break
            texts.append(answer)
            labels.append(1)
            sources.append("chatgpt")
            count += 1
    return texts, labels, sources


def load_deepfake_dataset(limit=500):
    ds = load_dataset("yaful/DeepfakeTextDetect", split="test").shuffle(seed=SHUFFLE_SEED)
    texts, labels, sources = [], [], []
    for row in ds.select(range(min(limit, len(ds)))):
        texts.append(row["text"])
        labels.append(row["label"])
        sources.append(row.get("source", "unknown"))
    return texts, labels, sources


def compute_all_metrics(labels, scores, threshold=0.5):
    predictions = (np.array(scores) > threshold).astype(int)
    labels = np.array(labels)

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "roc_auc": roc_auc_score(labels, scores) if len(set(labels)) > 1 else 0.0,
        "tpr": tp / (tp + fn) if (tp + fn) > 0 else 0.0,
        "fpr": fp / (fp + tn) if (fp + tn) > 0 else 0.0,
        "fnr": fn / (fn + tp) if (fn + tp) > 0 else 0.0,
    }


def run_evaluation(detector, detector_name, texts, labels, sources, output_dir):
    print(f"\nRunning {detector_name}...")
    scores = detector.predict(texts)
    metrics = compute_all_metrics(labels, scores)

    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"  TPR:      {metrics['tpr']:.4f}")
    print(f"  FPR:      {metrics['fpr']:.4f}")
    print(f"  FNR:      {metrics['fnr']:.4f}")

    return {"metrics": metrics, "scores": scores, "labels": labels, "sources": sources, "texts": texts}


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI text detectors")
    parser.add_argument("--dataset", default="raid", choices=["raid", "hc3", "deepfake"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    parser.add_argument("--detectors", nargs="+",
                        default=["ours", "gltr", "radar", "fast_detect_gpt", "llmdet"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading dataset: {args.dataset}")
    if args.dataset == "raid":
        texts, labels, sources = load_raid_dataset(limit=args.limit)
    elif args.dataset == "hc3":
        texts, labels, sources = load_hc3_dataset(limit=args.limit)
    elif args.dataset == "deepfake":
        texts, labels, sources = load_deepfake_dataset(limit=args.limit)

    print(f"Loaded {len(texts)} samples")

    detector_map = {
        "ours": ("Ours", OurDetector),
        "gltr": ("GLTR", GLTRDetector),
        "radar": ("RADAR", RADARDetector),
        "fast_detect_gpt": ("Fast-DetectGPT", FastDetectGPTDetector),
        "llmdet": ("LLMDet", LLMDetDetector),
    }

    all_results = {}
    metrics_rows = []

    for key in args.detectors:
        if key not in detector_map:
            print(f"Unknown detector: {key}, skipping")
            continue
        name, cls = detector_map[key]
        detector = cls()
        result = run_evaluation(detector, name, texts, labels, sources, args.output_dir)
        all_results[name] = result
        row = {"detector": name, **result["metrics"]}
        metrics_rows.append(row)

    results_df = pd.DataFrame(metrics_rows)
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    print(results_df.to_string(index=False))
    results_df.to_csv(os.path.join(args.output_dir, "results.csv"), index=False)

    predictions_path = os.path.join(args.output_dir, "predictions.csv")
    pred_rows = []
    for detector_name, result in all_results.items():
        for text, label, source, score in zip(
            result["texts"], result["labels"], result["sources"], result["scores"]
        ):
            pred_rows.append({
                "detector": detector_name,
                "text": text[:500],
                "label": label,
                "source": source,
                "score": score,
            })
    pd.DataFrame(pred_rows).to_csv(predictions_path, index=False)

    print(f"\nResults saved to {args.output_dir}/")


if __name__ == "__main__":
    main()

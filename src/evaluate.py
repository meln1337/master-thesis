import argparse
import os
from abc import ABC, abstractmethod

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoModelForSequenceClassification, AutoModel, AutoConfig,
)
from safetensors.torch import load_file as load_safetensors
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve,
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

            score = 0.5 * top10 + 0.3 * top100 + 0.2 * top1000
            scores.append(score)
        return scores


class RADARDetector(BaseDetector):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
        self.model = AutoModelForSequenceClassification.from_pretrained("TrustSafeAI/RADAR-Vicuna-7B")
        self.model.eval()
        self.device = get_device()
        self.model.to(self.device)

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="RADAR"):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            ).to(self.device)
            with torch.no_grad():
                logits = self.model(**inputs).logits
            score = torch.log_softmax(logits, dim=-1)[:, 0].exp().item()
            scores.append(score)
        return scores


class FastDetectGPTDetector(BaseDetector):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
        self.lm = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
        self.lm.eval()
        self.device = get_device()
        self.lm.to(self.device)

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="Fast-DetectGPT"):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=1024,
            ).to(self.device)
            with torch.no_grad():
                logits = self.lm(**inputs).logits[:, :-1, :]
            target_ids = inputs["input_ids"][:, 1:]

            log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
            probs = log_probs.exp()

            true_token_lp = log_probs.gather(2, target_ids.unsqueeze(-1)).squeeze(-1)
            analytic_mean = (probs * log_probs).sum(dim=-1)
            analytic_second_moment = (probs * log_probs.pow(2)).sum(dim=-1)
            analytic_var = (analytic_second_moment - analytic_mean.pow(2)).clamp(min=1e-10)

            total_logprob = true_token_lp.sum()
            total_mean = analytic_mean.sum()
            total_std = analytic_var.sum().sqrt()

            curvature = ((total_logprob - total_mean) / total_std).item()
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


# Checkpoints for the internal mAIDetector v2/v2.1/v3 lineage. Proprietary weights, not part of
# this repo -- kept in a sibling checkout and referenced by absolute path.
MAI_MODEL_DIRS = {
    "v2": "/Users/borys.melnychuk/Documents/repos/justdone-ai-detector/models/mAIDetector_v2",
    "v2.1": "/Users/borys.melnychuk/Documents/repos/justdone-ai-detector/models/mAIDetector_v2.1",
    "v3": "/Users/borys.melnychuk/Documents/repos/justdone-ai-detector/models/mAIDetector_v3",
}


class MAIDetector(BaseDetector):
    """Raw mAIDetector v2/v2.1/v3 checkpoint: v3 stores a bare encoder + mean-pooled linear
    head (state-dict keys prefixed "encoder."), v2/v2.1 are stock
    AutoModelForSequenceClassification checkpoints with a CLS-token head baked in."""

    def __init__(self, model_dir):
        self.device = get_device()
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        state_dict = load_safetensors(os.path.join(model_dir, "model.safetensors"))
        self.is_v3 = any(k.startswith("encoder.") for k in state_dict)

        if self.is_v3:
            config = AutoConfig.from_pretrained(model_dir)
            encoder_sd = {k[len("encoder."):]: v for k, v in state_dict.items() if k.startswith("encoder.")}
            self.encoder = AutoModel.from_config(config, attn_implementation="eager")
            self.encoder.load_state_dict(encoder_sd, strict=True)
            self.encoder.to(self.device).eval()

            self.head = torch.nn.Linear(config.hidden_size, 2)
            self.head.weight.data.copy_(state_dict["classifier.weight"])
            self.head.bias.data.copy_(state_dict["classifier.bias"])
            self.head.to(self.device)
        else:
            self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
            self.model.to(self.device).eval()

    def predict(self, texts):
        scores = []
        for text in tqdm(texts, desc="mAIDetector"):
            inputs = self.tokenizer(
                text, return_tensors="pt", truncation=True, max_length=512,
            ).to(self.device)
            with torch.no_grad():
                if self.is_v3:
                    hidden = self.encoder(**inputs).last_hidden_state
                    mask = inputs["attention_mask"].unsqueeze(-1).float()
                    pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)
                    logits = self.head(pooled)
                else:
                    logits = self.model(**inputs).logits
            score = torch.softmax(logits.float(), dim=-1)[:, 1].item()
            scores.append(score)
        return scores


SHUFFLE_SEED = 42


def load_raid_dataset(split="train", limit=500):
    ds = load_dataset("liamdugan/raid", split=split, streaming=True)
    ds = ds.shuffle(seed=SHUFFLE_SEED, buffer_size=10_000)

    human_target = limit // 2
    ai_target = limit - human_target
    texts, labels, sources = [], [], []
    human_count, ai_count = 0, 0

    for row in ds:
        if human_count >= human_target and ai_count >= ai_target:
            break
        is_human = row["model"] == "human"
        if is_human and human_count < human_target:
            texts.append(row["generation"])
            labels.append(0)
            sources.append(row["model"])
            human_count += 1
        elif not is_human and ai_count < ai_target:
            texts.append(row["generation"])
            labels.append(1)
            sources.append(row["model"])
            ai_count += 1

    return texts, labels, sources


def load_hc3_dataset(limit=500):
    hc3_parquet = "https://huggingface.co/datasets/Hello-SimpleAI/HC3/resolve/refs%2Fconvert%2Fparquet/all/train/0000.parquet"
    df = pd.read_parquet(hc3_parquet)
    df = df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)
    texts, labels, sources = [], [], []
    count = 0
    for _, row in df.iterrows():
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
    deepfake_parquet = "https://huggingface.co/datasets/yaful/DeepfakeTextDetect/resolve/refs%2Fconvert%2Fparquet/default/test/0000.parquet"
    df = pd.read_parquet(deepfake_parquet)
    df = df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    human_target = limit // 2
    ai_target = limit - human_target
    texts, labels, sources = [], [], []
    human_count, ai_count = 0, 0

    for _, row in df.iterrows():
        if human_count >= human_target and ai_count >= ai_target:
            break
        is_human = "human" in row["src"]
        if is_human and human_count < human_target:
            texts.append(row["text"])
            labels.append(0)
            sources.append(row["src"])
            human_count += 1
        elif not is_human and ai_count < ai_target:
            texts.append(row["text"])
            labels.append(1)
            sources.append(row["src"])
            ai_count += 1

    return texts, labels, sources


def load_m4gt_dataset(limit=500):
    """M4GT-Bench (justdone-org's private HF mirror: text/label/source/model, 6 generators + human)."""
    from huggingface_hub import hf_hub_download
    m4gt_parquet = hf_hub_download(
        repo_id="justdone-org/m4gt", filename="data/train-00000-of-00001.parquet", repo_type="dataset",
    )
    df = pd.read_parquet(m4gt_parquet, columns=["text", "label", "model"])
    df = df.sample(frac=1, random_state=SHUFFLE_SEED).reset_index(drop=True)

    human_target = limit // 2
    ai_target = limit - human_target
    texts, labels, sources = [], [], []
    human_count, ai_count = 0, 0

    for _, row in df.iterrows():
        if human_count >= human_target and ai_count >= ai_target:
            break
        is_human = row["label"] == 0
        if is_human and human_count < human_target:
            texts.append(row["text"])
            labels.append(0)
            sources.append(row["model"])
            human_count += 1
        elif not is_human and ai_count < ai_target:
            texts.append(row["text"])
            labels.append(1)
            sources.append(row["model"])
            ai_count += 1

    return texts, labels, sources


def best_threshold(labels, scores):
    fpr_vals, tpr_vals, thresholds = roc_curve(labels, scores)
    best_idx = np.argmax(tpr_vals - fpr_vals)
    return thresholds[best_idx]


def compute_all_metrics(labels, scores, threshold=None):
    labels = np.array(labels)
    scores = np.array(scores)

    if threshold is None:
        threshold = best_threshold(labels, scores) if len(set(labels)) > 1 else 0.5

    predictions = (scores > threshold).astype(int)

    tp = ((predictions == 1) & (labels == 1)).sum()
    fp = ((predictions == 1) & (labels == 0)).sum()
    fn = ((predictions == 0) & (labels == 1)).sum()
    tn = ((predictions == 0) & (labels == 0)).sum()

    return {
        "threshold": threshold,
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

    print(f"  Threshold: {metrics['threshold']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  F1:       {metrics['f1']:.4f}")
    print(f"  ROC AUC:  {metrics['roc_auc']:.4f}")
    print(f"  TPR:      {metrics['tpr']:.4f}")
    print(f"  FPR:      {metrics['fpr']:.4f}")
    print(f"  FNR:      {metrics['fnr']:.4f}")

    return {"metrics": metrics, "scores": scores, "labels": labels, "sources": sources, "texts": texts}


DATASET_LOADERS = {
    "raid": load_raid_dataset,
    "hc3": load_hc3_dataset,
    "deepfake": load_deepfake_dataset,
    "m4gt": load_m4gt_dataset,
}


def evaluate_dataset(dataset_name, limit, detectors, output_dir):
    print(f"\nLoading dataset: {dataset_name}")
    texts, labels, sources = DATASET_LOADERS[dataset_name](limit=limit)
    print(f"Loaded {len(texts)} samples")

    dataset_dir = os.path.join(output_dir, dataset_name)
    os.makedirs(dataset_dir, exist_ok=True)

    all_results = {}
    metrics_rows = []

    for name, detector in detectors.items():
        result = run_evaluation(detector, name, texts, labels, sources, dataset_dir)
        all_results[name] = result
        row = {"detector": name, **result["metrics"]}
        metrics_rows.append(row)

    results_df = pd.DataFrame(metrics_rows)
    print("\n" + "=" * 60)
    print(f"RESULTS SUMMARY — {dataset_name}")
    print("=" * 60)
    print(results_df.to_string(index=False))

    results_path = os.path.join(dataset_dir, "results.csv")
    if os.path.exists(results_path):
        prior = pd.read_csv(results_path)
        prior = prior[~prior["detector"].isin(results_df["detector"])]
        results_df = pd.concat([prior, results_df], ignore_index=True)
    results_df.to_csv(results_path, index=False)

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
    pred_df = pd.DataFrame(pred_rows)
    preds_path = os.path.join(dataset_dir, "predictions.csv")
    if os.path.exists(preds_path):
        prior_preds = pd.read_csv(preds_path)
        prior_preds = prior_preds[~prior_preds["detector"].isin(pred_df["detector"])]
        pred_df = pd.concat([prior_preds, pred_df], ignore_index=True)
    pred_df.to_csv(preds_path, index=False)

    print(f"Results saved to {dataset_dir}/")


def main():
    parser = argparse.ArgumentParser(description="Evaluate AI text detectors")
    parser.add_argument("--dataset", default="raid", choices=["raid", "hc3", "deepfake", "m4gt", "all"])
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--output_dir", default=RESULTS_DIR)
    parser.add_argument("--detectors", nargs="+",
                        default=["ours", "gltr", "radar", "fast_detect_gpt", "llmdet"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    detector_map = {
        "ours": ("Ours", OurDetector),
        "gltr": ("GLTR", GLTRDetector),
        "radar": ("RADAR", RADARDetector),
        "fast_detect_gpt": ("Fast-DetectGPT", FastDetectGPTDetector),
        "llmdet": ("LLMDet", LLMDetDetector),
        "mai_v2": ("mAIDetector-v2", lambda: MAIDetector(MAI_MODEL_DIRS["v2"])),
        "mai_v2.1": ("mAIDetector-v2.1", lambda: MAIDetector(MAI_MODEL_DIRS["v2.1"])),
        "mai_v3": ("mAIDetector-v3", lambda: MAIDetector(MAI_MODEL_DIRS["v3"])),
    }

    detectors = {}
    for key in args.detectors:
        if key not in detector_map:
            print(f"Unknown detector: {key}, skipping")
            continue
        name, cls = detector_map[key]
        detectors[name] = cls()

    dataset_names = list(DATASET_LOADERS.keys()) if args.dataset == "all" else [args.dataset]
    for dataset_name in dataset_names:
        evaluate_dataset(dataset_name, args.limit, detectors, args.output_dir)


if __name__ == "__main__":
    main()

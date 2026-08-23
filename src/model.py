import torch
from torch import nn
from transformers import PreTrainedModel, AutoConfig, AutoModel, AutoTokenizer


class AITextDetector(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        super().__init__(config)
        self.model = AutoModel.from_config(config)
        self.classifier = nn.Linear(config.hidden_size, 1)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, **kwargs):
        hidden_states = self.model(input_ids, attention_mask=attention_mask)[0]

        mask = attention_mask.unsqueeze(-1).expand(hidden_states.size()).float()
        summed = (hidden_states * mask).sum(dim=1)
        counted = mask.sum(dim=1).clamp(min=1e-9)
        pooled = summed / counted

        logits = self.classifier(pooled).squeeze(-1)
        return logits


WEIGHTS_DIR = "weights/ai-text-detector"


def load_detector(checkpoint=WEIGHTS_DIR, device=None):
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    tokenizer = AutoTokenizer.from_pretrained(checkpoint)
    model = AITextDetector.from_pretrained(checkpoint, trust_remote_code=True)
    model.to(device)
    model.eval()
    return model, tokenizer, device

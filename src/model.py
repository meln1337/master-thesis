import torch
from torch import nn
from transformers import PreTrainedModel, PretrainedConfig, AutoTokenizer
from transformers import DebertaV2Model


class AITextDetectorConfig(PretrainedConfig):
    model_type = "deberta-v2"

    def __init__(self, pretrained_model_name="microsoft/deberta-v3-large", **kwargs):
        super().__init__(**kwargs)
        self.pretrained_model_name = pretrained_model_name


class AITextDetector(PreTrainedModel):
    config_class = AITextDetectorConfig

    def __init__(self, config):
        super().__init__(config)
        self.backbone = DebertaV2Model.from_pretrained(config.pretrained_model_name)
        self.classifier = nn.Linear(self.backbone.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        hidden_states = outputs.last_hidden_state

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).float()
            pooled = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1e-9)
        else:
            pooled = hidden_states.mean(dim=1)

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

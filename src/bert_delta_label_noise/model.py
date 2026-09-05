import torch
import torch.nn as nn

from .config import BertDeltaLabelNoiseConfig


def build_model(cfg: BertDeltaLabelNoiseConfig):
    from transformers import AutoModelForSequenceClassification

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg.model_name,
        num_labels=cfg.num_labels,
    )
    if not hasattr(model, "bert"):
        raise TypeError(f"Expected a BERT model, got {type(model).__name__}")
    for parameter in model.parameters():
        parameter.requires_grad = False
    layers = model.bert.encoder.layer
    if not 1 <= cfg.trainable_top_layers <= len(layers):
        raise ValueError("trainable_top_layers must select at least one BERT layer")
    for layer in layers[-cfg.trainable_top_layers :]:
        for parameter in layer.parameters():
            parameter.requires_grad = True
    if model.bert.pooler is not None:
        for parameter in model.bert.pooler.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return model


def regularized_displacement_parameters(
    model,
    include_classifier: bool = False,
) -> tuple[list[str], list[nn.Parameter]]:
    names: list[str] = []
    parameters: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad or parameter.ndim != 2:
            continue
        if name.startswith("classifier.") and not include_classifier:
            continue
        names.append(name)
        parameters.append(parameter)
    return names, parameters


def trainable_state_dict(model) -> dict[str, torch.Tensor]:
    trainable = {
        name for name, parameter in model.named_parameters() if parameter.requires_grad
    }
    return {
        name: value.detach().cpu().clone()
        for name, value in model.state_dict().items()
        if name in trainable
    }

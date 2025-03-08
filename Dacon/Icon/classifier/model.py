import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

def get_model(num_channels, num_labels):
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    config.num_channels = num_channels
    config.num_labels = num_labels

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        ignore_mismatched_sizes=True
        config=config,
    )
    model.to(device)
    return model

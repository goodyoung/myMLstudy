import torch.nn as nn
from transformers import ViTForImageClassification, ViTConfig

def get_model(num_classes):
    config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    config.num_channels = 1  # Grayscale 이미지 적용

    model = ViTForImageClassification.from_pretrained(
        "google/vit-base-patch16-224-in21k",
        config=config,
        num_labels=num_classes,
        ignore_mismatched_sizes=True
    )

    # 1채널 입력을 받을 수 있도록 수정
    model.vit.embeddings.patch_embeddings.projection = nn.Conv2d(1, 768, kernel_size=(16, 16), stride=(16, 16))

    return model

import timm
import torch.nn as nn
# from transformers import ViTForImageClassification, ViTConfig

def get_model(num_channels, num_labels, device):
    # for ViT
    # config = ViTConfig.from_pretrained("google/vit-base-patch16-224-in21k")
    # config.num_channels = num_channels
    # config.num_labels = num_labels
    # model = ViTForImageClassification.from_pretrained(
    #     "google/vit-base-patch16-224-in21k",
    #     ignore_mismatched_sizes=True,
    #     config=config,
    # )
    model = timm.create_model("beitv2_base_patch16_224.in1k_ft_in22k_in1k",
                              num_classes=num_labels, in_chans=num_channels,
                                pretrained=False)
    model.to(device)
    return model

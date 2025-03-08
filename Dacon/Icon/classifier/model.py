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
    model = timm.create_model("eva_large_patch14_196.in22k_ft_in22k_in1k", pretrained=True)
    model.patch_embed.proj = nn.Conv2d(num_channels, 1024, kernel_size=(14, 14), stride=(14, 14))
    model.head = nn.Linear(model.head.in_features, num_labels)
    
    model.to(device)
    return model

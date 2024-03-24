import torch
import torchvision
from torch import nn
def create_model():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transform = weights.transforms()
    model = torchvision.models.vit_b_16(weights=weights)

        # 4. Freeze all layers in base model
    for param in model.parameters():
        param.requires_grad = False

    model.heads = nn.Sequential(
         nn.Linear(768,1)
    )
    return model, transform

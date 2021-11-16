import timm
import torch.nn as nn

# Timm R.Goss ViT Imagenet pretrained model
class Encoder(nn.Module):
    def __init__(self, 
                model_size, 
                patch_size:int, 
                input_size:int, 
                pretrained=True):
        assert model_size in ["small", "base", "large"]
        assert input_size == 384 or input_size == 224
        assert patch_size <= input_size and input_size%patch_size==0

        self.pretrained = pretrained
        self.vit_type = f"vit_{model_size}_patch{patch_size}_{input_size}"

        model = timm.create_model(self.vit_type, pretrained=pretrained)
        # Delect last 2 blocks
        self.model = nn.Sequential(*list(model.children())[:-2])

    def forward(self, x):
        return self.model(x)
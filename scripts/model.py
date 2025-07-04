import torch
import torchvision.models as models
import torch.nn as nn

def get_embedding_model():
    # Load ResNet18 pretrained on ImageNet
    # We're using ResNet-18, a compact yet powerful CNN model
    resnet = models.resnet18(pretrained=True)

    # Remove the classification head (fc layer)
    # We remove the final classifier layer, because we just want embeddings, not class predictions
    modules = list(resnet.children())[:-1]
    # The resulting model outputs a 512-dimensional vector for each image
    model = nn.Sequential(*modules)

    # Set to eval mode (disable dropout/batchnorm training behavior)
    model.eval()

    return model

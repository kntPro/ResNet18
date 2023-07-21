import torch
from torchvision.models import resnet18

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
model = resnet18()
print(type(model))
print(dir(model))
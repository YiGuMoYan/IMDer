import torch

from data_loader import MMDataset

print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.backends.cudnn.is_available())
print(torch.backends.cudnn.version())

data_loader = MMDataset()
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torch
embedding = nn.Embedding()
weight = torch.FloatTensor([[1, 2.3, 3], [4, 5.1, 6.3]],[[1, 2.3, 3], [4, 5.1, 6.3]])
embedding = nn.Embedding.from_pretrained(weight)
print(embedding)
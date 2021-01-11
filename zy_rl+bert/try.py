import torch
import torch.optim as optim

x = torch.tensor([1.], requires_grad=True)
w = torch.tensor([2.], requires_grad=True)

y = x * w + 1.
z = x ** 2
optimizer = optim.Adam([x, w], lr=0.1)

y.backward()

z.backward()
x.grad *= -3.

optimizer.step()
print(x)
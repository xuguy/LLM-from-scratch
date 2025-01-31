import torch.nn.functional as F
from torch.autograd import grad
import torch

# to test how batch data's gradients is calculated
# the answer is to cal each then average
y = torch.tensor([1.0])
x1 = torch.tensor([1.1])
# y = torch.tensor([0.0])
# x1 = torch.tensor([2.2])
w1 = torch.tensor([2.2], requires_grad= True)
b = torch.tensor([0.0], requires_grad= True)

z = x1*w1+b
a = torch.sigmoid(z)
loss = F.binary_cross_entropy(a, y)

grad_L_w1 = grad(loss, w1, retain_graph=True)
grad_L_b = grad(loss, b, retain_graph=True)

print(grad_L_w1)
print(grad_L_b)

loss.backward()
print(w1.grad)
print(b.grad)

input_tensor = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
mat1 = torch.tensor([[1.0, 0.0], [0.0, 1.0]], requires_grad=True)
mat2 = torch.tensor([[5.0, 6.0], [7.0, 8.0]], requires_grad=True)
tmp_tensor = torch.tensor([[1.,0.],[0.,1.]])
output = torch.addmm(input_tensor, mat1, mat2)
output.backward(tmp_tensor)
mat2.grad



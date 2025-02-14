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

# ========= Deep-Learning with pytorch ch01
# torchvision: torch for computer vision
from torchvision import models
from torchvision import transforms

# check models list
len(dir(models)) # total 221 models

# load alex net
resnet = models.resnet101(pretrained = True)
resnet

preprocess = transforms.Compose([transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
'''
preprocess image:
1. resize to 256x256
2. crop image to 224x224 around the center
3. transform it to a tensor
4. 
'''

from PIL import Image
import torch
# instead of using git clone, you can add 'ss' before the 'github.com' format file url to download
img = Image.open('../dlwpt-code/data/p1ch2/bobby.jpg')
img.show()

img_t = preprocess(img) # img_t.shape = torch.Size([3, 224, 224])
batch_t = torch.unsqueeze(img_t, 0)

resnet.eval()
out = resnet(batch_t)

with open('../dlwpt-code/data/p1ch2/imagenet_classes.txt') as f:
    labels = [line.strip() for line in f.readlines()]

# find the largest element and its index
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim = 1)[0]*100
labels[index[0]], percentage[index[0]].item()

# see the rank of top 5


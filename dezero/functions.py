import numpy as np
from dezero.core import Function

class Sin(Function):

    def forward(self, x):
        y = np.sin(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy*cos(x)
        return gx

def sin(x):
    return Sin()(x)

class Cos(Function):
    # 为什么这里直接用ndarray实例而不转换成Variable呢？因为没有必要，Variable类实现的add mul neg等基础运算都可以兼容ndarray类型的数据，况且正向传播不需要Variable类。
    def forward(self, x):
        y = np.cos(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy*(-sin(x))
        return gx

def cos(x):
    return Cos()(x)

class Tanh(Function):
    def forward(self, x):
        y = np.tanh(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy*(1-y*y)
        return gx
    
def tanh(x):
    return Tanh()(x)

class Reshape(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape # 先保存输入x原本的shape：self.x_shape
        y = x.reshape(self.shape) # 再把输入x转换为目标shape：self.shape，也就是初始化实例时接受的shape参数
        return y
    
    def backward(self, gy):
        return reshape(gy, self.x_shape) # 反向传播时，把gy转换成输入x本来的形状x_shape

from dezero.core import as_variable

def reshape(x, shape):
    if x.shape == shape: # 如果输入x的shape和目标shape一致，那么把x转换为Variable类后直接返回
                        # 建议回看一下as_variable的定义：如果x时Variable，那么直接返回；如果x是ndarray，那么返回Variable(x)，Variable类只接受ndarray作为输入，如果不是ndarray将会报错。
        return as_variable(x)
    return Reshape(shape)(x)

# the more genral one: support multi axes transpose
class Transpose(Function):
    def __init__(self, axes=None):
        self.axes = axes

    def forward(self, x):
        # transpose 不需要接受额外的参数，只需要待tran 参数
        # core.Variable中已经定义过transpose方法，因此可以直接用x.引用
        y = x.transpose(self.axes)
        return y
    
    # def backward(self, gy):
    #     gx = transpose(gy)
    #     return gx
    def backward(self, gy):
        if self.axes is None:
            return transpose(gy)
        
        axes_len = len(self.axes)
        inv_axes = tuple(np.argsort([ax % axes_len for ax in self.axes]))
        return transpose(gy, inv_axes)

# old transpose for 2 axes only    
# def transpose(x):
#     return Transpose()(x)
def transpose(x, axes=None):
    return Transpose(axes)(x)



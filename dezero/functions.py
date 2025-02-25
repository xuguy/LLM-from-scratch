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


class BroadcastTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        self.x_shape = x.shape
        y = np.broadcast_to(x, self.shape)
        return y

    def backward(self, gy):
        # 关于为什么反向传播是sum_to：broadcast本质是复制向量，因此复制后的向量的当梯度回传给被复制的向量x时，会多次传播到x
        # 多次传播到同一个x的结果就是梯度相加
        gx = sum_to(gy, self.x_shape)
        return gx

def broadcast_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return BroadcastTo(shape)(x)

from dezero import utils

class SumTo(Function):
    def __init__(self, shape):
        self.shape = shape

    def forward(self, x):
        # 此时 x从 __call__中传来，已经是ndarray
        self.x_shape = x.shape
        # 对输入x进行求和，输出和self.shape一样shape的tensor
        # 这里导入了专门处理ndarray数据的utils.sum_to()
        y = utils.sum_to(x, self.shape)
        return y
    def backward(self, gy):
        # 把gy复制成(broadcast)self.x_shape:
        # Sum (缩小，也即求和->减少维数) 的反向是扩（复制，也即broadcast）
        gx = broadcast_to(gy, self.x_shape)
        return gx
    
def sum_to(x, shape):
    if x.shape == shape:
        return as_variable(x)
    return SumTo(shape)(x)

# 旧的简易版sum，只支持简单sum（全部sum成标量）
# class Sum(Function):
#     def forward(self, x):
#         self.x_shape = x.shape
#         y = x.sum()
#         return y
    
#     def backward(self, gy):
#         gx = broadcast_to(gy, self.x_shape)
#         return gx
    
# def sum(x):
#         return Sum()(x)
    

class Sum(Function):
    def __init__(self, axis, keepdims):
        self.axis = axis
        self.keepdims = keepdims
    def forward(self, x):
        self.x_shape = x.shape
        y = x.sum(axis=self.axis, keepdims=self.keepdims)
        return y
    
    def backward(self, gy):
        gy = utils.reshape_sum_backward(gy,self.x_shape, self.axis, self.keepdims) # 调整gy的shape以适应不同的sum法
        gx = broadcast_to(gy, self.x_shape)
        return gx

def sum(x, axis = None, keepdims=False):
    return Sum(axis, keepdims)(x)


# MatMul反向传播的实现
class MatMul(Function):
    def forward(self, x, W):
        # x和W可能是框架中的张量对象（而非普通NumPy数组），其.dot()方法将要被重写。直接调用x.dot(W)会触发自定义的矩阵乘法操作（即MatMul类），从而在正向传播时记录计算图，为反向传播的梯度计算提供支持。若使用np.dot(x, W)，则会绕过框架的自动微分机制，导致梯度无法正确计算。
        y = x.dot(W)
        return y
    
    def backward(self, gy):
        x, W = self.inputs
        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)
        return gx, gW
    
def matmul(x, W):
    return MatMul()(x, W)

# old, low ram efficiency version
def mean_squared_error_simple(x0, x1):
    diff = x0 - x1 # both x0 and x1 are Variable dtype
    return sum(diff**2) / len(diff)

# new, high efficiency version
class MeanSquaredError(Function):
    def forward(self, x0, x1):
        diff = x0-x1
        # 这里用的是ndarray的方法
        y = (diff**2).sum()/len(diff)
        return y
    
    def backward(self, gy):
        # 反向传播的实现就是通过式子求导后将其编写成代码
        x0, x1 = self.inputs
        diff = x0-x1
        gx0 = gy*diff*(2./len(diff)) # x0的grad
        gx1 = -gx0 # x1的grad
        return gx0, gx1
    
def mean_squared_error(x0, x1):
    return MeanSquaredError()(x0, x1)

# 为什么新的MeanSquaredError的内存效率更高？因为新的MSE的计算图更简洁，因此有更少的中间变量:准确的说，新的计算图中没有中间变量，我们把MSE看成一个node，这个node接收2个输入x0和x1，前向传播时x0和x1计算产生的中间数据只会存在于MSE里面的forward方法中，一旦离开forward的作用范围，就马上从内存中被清除，但我们需要的的链接（inputs<->creator<->outputs）已经成功建立，反向传播也同理。
import numpy as np
from dezero import cuda
from dezero.core import Function, Variable, as_variable, as_array

'''
我们会发现dezero.functions里面函数大致分为两类：一类函数的反向传播的计算涉及到了正向传播的输入或者输出，需要保存inputs/outputs；另一类函数的反向传播只对回传过来的梯度gy做变形处理
'''

# ========= basic functions: sin/cos/tanh/exp/log

# cupy adapted
class Sin(Function):

    def forward(self, x):
        #根据输入的变量的module选择
        xp = cuda.get_array_module(x)
        y = xp.sin(x)
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

class Exp(Function):
    def forward(self, x):
        y = np.exp(x)
        return y
    def backward(self, gy):
        y = self.outputs[0]() # weakref
        gx = gy*y
        return gx
def exp(x):
    return Exp()(x)

class Log(Function):
    def forward(self, x):
        y = np.log(x)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        gx = gy / x
        return gx
    
def log(x):
    return Log()(x)

# ========= max / min/ clip
class Max(Function):
    def __init__(self, axis = None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def forward(self, x):
        y = x.max(axis = self.axis, keepdims = self.keepdims)
        return y
    # 反向传播的目标是：将梯度 gy（损失对输出的梯度）传播到输入 x 中最大值的位置，其余位置梯度为0。
    def backward(self, gy):
        x = self.inputs[0]
        y = self.outputs[0]()

        shape = utils.max_backward_shape(x, self.axis)
        gy = reshape(gy, shape)
        y = reshape(y, shape)
        # 生成掩码，仅在最大值位置为 True，其他位置为 False
        # 将梯度 gy 仅保留在最大值位置，其他位置置0。
        cond = (x.data == y.data)
        # brodcast_to会将gy在broadcast的地方复制若干次
        # cond.shape就是原始数据的shape
        gy = broadcast_to(gy, cond.shape)
        return gy*cond


class Min(Max):
    def forward(self, x):
        y = x.min(axis=self.axis, keepdims = self.keepdims)
        return y
    
def max(x, axis = None, keepdims = False):
    return Max(axis, keepdims)(x)

def min(x, axis=None, keepdims = False):
    return Min(axis, keepdims)(x)


class Clip(Function):
    def __init__(self, x_min, x_max):
        self.x_min = x_min
        self.x_max = x_max

    def forward(self, x):
        y = np.clip(x, self.x_min, self.x_max)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        mask = (x.data >= self.x_min)*(x.data <= self.x_max)
        gx = gy*mask
        return gx

def clip(x, x_min, x_max):
    return Clip(x_min, x_max)(x)


#============= shape =================

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

# GetFunction的注释从后往前看
class GetItem(Function):
    def __init__(self, slices):
        # 获取并保存人为指定的切片位置信息slices
        self.slices = slices
    def forward(self, x):
        #前向传播就是获取输入数据的切片
        y = x[self.slices]
        return y
    def backward(self, gy):
        x, = self.inputs
        f = GetItemGrad(self.slices, x.shape)
        return f(gy)
    
class GetItemGrad(Function):
    def __init__(self, slices, in_shape):
        self.slices = slices

        # in_shape 输入数据的shape，因为我们在反向传播的时候需要用到这个信息，这样才能正确还原输入的shape
        self.in_shape = in_shape

    def forward(self, gy):
        # 目前暂未使用cupy模块适配gpu
        assert isinstance(gy, np.ndarray), 'gy must be np.ndarray'
        # 初始化一个gx用来保存数据，这个gx的形状要求和输入数据x一致，数据类型指定为gy.dtype，因为所有继承自Function的函数的outputs都会被as_variable()转换成Variable类型，反倒是输入不一定时Variable类型：
        # Function：outputs = [Variable(as_array(y)) for y in ys]
        gx = np.zeros(self.in_shape, dtype = gy.dtype)
        # add ‘gy’ to ‘gx’ at position ‘self.slices’
        np.add.at(gx, self.slices,gy)
        return gx
    def backward(self, ggx):
        return get_item(ggx, self.slices)
    
def get_item(x, slices):
    # slices 这个参数就是指定切片的位置
    f = GetItem(slices)
    return f(x)


# ===== tensor related =====
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

# 为什么新的MeanSquaredError的内存效率更高？因为ndarray这个数据结构被高度优化过，ndarray类型数据一旦离开.forward的作用域，就会被清除。而我们前面定义的mse_simple因为用的是Variable变量做计算，因此有相当一部分中间变量被保留。
# 函数的输出作为 Variable 实例记录在计算图中，也就是说，在计算图存在期间，Variable实例和它内部的数据（ndarray）会保存在内存中。原来的实现方法每一个中间变量都是Variable实例，因此内存开销较大。

#而继承Function类实现的方法，中间结果没有座位Variable实例存储在内存中，所以在forward中使用的数据在正向传播完成后会立即被删除。

# 仿射变换，简单版
def linear_simple(x, W, b=None):
    t = matmul(x, W)
    if b is None:
        return t
    y = t+b
    t.data = None # release t.data(ndarray) for memory efficiency
    # 注意，只有t.data会被删除，而因为反向传播需要inputs/outputs/creator等数据，这些数据会被保留
    return y

# 放射变化，高效版
class Linear(Function):
    def forward(self, x, W, b):
        y = x.dot(W)
        if b is not None:
            y += b
        return y
    
    def backward(self, gy):
        x, W, b = self.inputs
        # 如果有偏置项，那么b的反向传播只接收gy即可，sum_to是考虑了broadcast_to的反向传播
        gb = None if b.data is None else sum_to(gy, b.shape)

        gx = matmul(gy, W.T)
        gW = matmul(x.T, gy)

        return gx, gW, gb

def linear(x, W, b=None):
    return Linear()(x, W, b)


class Softmax(Function):
    def __init__(self, axis = 1):
        self.axis = axis

    def forward(self, x):
        # 防止溢出
        y = x - x.max(axis=self.axis, keepdims = True)
        y = np.exp(y)
        y /= y.sum(axis = self.axis, keepdims = True)
        return y
    
    def backward(self, gy):
        '''
        check derivation 
        https://medium.com/towards-data-science/derivative-of-the-softmax-function-and-the-categorical-cross-entropy-loss-ffceefc081d1
        '''
        y = self.outputs[0]()
        gx = y*gy
        sumdx = gx.sum(axis = self.axis, keepdims = True)
        gx -= y*sumdx
        return gx

# 默认对行求softmax
def softmax(x, axis = 1):
    return Softmax(axis)(x)



# ======= neural activation function ======
def sigmoid_simple(x):
    # x=as_variable(x)确保ndarray被转换为Variable，虽然我觉得没有必要，但还是遵循传统，保留这一步
    x = as_variable(x)
    y = 1/(1+exp(-x))
    return y

class Sigmoid(Function):
    def forward(self, x):
        y = np.tanh(x*0.5)*0.5 + 0.5
        return y
    def backward(self, gy):
        y = self.outputs[0]()
        gx = gy*y*(1-y)
        return gx
    
def sigmoid(x):
    return Sigmoid()(x)

class ReLU(Function):
    def forward(self, x):
        y = np.maximum(x, 0.0)
        return y
    
    def backward(self, gy):
        x, = self.inputs
        mask = x.data >0
        gx = gy * mask
        return gx
    
def relu(x):
    return ReLU()(x)


# ============ loss function ================

# cross entropy loss
def softmax_cross_entropy_simple(x, t):

    # 这里的t是模型的输出
    x, t = as_variable(x), as_variable(t)
    N = x.shape[0]
    # 注意，这里softmax(x)以后得到的是和x.shape一样的数据
    p = softmax(x)
    p = clip(p, 1e-15, 1.0) # 防止log(0),将p设为大于1e-15
    log_p = log(p)
    # 提取出对应于训练数据的模型输出
    tlop_p = log_p[np.arange(N), t.data]
    y = -1*sum(tlop_p) / N
    return y

class SoftmaxCrossEntropy(Function):
    def forward(self, x, t):
        # x: model output: indices, not one-hot
        # N: numbder of data entries
        # t: true label, not one-hot
        N = x.shape[0]
        log_z = utils.logsumexp(x, axis = 1)
        log_p = x - log_z
        '''
        .flatten与.ravel()这两个函数实现的功能一样(展开数组),但在使用过程中flatten()分配了新的内存,但ravel()返回的是一个数组的视图,修改数组的视图，原数组也会被修改，但flatten因为分配了新的内存，因此修改flatten（）后的向量并不会影响原向量

        ravel的目的是将t这个(N,1)数组展平为(N,1),因为t有可能是(N,); t.ravel() 的作用：将 t 展平为一维数组，无论其原始形状是 (N,) 还是 (N, 1),确保 log_p 能正确索引到每个样本的真实类别位置
        '''
        log_p = log_p[np.arange(N), t.ravel()]
        y = -log_p.sum() / np.float32(N)
        return y
    
    def backward(self, gy):
        x, t = self.inputs
        N, CLS_NUM = x.shape

        gy *= 1/N
        y = softmax(x)

        #convert to one-hot
        # np.eye:  2-D array with ones on the diagonal and zeros elsewhere.
        t_onehot = np.eye(CLS_NUM, dtype = t.dtype)[t.data]
        # check https://www.cnblogs.com/chenmoshaoalen/p/18103756
        y = (y - t_onehot)*gy
        return y
    
def softmax_cross_entropy(x, t):
    return SoftmaxCrossEntropy()(x, t)

def accuracy(y, t):
    # 这里将y, t 又手动转化为 Variable是为了更好的兼容性:
    # 假如一开始传入的是Variable数据，因为我们没有在Variable类的方法中定义argmax，所以没办法用这个函数，我们只能取出Variable类里面的self.data，才能用.argmax，但是取data的这个操作是ndarray不具备的
    # 假如一开始传入的是ndarray，自然是没问题，不过为了兼容Variable，我们不如一开始就把他们全部转化为Variable
    y, t = as_variable(y), as_variable(t)

    pred = y.data.argmax(axis = 1).reshape(t.shape)
    result = (pred == t.data)
    acc = result.mean() # return np.float->scalar not ndarray, must be turn into ndarray via as_array
    return Variable(as_array(acc))
# 深度学习入门1：CNN
# 卷积层和池化层的实现
import numpy as np

# # use import to load function or define it within this file
# import sys
# sys.path.append('DL-code')
# from common.util import im2col， col2im

def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """
    把输入数据展开以适合滤波器，因此需要输入滤波器的形状filter_h, filter_w；而滤波器的通道数必然和输入数据一样，因此不需要输入
    Parameters
    ----------
    input_data : 由(数据量, 通道, 高, 长)的4维数组构成的输入数据
    filter_h : 滤波器的高
    filter_w : 滤波器的长
    stride : 步幅
    pad : 填充

    Returns
    -------
    col : 2维数组
    """
    N, C, H, W = input_data.shape
    # 计算经过卷积运算后输出值的形状
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # np.pad : [(0,0), (0,0), (pad,pad), (pad,pad)]-> [dim0, dim1, dim2, dim3]; (pad_before_0st_elem, pad_after_last_elem), elem refer to the elements of the dimension, remember that each dimension is a vector, so it has first and last element.
    img = np.pad(input_data, [(0,0), (0,0), (pad, pad), (pad, pad)], 'constant')

    # 用来保存展开后的输入数据，方便进行卷积运算
    # N = batch_size, C = channel num，(filter_h, filter_w)储存单个滤波作用域的数据，看作一个整体，那么有多少个这样的作用域呢？答案和最终输出的矩阵的元素个数一样，最终我们会输出一个(out_h, out_w)的矩阵（滤波器作用的结果），而每个滤波器作用域会产生1个元素，这就是最后两个维度的来源。
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))
    
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w
            #把image相应patches放入col：分区（patches）
            # y:y_max:stride (for H) → Start at y, take values up to y_max, moving with step stride
            # example: tmp = np.arange(12)
            # tmp[0:10:3] = array([0, 3, 6, 9])
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    print(col.shape)
    # 为什么要这样transpose：因为我们的最终目的是为了输出一个2dim矩阵以供计算，所以要调整原来的5维array，让特定的维度被压缩成1维；这里transpose以后进行了reshape，我们观察reshape里面的参数：N*out_h*out_w正好把col的第0、4、5维压缩成1维，第二个参数-1表示把余下的维度压缩成1维
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1) # (N*out_h*out_w, C*filter_h*filter_w)
    return col

# test code for better understanding
tmp = np.arange(10)
tmp[0:8:2]
# ========= end im2col def ==========
'''
page 218
- 如何理解：
'im2col会把输入数据展开以适合滤波器（权重）。具体地说，如图7-18所示,对于输入数据，将应用滤波器的区域（3维方块）横向展开为1列'？
即把单个滤波器应用区域（方块）展开成一行，然后再把所有滤波器应用区域（方块—）展开后的行排成一列。

- 如何理解：
'使用 im2col展开输入数据后，之后就只需将卷积层的滤波器（权重）纵
向展开为1列，并计算2个矩阵的乘积即可（参照图7-19）'：
因为展开输入数据后得到的是很多行排成的列，那么就需要把多个滤波器（权重）展开成很多单独的列（每个滤波器展开成一列）排成的行，进行常规矩阵乘法：行x列。(page 219 的图7-19是很好的illus)

'''
x = np.random.rand(10, 1, 28, 28)
x.shape
x[0].shape

x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1, 5, 5, stride = 1, pad = 0)
col1.shape # (9, 75)
'''
如何理解输出的shape：
- 先来看输入的shape：(1,3,7,7), N=1, C=3, H=W=7
- 再看指定的filter shape：3x5x5，stride=1，那么这个filter作用在输入数据上会产生一个(N, H, W) = (3,5,5)的方块，把一个这样的方块横向展开后，会得到3x5x5=75个元素（横向），这样的方块一共会有3x3=9个（边长为5的filter在7x7的square上以stride=1只能在横纵坐标上移动3格），而我们需要把9个这样的方块展开成的行排成一列（“横向展开为1列”），因此一共有9行，这就是（9，75）的由来。
- 最后来推敲一下，为什么.transpose(0,4,5,1,2,3).reshape(N*out_h*out_w)可以达到我们的目的：先来重申一下各个维度分别是什么：
(0, 1, 2, 3, 4, 5) = (N, C, filter_h, filter_w, out_h, out_w)
我们把N out_h out_w 压成1维，遵循以下原则：从上到下、从左到右，也就是说，把N个（out_h, out_w)的2维矩阵合成一个向量，这个向量的每一个元素都是一个2维矩阵(out_h, out_w)，接着，在不改变每个2维矩阵的左右顺序的情况下，在二维矩阵各自的位置上把每一个二维矩阵(out_h, out_w)的行按照从上到下的顺序从左到右排列。排好以后得到的这个向量的每一个元素，又是一个3维array(C, filter_h, filter_2)，也就是每个filter的作用区域，这样的作用区域一共有N*out_h*out_w个，完美符合我们上面一个小节的reasoning。
'''

# 我们将要对im2col进行逆运算：
def col2im(col, input_shape, filter_h, filter_w, stride=1, pad=0):
    """

    Parameters
    ----------
    col :
    input_shape : 输入数据的形状（例：(10, 1, 28, 28)）
    filter_h :
    filter_w
    stride
    pad

    Returns
    -------

    """
    N, C, H, W = input_shape # input：.forward(x)中的x，也就是原始图像batch
    out_h = (H + 2*pad - filter_h)//stride + 1
    out_w = (W + 2*pad - filter_w)//stride + 1

    # col.shape = (N*out_h*out_w, C*filter_h*filter_w)
    #.reshape.transpose->(N,C,filter_h,filter_w,out_h,out_w)
    # 也就是转换成col2im里面col的shape
    col = col.reshape(N, out_h, out_w, C, filter_h, filter_w).transpose(0, 3, 4, 5, 1, 2)
    
    # 用来保存提取的元素，我们需要从filter的行为倒推img的shape
    # 注意，filter有可能有重叠区域，因此img的shape最好能兼容filter的stride以及pad（filter可能跑到x的外面），哪怕比x大一圈，这样可以省去很多判断，最后我们只需要输出指定区域的img就可以还原x
    # 例子：原始图像4x4,filter.shape4x4,stride=2,pad=0，那么特征图是一个2x2的
    # original book error: np.zeros((N, C, H + 2*pad + stride - 1, W + 2*pad + stride - 1))
    img = np.zeros((N, C, H + 2*pad , W + 2*pad))
    for y in range(filter_h):
        y_max = y + stride*out_h
        for x in range(filter_w):
            x_max = x + stride*out_w

            # 计算当前卷积核在图像中的覆盖范围 y:y_max:stride 和 x:x_max:stride, 原代码这里出错，不是+=，而是=，因为加的操作只会在卷积的时候进行，也就im2col，而还原的时候不需要。退一步说，原始图片的每个像素点只有一个唯一的值，你整别的加上去干什么？
            img[:, :, y:y_max:stride, x:x_max:stride] = col[:, :, y, x, :, :]

    return img[:, :, pad:H + pad, pad:W + pad]

# test

tmp_mat = np.array(np.around(np.random.rand(1,3,4,4)*100))
col1 = im2col(tmp_mat,3,3,pad=1)

rev = col2im(col1, tmp_mat.shape,3,3,pad=1)
#%%
#========= 实现卷积层 ==========
import numpy as np

class Convolution:
    def __init__(self, W, b, stride = 1, pad = 0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad

        # 中间数据（backward时使用）
        self.x = None   
        self.col = None
        self.col_W = None
        
        # 权重和偏置参数的梯度
        self.dW = None
        self.db = None


    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1+(H+2*self.pad - FH)/self.stride)
        out_w = int(1+(W+2*self.pad - FW)/self.stride)

        col = im2col(x, FH, FW, self.stride, self.pad)
        # 因为滤波器参数矩阵W的shape比较简单: (FN, C, FH, FW)，不需要像im2col那样的专门处理的函数就可以把他转换成希望的样子
        col_W = self.W.reshape(FN, -1).T # shape: (C*FH*FW,FN)
        # 乘法的示意图见书page213图7-13:
        # (N,C,H,W) DOT (FN,C,FH,FW) ->(N,FN,OH,OW)
        # (N,FN,OH,OW) + (FN,1,1) ->(N,FN,OH,OW)
        # 经过im2col后
        # col.shape = (N*out_h*out_w, C*filter_h*filter_w)
        # 等价于 (N*out_h*out_w, C*FH*FW)
        # 因此下面这个out是(N*out_h*out_w, C*FH*FW) dot (C*FH*FW,FN) = (N*out_h*out_w, FN), 注意，加法不影响shape
        out = np.dot(col, col_W) + self.b

        # reshape->(N,out_h, out_w, FN) -transpose->(N,FN,out_h, out_w)
        # 结果和图7-13的输出结果的shape完全一致
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)

        # 储存中间数据，backward时使用
        self.x = x
        self.col = col
        self.col_W = col_W

        return out
    def backward(self, dout):
        '''
        affine反向传播参考：https://blog.csdn.net/m0_60461719/article/details/133951221
        '''
        FN, C, FH, FW = self.W.shape
        dout = dout.transpose(0,2,3,1).reshape(-1, FN)

        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.col.T, dout)
        self.dW = self.dW.transpose(1, 0).reshape(FN, C, FH, FW)

        dcol = np.dot(dout, self.col_W.T)
        # must define col2im function
        dx = col2im(dcol, self.x.shape, FH, FW, self.stride, self.pad)

        return dx


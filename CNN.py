# 深度学习入门1：CNN
# 卷积层和池化层的实现
import numpy as np

# # use import to load function or define it within this file
# import sys
# sys.path.append('DL-code')
# from common.util import im2col

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
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]
    print(col.shape)
    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N*out_h*out_w, -1)
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
'''
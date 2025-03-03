import numpy as np
from dezero import cuda
from dezero.core import Function, as_variable
from dezero.functions import linear, broadcast_to
from dezero.utils import pair, get_conv_outsize, get_deconv_outsize


# =========== numpy im2col ============
def im2col_array(img, kernel_size, stride, pad, to_matrix = True):
    '''
    img: input data, typically a np.array
    '''

    N, C, H, W = img.shape
    KH, KW = pair(kernel_size)
    SH, SW = pair(stride)
    PH, PW = pair(pad)
    OH = get_conv_outsize(H, KH, SH, PH)
    OW = get_conv_outsize(W, KW, SW, PW)

    xp = cuda.get_array_module(img)
    if xp != np:
        col = _im2col_gpu(img, kernel_size, stride, pad)
    else:
        img = np.pad(img, ((0, 0), (0, 0), (PH, PH + SH -1), (PW, PW + SW - 1)), mode = 'constant', constant_values = (0,))

        col = np.ndarray((N, C, KH, KW, OH, OW), dtype = img.dtype)

        for j in range(KH):
            j_lim = j + SH*OH
            for i in range(KW):
                i_lim = i + SW*OW
                col[:, :, j, i, :, :] = img[:, :, j:j_lim:SH, i:i_lim:SW]
    '''
    用子数组的思路考察ndarray切片与赋值
    - 右侧切片：img[:, :, j:j_lim:SH, i:i_lim:SW],该切片从填充后的输入图像中提取数据：

    - 高度方向：起始索引为j，步长SH，结束索引j_lim = j + SH*OH。由于步长为SH且需覆盖OH个输出位置，实际切片长度为OH。

    - 宽度方向：同理，切片长度为OW。

    结果形状：(N, C, OH, OW)，对应批大小、通道数、输出特征图的高和宽。

    左侧赋值：col[:, :, j, i, :, :]
    目标位置是col数组的(N, C, j, i, OH, OW)部分。由于j和i是标量索引，col的这部分 子数组 形状为(N, C, OH, OW)，与右侧切片形状完全一致
    '''
    if to_matrix:
        col = col.transpose((0, 4, 5, 1, 2, 3)).reshape((N*OH*OW, -1))

    return col

# =========== im2col / col2im ============
class Im2col(Function):
    def __init__(self, kernel_size, stride, pad, to_matrix):
        super().__init__()
        self.input_shape = None
        self.kernel_size = kernel_size
        self.stride = stride
        self.pad = pad
        self.to_matrix = to_matrix

    def forward(self, x):
        self.input_shape = x.shape
        y = im2col_array(x, self.kernel_size, self.stride, self.pad, self.to_matrix)
        return y
    
    def backward(self, gy):
        gx = col2im(gy, self.input_shape, self.kernel_size, self.stride, self.pad, self.to_matrix)

        return gx
    

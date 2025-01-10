import torch

inputs = torch.tensor(
    [[0.43, 0.15, 0.89], # Your (x^1)
    [0.55, 0.87, 0.66], # journey (x^2)
    [0.57, 0.85, 0.64], # starts (x^3)
    [0.22, 0.58, 0.33], # with (x^4)
    [0.77, 0.25, 0.10], # one (x^5)
    [0.05, 0.80, 0.55]] # step (x^6)
    )

# for 1 query

# step 1: calculate attension scores:
# pick the 2nd input as "query"
query = inputs[1]
# initialize a vector(tensor) to store the result
# torch.empty() is similar to np.empty()
# inputs.shape: torch.Size([6, 3])
attn_scores_2 = torch.empty(inputs.shape[0])

#把inputs的每一行x_i拿出来和query做点乘
for i, x_i in enumerate(inputs):
    attn_scores_2[i] = torch.dot(x_i,query)
print(attn_scores_2)

# above for loop equivalent to:
attn_scores_2 = inputs@query

# step 2: normalize (sum to 1, make it probabalistic)

# example: simple sum
attn_weights_2_tmp = attn_scores_2/attn_scores_2.sum()
print("Attention weights:", attn_weights_2_tmp)
print("Sum:", attn_weights_2_tmp.sum())

# softmax naive way (take exp element-wise then sum)
def softmax_naive(x):
    return torch.exp(x)/torch.exp(x).sum(dim=0)

attn_weights_2_naive = softmax_naive(attn_scores_2)
print("Attention weights:", attn_weights_2_naive)
print("Sum:", attn_weights_2_naive.sum())

# softmax torch way
# 注意，softmax里面dim=0，因为attn_scores_2是一个向量，向量默认为矩阵的列向量，要对矩阵的列向量做softmax就是dim=0 （类似axis=0）
attn_weights_2 = torch.softmax(attn_scores_2, dim=0)
print("Attention weights:", attn_weights_2)
print("Sum:", attn_weights_2.sum())

# step 3: get context vector
query = inputs[1]

# initialize an empty vec to store results
context_vec_2 = torch.zeros(query.shape)

# weighted (attn weight) sum of all input tokens
for i, x_i in enumerate(inputs):
    context_vec_2 += attn_weights_2[i]*x_i


# for all query
attn_scores = torch.empty(6,6)
for i, x_i in enumerate(inputs):
    for j, x_j in enumerate(inputs):
        attn_scores[i,j] = torch.dot(x_i,x_j)
print(attn_scores)

# above equivalent to:
# attn_scores_2 is the 2nd row of attn_scores
attn_scores = inputs@inputs.T

# softmax (maxtrix form)

# 注意，这里dim=-1等价于2维矩阵的dim=1，为什么之前softmax的dim=0而这里是dim=1？因为之前仅仅计算一个query，attn_scores_2是一个列向量，但是在这里计算所有querys，计算结果attn_scores是一个矩阵，原来的列向量是这个矩阵的行向量，即attn_scores_2是attn_scores的第二行
attn_weights = torch.softmax(attn_scores, dim=-1)
print(attn_weights)

# verify:
row_2_sum = torch.sum(attn_weights[1])
row_sum = torch.sum(attn_weights, dim=1)
print(row_2_sum)
print(row_sum)

# get context matrix(calculate and stack all context vec)
all_context_vec = attn_weights@inputs

#===== part 2: adding trainable weights =====

# step 1: initialize
torch.manual_seed(123)
x_2 = inputs[1]
d_in=inputs.shape[1] # the input embeddin size = 3
d_out=2
W_query = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False) #object type: torch.nn.parameter.Parameter
W_key = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)
W_value = torch.nn.Parameter(torch.rand(d_in, d_out), requires_grad=False)

# should all be vector of dim 2
query_2 = x_2@W_query
key_2 = x_2@W_key
value_2 = x_2@W_value

# step 2: compute attn scores:
# simply use query_2 to do dot product with all other input: 
# attn scores = q@k

# we first obtain all keys and values via matrix multiplication
keys = inputs @ W_key
values = inputs @ W_value

# attn scores of input 2
keys_2 = keys[1]
attn_score_22 = query_2.dot(keys_2)
print(attn_score_22)

# we can generalize this computation to all keys to get the attn scores of input 2
attn_scores_2 = query_2@keys.T
# more general, attn scores to all inputs
inputs@W_query@keys.T # the 2nd row is the attn scores of input 2
print(attn_scores_2)

d_k = keys.shape[-1] # the embedding dim of keys
# attn_scores_2 is a vector, hence dim=-1 equal dim = 0
attn_weights_2 = torch.softmax(attn_scores_2/d_k**0.5, dim=-1) 
print(attn_weights_2)

# step 3: compute the context vector as a weighted sum over the value vectors
context_vec_2 = attn_weights_2@values
print(context_vec_2)


# final step: assemble
import torch.nn as nn

class SelfAttention_v1(nn.Module):
    def __init__(self, d_in, d_out):
        super().__init__()
        self.W_query = nn.Parameter(torch.rand(d_in, d_out))
        self.W_key = nn.Parameter(torch.rand(d_in, d_out))
        self.W_value = nn.Parameter(torch.rand(d_in, d_out))
    
    def forward(self, x):
        keys = x@self.W_key
        queries = x@self.W_query
        values = x@self.W_value
        attn_scores = queries@keys.T #omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # here, attn_scores is a matrix, dim=-1 equal dim=1, meaning sum along row
        context_vec = attn_weights@values
        return context_vec
    
# test
torch.manual_seed(123)
sa_v1 = SelfAttention_v1(d_in, d_out)
# the 2nd row of result should equal to context_vec_2
print(sa_v1.forward(inputs)) # equivalent to sa_v1(inputs)

# v2: use nn.Linear instead of nn.Parameter to improve efficiency
#%%
class SelfAttention_v2(nn.Module):
    def __init__(self, d_in, d_out,qkv_bias=False):
        super().__init__()
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        # nn.Linear is a class, with initialization with in_features and out_features, here are d_in and d_out, after initiating the nn.Linear, you can pass 
    def forward(self, x):
        keys = self.W_key(x) # same as self.W_key.forward(x)
        queries = self.W_query(x)
        values = self.W_value(x)
        attn_scores = queries@keys.T #omega
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1) # here, attn_scores is a matrix, dim=-1 equal dim=1, meaning sum along row
        context_vec = attn_weights@values
        return context_vec
#%%
# test2
'''
v1 and v2 give different outputs because
they use different initial weights for the weight matrices since nn.Linear uses a more sophisticated weight initialization scheme.
'''
torch.manual_seed(789)
sa_v2 = SelfAttention_v2(d_in, d_out)
# the 2nd row of result should equal to context_vec_2
print(sa_v2.forward(inputs)) # different from that of v1


# compare how their weights differ
sa_v2.W_key.weight
sa_v1.W_key

# %%
# ====== mask =========
queries = sa_v2.W_query(inputs)
keys = sa_v2.W_key(inputs)
attn_scores = queries@keys.T
attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5,dim=-1)
print(attn_weights)

context_length=attn_scores.shape[0]

# tril = lower trianle, triu = upper triangle
mask_simple = torch.tril(torch.ones(context_length, context_length))
print(mask_simple)
masked_simple = attn_weights*mask_simple
print(masked_simple) # elements above diag expected to be zero

# then renormalize attn weights to sum up to 1 in each row
# keepdim=True will keep the results as column vector, not row vector
row_sums = masked_simple.sum(dim=-1, keepdim=True)
masked_simple_norm = masked_simple/row_sums
print(masked_simple_norm)


# another way of doing masked normalization
# create a squared matrix with all elements above(not included) diag to be 1, other 0 (upper triangle matrix)
# diagonal=0 means start from main diag, diagonal=1 means start from 1 above main diag. diagonal can be negative
mask = torch.triu(torch.ones(context_length, context_length), diagonal=1)
# .masked_fill 把mask.bool()为True的位置全部fill成 -torch.inf
masked = attn_scores.masked_fill(mask.bool(), -torch.inf)
attn_weights = torch.softmax(masked/keys.shape[-1]**0.5, dim=1)
'''
此处应添加笔记，为什么上面的simple_norm用的是0 而这里的modified softmax版本用的是负无穷：因为在softmax里面，输入值取0，softmax之后会变成1，只有当输出为0时，我们才可以把他忽略掉，也就是被masked掉，取正值还是负值都会对最终的weights造成影响，也就是information leakage。参见page 76页的notes
'''

# %%
# Combining dropouts:
torch.manual_seed(123)
# initializatino, setting dropout rate
dropout = torch.nn.Dropout(0.5)
example = torch.ones(6,6)
print(dropout(example))

torch.manual_seed(123)
print(dropout(attn_weights))

# adapt attention class to mini batch
batch = torch.stack((inputs,inputs), dim = 0)
print(batch.shape)


# full causal attention
class CausalAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, qkv_bias=False):
        super().__init__()
        self.d_out = d_out
        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.dropout = nn.Dropout(dropout)
        # make sure all params run in same device
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))

    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        attn_scores = queries@keys.transpose(1,2) #transpose dim1 and dim2
        #note that masked_fill_ with a trailing _, this is to avoid unnecessay memory copies

        # why [:num_tokens, :num_tokens]?
        # Original mask truncated to the number of tokens and converted to boolean, because num of tokens not necessarily = context_length

        # remember that in gpt2, context_length=1024, context_length is related to RAM consumption so it has to be considered carefully.
        attn_scores.masked_fill_(self.mask.bool()[:num_tokens, :num_tokens], -torch.inf)

        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)

        attn_weights = self.dropout(attn_weights)

        context_vec = attn_weights@values
        return context_vec
    
torch.manual_seed(123)
# here we set context_length to be the num_of_tokens, keep in mind that
# this is only for demonstration, it is not always true
context_length = batch.shape[1]
# dropout rate = 0.0
d_out=2
ca = CausalAttention(d_in, d_out, context_length, 0,0)

context_vecs = ca(batch)
print(context_vecs)
print(f'context_vecs shape:{context_vecs.shape}')





# extend single head to multihead
class MultiHeadAttentionWrapper(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        # simply stack num_heads CausalAttention
        self.heads = nn.ModuleList([CausalAttention(d_in, d_out, context_length, dropout, qkv_bias) for _ in range(num_heads)])

    def forward(self, x):
        # pass x to num_heads of CausalAttention then stack the results
        return torch.cat([head(x) for head in self.heads], dim=-1)

#%%
# test multihead
torch.manual_seed(123)
context_length = batch.shape[1] # set it to number of tokens for simplicity
d_in, d_out = 3, 2

mha = MultiHeadAttentionWrapper(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)

print(context_vecs)
# print(f'context_vecs shape: {context_vecs.shape}')


# context_vecs.view(2,6,2,2).shape
# tmp = nn.Linear(2,2)
# tmp(context_vecs.view(2,6,2,2)).view(2,6,4)
# context_vecs.contiguous().view()

#%%
'''
the following MultiHeadAttention class integrates the multi-head functionality within a single class.
'''

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert (d_out%num_heads == 0), "d_out must be divisable by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        # reason of using '//': since before we have already ensure d_out must be divisable by n_heads, using // instead of / make sure the result be integer. try comparing the difference of 10//2 and 10/2
        self.head_dim = d_out//num_heads
        self.W_query = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias = qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias = qkv_bias)
        # this projection layer is not necessary, but for completeness*
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.triu(torch.ones(context_length, context_length), diagonal=1))
    
    def forward(self, x):
        b, num_tokens, d_in = x.shape
        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        # split weights
        keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
        values = values.view(b, num_tokens, self.num_heads, self.head_dim)
        queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)
        # optional, to help understand the transformation
        print(f'#1 qkv matrices like:\n{keys}')

        # compare below to old method: attn_scores = queries@keys.transpose(1,2)
        # dim1 and dim2 of old methods are in fact the dim2 and dim3 of current method
        # Transpose: (b, num_tokens, num_heads, head_dim) -> (b, num_heads, num_tokens, head_dim)
        # easy to understand why we need to transpose dim1 dim2: remember in old method, within each head, the 2 dims involved in matrix multiplication are (num_tokens, head_dim)
        keys = keys.transpose(1,2)
        values = values.transpose(1,2)
        queries = queries.transpose(1,2)
        # print(f'#2 shape of qkv after transpose:\n{keys.shape}')

        attn_scores = queries@keys.transpose(2,3)
        # print(f'#3 attn_scores:\n{attn_scores}')
        mask_bool = self.mask.bool()[:num_tokens, :num_tokens]

        attn_scores.masked_fill_(mask_bool, -torch.inf)
        # print(f'#4 masked attn_scores:\n{attn_scores}')
        attn_weights = torch.softmax(attn_scores/keys.shape[-1]**0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights@values).transpose(1,2)
        # print(f'#5 context_vec before merge:\n{context_vec}')
        # contiguous make sure the memory are contiguous, for efficiency
        context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
        # print(f'#6 context_vec after merge:\n{context_vec}')
        context_vec = self.out_proj(context_vec)
        # print(f'#7 context_vec being out_proj:\n{context_vec}')

        return context_vec




# test
#%%
torch.manual_seed(123)
batch_size, context_length, d_in = batch.shape
d_out=4 # 书上用的是2，但其实不正确
mha = MultiHeadAttention(d_in, d_out, context_length, 0.0, num_heads=2)
context_vecs = mha(batch)
print(context_vecs)
mha.W_key(batch).view(2,6,2,2).transpose(1,2).shape
'''
注意，虽然两种计算多头注意力的方法非常相似，只是第二种用了一种更加高效的方法去计算，但实际上，因为二者初始化的nn.Linear的shape不同，而nn.Linear的初始化是具有随机性的，因此二者最后的计算结果也不同
'''


#%%
# 思考与验证
torch.manual_seed(123)
tmp = nn.Linear(4,4)
tmp.weight
# 把最后一个维度split成2x2
tmp.weight.view(4,2,2)
#注意观察最后生成的tensor，是4个2x2的矩阵组成的一列
'''
那么上面的multiheadattn的维度是怎么转换怎么split的呢？
观察，一开始初始化的W_q/W_k/W_v的维度都是(d_in, d_out)，接着我们把d_out分拆成了
head_dim = d_out//num_heads -> d_out = head_dim*num_heads，而num_heads=2，d_out=4，因此head_dim=2，也就是说，每个head最后生成的context_vecs的列数是2，这正好和老方法的输入维度相对应。

由此我们可以得出，老方法的输入shape为(2,6,2)，最后2个head的输出结果concat在一起，输出context_vec的shape是(2,6,4)，因为新老等价，而新方法的初始矩阵的形状是(2,6,4)，我们需要把最后一维split开，变成了(2,6,2,2)，把多出来的一维num_heads和num_tokens交换，得到(2,2,6,2)，由4变成2x2:head 1 (2,6,2), head 2 (2,6,2)，同时对这两个heads做运算，运算方法和老的计算方法一样。这里之所以要进行transpose(1,2)就是因为split的动作会多出一个维度num_heads并且这个维度出现在我们不需要的位置上，我们真正要进行矩阵乘法运算的维度和老方法一样，都是(6,2)(对应(num_tokens,head_dim)，其中head_dim又是单头里面的d_out)。第二次transpose(2,3)等价于牢房里面的.transpose(1,2)，矩阵乘法右边的那个矩阵的最后两个维度转置，这才能做乘法。
'''
# %%

# LLM from scratch notes

## Attentino Mechanism
- **notations**: **attn scores** $\omega$, **attn weight** $\alpha$
- **attention weights**(page 66): typically calcualted by softmaxing the attention scores, and it determines the extent to which a context vector depends on the different parts of the input (i.e., to what extent the network focuses on different parts of the input) = **context**
- **weight parameters**(page 66): aka **weight matrices** $W_k, W_q, W_v$ for example, are the fundamental, learned coefficients that define the network’s connections, while attention weights are dynamic, context-specific values. 
- **Why scaling the attn scores?**(page 69): The reason for the scaling by the embedding dimension size is to improve the training performance by **avoiding small gradients**. For instance, when scaling up the embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate. 
- **nn.Module**(page 71): a fundamental building block of PyTorch models that provides necessary functionalities for model layer creation and management.
- **新老多头注意力计算方法之区别**

    那么新的multiheadattn的维度是怎么转换怎么split的呢？
    观察，一开始初始化的W_q/W_k/W_v的维度都是(d_in, d_out)，注意，这里的d_out如果转换成head，就是单头里面的head_dim（单个头的维度）。初始化以后的W_qkv接受batch(6,3)作为输入参数，然后输出queries keys values的shape都是(6,2)。接着我们把d_out（=4）分拆成了 head_dim = d_out//num_heads -> d_out = head_dim*num_heads，而num_heads=2，d_out=4，因此head_dim=2，也就是说，每个head最后生成的context_vecs的列数是2，这正好和老方法的输入维度相对应。

    \
    由此我们可以得出：老方法的输入shape为(2,6,2)，最后2个head的输出结果concat在一起，输出context_vec的shape是(2,6,4)，因为新老等价，而新方法的初始矩阵的形状是(2,6,4)，我们需要把最后一维split开，变成了(2,6,2,2)，把多出来的一维num_heads和num_tokens交换，得到(2,2,6,2)，由4变成2x2:head 1 (2,6,2), head 2 (2,6,2)，同时对这两个heads做运算，运算方法和老的计算方法一样。这里之所以要进行transpose(1,2)就是因为split的动作会多出一个维度num_heads并且这个维度出现在我们不需要的位置上，我们真正要进行矩阵乘法运算的维度和老方法一样，都是(6,2)(对应(num_tokens,head_dim)，其中head_dim又是单头里面的d_out)。第二次transpose(2,3)等价于牢房里面的.transpose(1,2)，矩阵乘法右边的那个矩阵的最后两个维度转置，这才能做乘法。
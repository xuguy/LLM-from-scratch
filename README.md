# LLM from scratch notes

## Attentino Mechanism
- **notations**: **attn scores** $\omega$, **attn weight** $\alpha$
- **attention weights**(page 66): typically calcualted by softmaxing the attention scores, and it determines the extent to which a context vector depends on the different parts of the input (i.e., to what extent the network focuses on different parts of the input) = **context**
- **weight parameters**(page 66): aka **weight matrices** $W_k, W_q, W_v$ for example, are the fundamental, learned coefficients that define the network’s connections, while attention weights are dynamic, context-specific values. 
- **Why scaling the attn scores?**(page 69): The reason for the scaling by the embedding dimension size is to improve the training performance by **avoiding small gradients**. For instance, when scaling up the embedding dimension, which is typically greater than 1,000 for GPT-like LLMs, large dot products can result in very small gradients during backpropagation due to the softmax function applied to them. As dot products increase, the softmax function behaves more like a step function, resulting in gradients nearing zero. These small gradients can drastically slow down learning or cause training to stagnate. 
- **nn.Module**(page 71): a fundamental building block of PyTorch models that provides necessary functionalities for model layer creation and management.
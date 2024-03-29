# https://jaykmody.com/blog/gpt-from-scratch/

import numpy as np

# progress bar for Command Line Interface
from tqdm import tqdm
from utils import load_encoder_hparams_and_params
import fire


def gelu(x):
    """
    GELU activation function.

    :param x: The input tensor.

    :return: The output tensor.
    """

    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))


def softmax(x):
    """
    Softmax activation function.
    
    :param x: The input tensor.
    
    :return: The output tensor.
    """

    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def layer_norm(x, g, b, eps: float=1e-5):
    """
    Layer normalization.
    
    :param x: The input tensor.
    :param g: The gamma parameter.
    :param b: The beta parameter.
    
    :return: The output tensor.
    """
    mean = np.mean(x, axis=-1, keepdims=True)
    variance = np.var(x, axis=-1, keepdims=True)
    x = (x - mean) / np.sqrt(variance + eps)  # normalize x to have mean=0 and var=1 over last axis
    return g * x + b  # scale and offset with gamma/beta params


def linear(x, w, b):
    """
    Linear transformation.
    
    :param x: The input tensor.
    :param w: The weight matrix.
    :param b: The bias vector.
    
    :return: The output tensor.
    """

    # return np.dot(x, w) + b
    return x @ w + b


def ffn(x, c_fc, c_proj):
    """
    Position-wise feed forward network.
    
    :param x: The input tensor.
    :param c_fc: The parameters for the first fully connected layer.
    :param c_proj: The parameters for the second fully connected layer.
    
    :return: The output tensor.
    """

    #  project up
    a = gelu(linear(x, **c_fc))  

    # project back down
    x = linear(a, **c_proj) 

    return x


def attention_mask(q, k, v, mask):
    """
    Apply a mask to the attention scores.
    
    :param q: The query tensor.
    :param k: The key tensor.
    :param v: The value tensor.
    :param mask: The mask tensor.
    
    :return: The output tensor.
    """

    return softmax(q @ k.T / np.sqrt(q.shape[-1]) + mask) @ v


def mha(x, c_attn, c_proj, n_head):  # [n_seq, n_embd] -> [n_seq, n_embd]
    """
    Multi-head causal self attention.
    
    :param x: The input tensor.
    :param c_attn: The parameters for the attention projection.
    :param c_proj: The parameters for the output projection.
    :param n_head: The number of attention heads.
    
    :return: The output tensor.
    """
    
    # qkv projection
    x = linear(x, **c_attn)  # [n_seq, n_embd] -> [n_seq, 3*n_embd]

    # split into qkv
    qkv = np.split(x, 3, axis=-1)  # [n_seq, 3*n_embd] -> [3, n_seq, n_embd]

    # split into heads
    qkv_heads = list(map(lambda x: np.split(x, n_head, axis=-1), qkv))  # [3, n_seq, n_embd] -> [3, n_head, n_seq, n_embd/n_head]

    # causal mask to hide future inputs from being attended to
    causal_mask = (1 - np.tri(x.shape[0], dtype=x.dtype)) * -1e10  # [n_seq, n_seq]

    # perform attention over each head
    out_heads = [attention_mask(q, k, v, causal_mask) for q, k, v in zip(*qkv_heads)]  # [3, n_head, n_seq, n_embd/n_head] -> [n_head, n_seq, n_embd/n_head]

    # merge heads
    x = np.hstack(out_heads)  # [n_head, n_seq, n_embd/n_head] -> [n_seq, n_embd]

    # out projection
    x = linear(x, **c_proj)  # [n_seq, n_embd] -> [n_seq, n_embd]

    return x


def transformer_block(x, mlp, attn, ln_1, ln_2, n_head):
    """
    A transformer block.
    
    :param x: The input tensor.
    :param mlp: The parameters for the position-wise feed forward network.
    :param attn: The parameters for the multi-head causal self attention.
    :param ln_1: The parameters for the first layer normalization.
    :param ln_2: The parameters for the second layer normalization.
    :param n_head: The number of attention heads.
    
    :return: The output tensor.
    """

    # multi-head causal self attention
    x = x + mha(layer_norm(x, **ln_1), **attn, n_head=n_head)  
    # position-wise feed forward network
    x = x + ffn(layer_norm(x, **ln_2), **mlp)  

    return x


# ______________________________MODEL______________________________


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    GPT-2 model.

    :param inputs: A list of integers representing the input tokens.
    :param wte: The word token embeddings. (a lookup matrix of N-vocab, n_embd) (in order to choose words)    
    :param wpe: The position embeddings. (a lookup matrix of N-ctx, n_embd) (in order to order words in a sentence)
    :param blocks: A list of transformer blocks. 
    :param ln_f: The final layer normalization.
    :param n_head: The number of attention heads.

    :return: A list of integers representing the output tokens.
    """

    x = wte[inputs] + wpe[np.arange(len(inputs))]
    for block in blocks:
        x = transformer_block(x, **block, n_head=n_head)
        
    x = layer_norm(x, **ln_f)  
    return x @ wte.T  


def generate_response(inputs, params, n_head, n_tokens_to_generate=100):
    """
    Generate a response from the GPT model.

    :param inputs: A list of integers representing the input tokens.
    :param n_tokens: The number of tokens to generate.
    :return: A list of integers representing the output tokens.
    """

    for _ in tqdm(range(n_tokens_to_generate), "generating"):  # auto-regressive decode loop
        logits = gpt2(inputs, **params, n_head=n_head)  # model forward pass
        next_id = np.argmax(logits[-1])  # greedy sampling
        inputs.append(int(next_id))  # append prediction to input

    return inputs[len(inputs) - n_tokens_to_generate :]  # only return generated ids

# ______________________________MAIN______________________________


def main(prompt:str, n_tokens_to_generate: int=40, model_size: str="124M", models_dir: str="models"):   
    # load encoder, hparams, and params from the released open-ai gpt-2 files
    encoder, hparams, params = load_encoder_hparams_and_params(model_size, models_dir)

    # encode the input string using the BPE tokenizer
    input_ids = encoder.encode(prompt)

    # make sure we are not surpassing the max sequence length of our model
    assert len(input_ids) + n_tokens_to_generate < hparams["n_ctx"]

    # generate output ids
    output_ids = generate_response(input_ids, params, hparams["n_head"], n_tokens_to_generate)

    # decode the ids back into a string
    output_text = encoder.decode(output_ids)

    return output_text
    

if __name__ == "__main__":
    fire.Fire(main)

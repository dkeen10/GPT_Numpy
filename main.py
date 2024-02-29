# https://jaykmody.com/blog/gpt-from-scratch/

import numpy as np

# progress bar for Command Line Interface
from tqdm import tqdm
from transformers import GPT2Tokenizer
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

    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)




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


def ffn(x, c_fc, c_proj):
    """
    Position-wise feed forward network.
    
    :param x: The input tensor.
    :param c_fc: The parameters for the first fully connected layer.
    :param c_proj: The parameters for the second fully connected layer.
    
    :return: The output tensor.
    """

    # [n_seq, n_embd] -> [n_seq, n_embd]
    # project up
    a = gelu(linear(x, **c_fc))  # [n_seq, n_embd] -> [n_seq, 4*n_embd]

    # project back down
    x = linear(a, **c_proj)  # [n_seq, 4*n_embd] -> [n_seq, n_embd]

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
    return x


def generate_response(inputs, params, n_head, n_tokens_to_generate=100):
    """
    Generate a response from the GPT model.

    :param inputs: A list of integers representing the input tokens.
    :param n_tokens: The number of tokens to generate.
    :return: A list of integers representing the output tokens.
    """

    for i in tqdm(range(n_tokens_to_generate)):
        output = gpt2(inputs, **params, n_head=n_head)
        next_token = np.argmax(output[-1])
        inputs.append(int(next_token))

    # return generated tokens only
    return inputs[len(inputs) - n_tokens_to_generate]






# ______________________________TRAINING______________________________


def lm_loss(inputs, targets):
    """
    Get the cross entropy loss of the model.
    """
    x, y = inputs[:-1], inputs[1:]
    
    output = gpt(x, targets)

    loss = np.mean(-np.log(output[y]))

    return loss


def train(texts: list[list[str]], params):
    """
    Train the model.  This is the expensive bit, big companies pull from big data
    """
    for text in texts:
        inputs = tokenizer.encode(text)
        loss = lm_loss(inputs, params)
        gradients = compute_gradients_via_back_propagation(loss)
        params = gradient_descent(params, gradients)
    return params


# ______________________________MAIN______________________________


def main(prompt:str, n_tokens_to_generate: int=40, model_size: str="124M", models_dir: str="models"):   
    encoder, h_params, params = load_encoder_hparams_and_params(model_size)

    input_tokens = encoder.encode(prompt)

    assert len(input_tokens) + n_tokens_to_generate < h_params["n_ctx"], "Cannot generate more tokens than the model allows."

    output_tokens = generate_response(input_tokens, params, h_params["n_head"], n_tokens_to_generate)
    output_text =  encoder.decode(output_tokens)

    print(output_text)

if __name__ == "__main__":
    fire.Fire(main)

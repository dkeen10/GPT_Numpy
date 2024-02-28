# https://jaykmody.com/blog/gpt-from-scratch/

import numpy as np
from tqdm import tqdm
from transformers import GPT2Tokenizer
from utils import load_encoder_hparams_and_params


# ______________________________MODEL______________________________


def gpt2(inputs, wte, wpe, blocks, ln_f, n_head):
    """
    GPT-2 model.
    """
    x = inputs
    for block in blocks:
        x = block(x, wte, wpe, ln_f, n_head)
    return x


def generate_response(inputs, params, n_head, n_tokens_to_generate=100):
    """
    Generate a response from the GPT model.

    :param inputs: A list of integers representing the input tokens.
    :param n_tokens: The number of tokens to generate.
    :return: A list of integers representing the output tokens.
    """
    for i in range(n_tokens_to_generate):
        output = gpt(inputs)
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


def main():
    inputs = [1, 2, 3]
    response = generate_response(inputs)
    print(response)


if __name__ == "__main__":
    main()

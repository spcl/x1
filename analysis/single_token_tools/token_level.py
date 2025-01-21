# Copyright (c) 2025 ETH Zurich.
#                    All rights reserved.
#
# Use of this source code is governed by a BSD-style license that can be
# found in the LICENSE file.
#
# main author: Julia Barth

import torch
import matplotlib.pyplot as plt
from matplotlib import colors
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    PreTrainedTokenizerFast,
)
import json
import torch.nn.functional as F
from typing import Any


def generate_with_top_k_tracking(
    model: Any,
    tokenizer: PreTrainedTokenizer | PreTrainedTokenizerFast,
    prompt: str,
    max_length: int = 50,
    k: int = 3,
) -> tuple[list[dict], str]:
    """
    Generate the model output with at most 'max_length' tokens, while tracking the k tokens with
    the highest probability for each position.

    format of generated_token_info:
    [
        {
            "text": str,
            "logit": float,
            "probability": float,
            "top_k: [
                {
                    "text": str,
                    "logit": float,
                    "probability": float,
                },
                ...
            ]
        },
        ...
    ]

    Args:
        model (Any): Model to be queried.
        tokenizer (PreTrainedTokenizer | PreTrainedTokenizerFast): Tokenizer to be used.
        prompt (str): Input prompt.
        max_length (int): Maximum generation length. Defaults to 50.
        k (int): Number of tokens to track for each position. Defaults to 3.
    Returns:
        Tuple[list[dict], str]: Tuple of the generated token information and the generated texts.
    """
    # Tokenize the initial prompt
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids
    generated_ids = input_ids.clone()  # Start generation with the prompt

    generated_tokens_info = []  # List to hold info
    generated_text = ""

    for _ in range(max_length):
        # Get logits for the current generated sequence
        outputs = model(input_ids=generated_ids)
        logits = outputs.logits[
            :, -1, :
        ]  # Only get the logits for the last generated token

        # Calculate probabilities
        token_probs = F.softmax(logits, dim=-1).squeeze()

        # Greedy sampling: select the token with the highest probability
        chosen_token_id = torch.argmax(token_probs).item()
        chosen_token_prob = token_probs[chosen_token_id].item()
        chosen_token_logit = logits[0, chosen_token_id].item()
        chosen_token_text = tokenizer.decode([chosen_token_id])

        generated_text += chosen_token_text

        # Store chosen token information
        token_info = {
            "text": chosen_token_text,
            "logit": chosen_token_logit,
            "probability": chosen_token_prob,
        }

        # Get the top-k tokens
        top_k_probs, top_k_indices = torch.topk(token_probs, k)

        # Store top-k information for this position
        top_k_info = []
        for j in range(1, k):
            top_k_info.append(
                {
                    "text": tokenizer.decode(top_k_indices[j]),
                    "logit": logits[0, top_k_indices[j]].item(),
                    "probability": top_k_probs[j].item(),
                }
            )

        # Add the top-k info to the chosen token info for this position
        token_info["top_k"] = top_k_info
        generated_tokens_info.append(token_info)

        # Append chosen token ID to the generated sequence
        generated_ids = torch.cat(
            [generated_ids, torch.tensor([[chosen_token_id]])], dim=1
        )

        # Stop if the end-of-sequence token is generated
        if chosen_token_id == tokenizer.eos_token_id:
            break

    return generated_tokens_info, generated_text


def format_token(token_info: dict, threshold: float = 0.7) -> str:
    """
    Helper function to format tokens based on probability.
    If the token probability is below the threshold, display its probability as well as the
    alternative tokens and their probabilities.

    Args:
        token_info (dict): Token information.
        threshold (float): Tokens below the threshold are marked as red. Defaults to 0.7.
    Returns:
        str: Formatted HTML text.
    """
    prob = token_info["probability"]
    token_text = token_info["text"]
    top_k_info = token_info["top_k"]

    # If the token probability is below the threshold, make it red with extra information
    if prob < threshold:
        top_k_str = ", ".join(
            [f"{alt['text']}: {alt['probability']:.3f}" for alt in top_k_info]
        )
        formatted_text = f"<span style='color: red'>{token_text}</span> <span style='color: yellow'>[{prob:.3f}, Top {len(top_k_info)}: ({top_k_str})]</span>"
    else:
        formatted_text = token_text

    return formatted_text


# from IPython.display import display, HTML
def html_with_low_prob_highlighting(
    tokens_info: list[dict], output_file: str, threshold: float = 0.7
) -> None:
    """
    Print the token sequence and highlight tokens with low probability in red.
    For low probability tokens additional information is displayed.

    Args:
        tokens_info (list[dict]): Token sequence.
        output_file(str): Path to the file, where the html is stored.
        threshold (float): Tokens below the threshold are marked as red. Defaults to 0.7.
    """
    text_html = ""
    for token_info in tokens_info:
        text_html += format_token(token_info, threshold=threshold) + " "

    output = f"<p style='font-family: monospace'>{text_html}</p>"

    # display(HTML(f"<p style='font-family: monospace'>{text_html}</p>"))
    with open(output_file, "w") as text_file:
        text_file.write(output)


def print_full_info(full_info: dict) -> None:
    """
    Print the chosen token info and top-k tokens for each generated position.

    Args:
        full_into (dict): Prompt and top-k tokens for each position.
    """
    print(
        f"Generated Text Sequence:\nInput: {full_info['prompt']}\nOutput: {full_info['generated_text']}\n\n"
    )
    for i, token_info in enumerate(full_info["tokens_info"]):
        print(f"Position {i + 1}:")
        print(
            f"  Chosen Token: '{token_info['text']}' (Logit: {token_info['logit']}, Probability: {token_info['probability']})"
        )
        print("  Top-3 Alternatives:")
        for alt in token_info["top_k"]:
            print(
                f"    - Token: '{alt['text']}' (Logit: {alt['logit']}, Probability: {alt['probability']})"
            )
        print("\n")


def print_highlighted_sequence(tokens_info: list[dict], position: int) -> None:
    """
    Helper function to print a sequence and highlight a certain position.

    Args:
        tokens_info (list[dict]): Token sequence to print.
        position (int): Sequence position to highlight.
    """
    assert position < len(
        tokens_info
    ), f"Position {position} is out of range for the generated sequence."

    sequence_text = ""
    for i, token_info in enumerate(tokens_info):
        token_text = token_info["text"]
        if i == position:
            # Highlight the token at the specified position
            sequence_text += f"\033[91m{token_text}\033[0m"  # Red text for highlight
        else:
            sequence_text += token_text
    print(sequence_text)


def plot_top_k_probabilities(
    tokens_info: list[dict], position: int, output_file: str
) -> None:
    """
    Plot the top k probabilities at a specific position.

    Args:
        tokens_info (list[dict]): Token sequence.
        position (int): Sequence position to plot.
        output_file(str): Path to the file, where the plot is stored.
    """
    assert position < len(
        tokens_info
    ), f"Position {position} is out of range for the generated sequence."

    # Handle chosen token differently
    tokens = [tokens_info[position]["text"]]
    probabilities = [tokens_info[position]["probability"]]
    # Remaining top k tokens
    top_k_info = tokens_info[position]["top_k"]
    tokens += [info["text"] for info in top_k_info]
    probabilities += [info["probability"] for info in top_k_info]

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.barh(tokens, probabilities, color="skyblue")
    plt.xlabel("Probability")
    plt.ylabel("Token")
    plt.title(f"Top-{len(top_k_info)+1} Token Probabilities at Position {position}")
    plt.gca().invert_yaxis()  # Reverse the order for readability
    # plt.show()
    plt.savefig(output_file)


if __name__ == "__main__":
    model_path = "<Set path to model.>"

    model = AutoModelForCausalLM.from_pretrained(  # AutoModelForCausalLM for step-by-step
        model_path,
        low_cpu_mem_usage=True,
        torch_dtype=torch.float16,
        device_map="auto",  # 'cuda'
        # Temperature
    )
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    prompt = 'How many "s"\'s are in strassburg?'

    generated_tokens_info, generated_text = generate_with_top_k_tracking(
        model, tokenizer, prompt, max_length=20, k=3
    )

    full_info = {
        "prompt": prompt,
        "generated_text": generated_text,
        "tokens_info": generated_tokens_info,
    }

    with open("full_info.json", "w") as f:
        json.dump(full_info, f, indent=4)

    # Play around with an earlier generated sample.
    # with open("example.json", "r") as f:
    #     full_info = json.load(f)

    position_to_inspect = 1

    # Print prompt, generated output and the top k for each position
    print_full_info(full_info)

    # Highlight a token at a given position
    print_highlighted_sequence(full_info["tokens_info"], position_to_inspect)

    # Generate a html formatted text, where low probabilty tokens are marked in red instead of black and
    # additionally print the other top k tokens and their probability.
    html_with_low_prob_highlighting(full_info["tokens_info"], "token_probs.html")

    # # Visualize the top k tokens and their probabilities for a given position
    plot_top_k_probabilities(
        full_info["tokens_info"],
        position_to_inspect,
        f"topk_at_{position_to_inspect}.pdf",
    )

from dotenv import load_dotenv

load_dotenv()

import torch
from rich import print
from rich.text import Text
from transformers import AutoModelForCausalLM, AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

# load custom jinja template and set chat template
with open("./recipes/doctrio/sft/qwen3.jinja", "r") as f:
    template = f.read()
tokenizer.chat_template = template

conversation = [
    {"role": "system", "content": "You are a friendly assistant"},
    {"role": "user", "content": "Hello assistant"},
    {"role": "assistant", "content": "Hello user"},
    {"role": "user", "content": "How are you?"},
    {"role": "assistant", "content": "Fine thanks, and you?"},
    {"role": "user", "content": "Great, thanks"},
]
input_text = tokenizer.apply_chat_template(conversation, add_generation_prompt=False, tokenize=False)
tokenized_output = tokenizer.apply_chat_template(
    conversation,
    return_assistant_tokens_mask=True,
    return_dict=True,
)
print("Tokenized Output with Assistant Mask:")
print(tokenized_output)

# Visualize using rich
text_visualization = Text()
tokens = tokenizer.convert_ids_to_tokens(tokenized_output.input_ids)
for token, mask in zip(tokens, tokenized_output.assistant_masks):
    color = "cyan" if mask else "white"
    text_visualization.append(token.replace('Ġ', ' ').replace("Ċ", "\n"), style=color)

print(text_visualization)
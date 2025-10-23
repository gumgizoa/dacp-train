import os
from dotenv import load_dotenv

load_dotenv()

from datasets import load_dataset
import re
import ast
import json
from typing import List, Dict, Any

def _safe_parse_struct(s: str) -> Any:
    """
    Try to parse a string representing a dict/list:
    1) ast.literal_eval (Python literal)
    2) json.loads (JSON)
    3) fallback: replace single quotes with double quotes and try json.loads
    If all fail, return {'raw': s}
    """
    s_clean = s.strip()
    if not s_clean:
        return None

    try:
        return ast.literal_eval(s_clean)
    except Exception:
        pass

    try:
        return json.loads(s_clean)
    except Exception:
        pass

    try:
        guess = s_clean.replace("'", '"')
        return json.loads(guess)
    except Exception:
        return {"raw": s_clean}


def parse_assistant_content_with_thinking(text: str) -> List[Dict[str, Any]]:
    """
    Parse an assistant message `text` into a list of message dicts preserving:
      - normal public text chunks -> {'role':'assistant', 'content': '...'}
      - thinking blocks -> {'role':'assistant', 'content': '', 'reasoning_content': '...'}
      - tool calls -> {'role':'assistant', 'content': '', 'tool_call': <parsed dict>}
    Order is preserved.
    """
    parts: List[Dict[str, Any]] = []
    pattern = re.compile(r"(<think>.*?</think>|<tool_call>.*?</tool_call>)", flags=re.DOTALL)
    last = 0
    for m in pattern.finditer(text):
        between = text[last:m.start()].strip()
        if between:
            parts.append({"role": "assistant", "content": between})

        token = m.group(1)
        if token.startswith("<think>"):
            # extract inner content (without <think> tags)
            inner_match = re.match(r"<think>\s*(.*?)\s*</think>", token, flags=re.DOTALL)
            inner_content = inner_match.group(1).strip() if inner_match else ""
            parts.append({
                "role": "assistant",
                "content": "",
                "reasoning_content": inner_content
            })
        else:  # token starts with <tool_call>
            inner_match = re.match(r"<tool_call>\s*(.*?)\s*</tool_call>", token, flags=re.DOTALL)
            inner = inner_match.group(1) if inner_match else token
            parsed = _safe_parse_struct(inner)
            parts.append({
                "role": "assistant",
                "content": "",
                "tool_calls": [parsed]
            })

        last = m.end()

    tail = text[last:].strip()
    if tail:
        parts.append({"role": "assistant", "content": tail})

    return parts


def extract_tool_responses_from_tool_message(text: str) -> List[Dict[str, Any]]:
    """
    If the tool message contains <tool_response>...</tool_response> blocks,
    extract them into [{'role':'tool', 'content': <inner>}, ...].
    If no such blocks exist, return the original tool message as-is in a single-item list.
    """
    matches = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, flags=re.DOTALL)
    if not matches:
        return [{"role": "tool", "content": text}]
    results = []
    for m in matches:
        inner = m.strip()
        results.append({"role": "tool", "content": inner})
    return results


def format_messages_with_thinking(example: dict) -> dict:
    """
    Convert example['messages'] into a list where:
     - user messages are kept as-is
     - assistant messages are split into public text, reasoning traces, and tool_call entries (order preserved)
     - tool messages either yield extracted <tool_response> blocks or are kept whole
    Also attempts to parse any xml_tools in chat_template_kwargs.
    """
    chat_template_kwargs = example.get("chat_template_kwargs") or {}
    
    # handle xml_tools
    if "xml_tools" in chat_template_kwargs:
        xml_tools = chat_template_kwargs.pop("xml_tools")
        if isinstance(xml_tools, list) and xml_tools:
            xml_tools_str = xml_tools[0]
            tools = []
            for tool_str in xml_tools_str.splitlines():
                try:
                    tool = ast.literal_eval(tool_str.strip())
                    xml_tools_type = "python_dict"
                except Exception:
                    tool = json.loads(tool_str.strip())
                    xml_tools_type = "json"
                tools += [tool]
        else:
            xml_tools_type = "others"
            tools = []
    else:
        xml_tools_type = "others"
        tools = []
        
    example["tools"] = tools
    example["xml_tools_type"] = xml_tools_type
    chat_template_kwargs = {"enable_thinking": chat_template_kwargs["enable_thinking"]}
    example["chat_template_kwargs"] = chat_template_kwargs

    out_messages: List[Dict[str, Any]] = []
    for message in example.get("messages", []):
        role = message.get("role")
        content = message.get("content", "")

        if role == "assistant":
            parsed_parts = parse_assistant_content_with_thinking(content)
            out_messages.extend(parsed_parts)

        elif role == "tool":
            tool_msgs = extract_tool_responses_from_tool_message(content)
            out_messages.extend(tool_msgs)

        else:
            out_messages.append(message)

    example["messages"] = out_messages
    return example

from tqdm import tqdm

ds = load_dataset("HuggingFaceTB/smoltalk2", name="SFT", split="smolagents_toolcalling_traces_think")
examples = []
for example in tqdm(ds):
    examples += [format_messages_with_thinking(example)]
    
    
hallucinated_examples = []
final_examples = []
for i, data in enumerate(examples):
    tool_spec = {}
    for tool in data["tools"]:
        fn = tool["function"]
        name = fn["name"]
        allowed_args = set(fn["parameters"]["properties"].keys())
        tool_spec[name] = allowed_args

    for msg in data["messages"]:
        tool_call = msg.get("tool_calls")
        if not tool_call:
            continue
        tool_call = tool_call[0]
        name = tool_call.get("name")
        args = tool_call.get("arguments")
        if not isinstance(args, dict):
            hallucinated_examples += [data]
            break

        if name not in tool_spec:
            hallucinated_examples += [data]
            break

        allowed_keys = tool_spec[name]
        if not set(args.keys()).issubset(allowed_keys):
            hallucinated_examples += [data]
            break
    else:
        data["messages"] = json.dumps(data["messages"])
        data["tools"] = json.dumps(data["tools"])
        final_examples += [data]
        
        
from datasets import Dataset

ex_ds = Dataset.from_list(final_examples)
ex_ds.push_to_hub("eungizoa/agentic-collection", config_name="SFT", split="smolagents_toolcalling_traces")


import json
from typing import List, Dict, Any

def compose_messages_with_thinking(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Recompose parsed messages into the serialized conversation format:
    
    [
      {"role": "user", "content": "~~"},
      {"role": "assistant", "content": "<think>...</think>\n<tool_call>...</tool_call>"},
      {"role": "tool", "content": "~~"},
      {"role": "assistant", "content": "~~"},
      ...
    ]

    Rules:
      - Assistant messages may contain:
          * reasoning_content  → wrapped in <think>...</think>
          * tool_call          → wrapped in <tool_call>...</tool_call>
          * content            → plain text (public output)
      - Consecutive assistant messages (reasoning/tool/content) are merged into one.
      - Tool and user messages remain as-is.
    """
    composed: List[Dict[str, Any]] = []
    buffer: Dict[str, Any] = None  # holds partial assistant message

    def flush_buffer():
        """If an assistant message is being built, finalize and append it."""
        nonlocal buffer
        if buffer is not None:
            # build content string
            content_parts = []
            if "reasoning_content" in buffer and buffer["reasoning_content"]:
                content_parts.append(f"<think>\n{buffer['reasoning_content'].strip()}\n</think>")
            if "tool_calls" in buffer and buffer["tool_calls"]:
                # Serialize tool_call (dict → pretty string)
                if isinstance(buffer["tool_calls"], (dict, list)):
                    tool_str = json.dumps(buffer["tool_calls"], ensure_ascii=False)
                else:
                    tool_str = str(buffer["tool_calls"])
                content_parts.append(f"<tool_call>\n{tool_str}\n</tool_call>")
            if "content" in buffer and buffer["content"]:
                content_parts.append(buffer["content"].strip())

            final_content = "\n".join(content_parts).strip()
            composed.append({"role": "assistant", "content": final_content})
            buffer = None

    for msg in messages:
        role = msg.get("role")
        if role == "assistant":
            # If buffer is empty, start a new assistant message
            if buffer is None:
                buffer = {"role": "assistant"}

            # Merge all relevant fields
            if "reasoning_content" in msg:
                buffer["reasoning_content"] = (
                    buffer.get("reasoning_content", "") + "\n" + msg["reasoning_content"]
                ).strip()
            elif "tool_calls" in msg:
                buffer["tool_calls"] = msg["tool_calls"]
            elif "content" in msg:
                # normal assistant content
                buffer["content"] = (
                    buffer.get("content", "") + "\n" + msg["content"]
                ).strip()
        else:
            # Before switching to another role, flush any assistant buffer
            flush_buffer()
            composed.append(msg)

    # flush last assistant block
    flush_buffer()
    return composed


from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-14B")

print(tokenizer.apply_chat_template(
    compose_messages_with_thinking(examples[0]["messages"]),
    tools=examples[0]["tools"],
    tokenize=False
))


print("=====================================================")
from datasets import load_dataset

ds = load_dataset("HuggingFaceTB/smoltalk2", name="SFT", split="hermes_function_calling_v1_no_think")

import re
import ast
import json

def strip_tool_tags(text: str) -> str:
    """
    Remove only <tool>, </tool>, <tool_response>, and </tool_response> tags
    but keep the inner content.
    """
    # Replace opening/closing tags with nothing
    text = re.sub(r"</?tools>", "", text)
    return text.strip()

def extract_tool_calls(text: str):
    """
    Extract all <tool_response>...</tool_response> blocks
    and convert them into [{"role": "tool", "content": "..."}].
    """
    matches = re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, flags=re.DOTALL)
    tool_calls = []
    for m in matches:
        # Keep raw string content (trim whitespace/newlines)
        cleaned = m.strip().replace('\\n', '')
        tool_call = ast.literal_eval(cleaned)
        tool_calls += [tool_call]
    
    return {"role": "assistant", "content": "", "tool_calls": tool_calls}

def extract_tool_responses(text: str) -> list[dict]:
    """
    Extract all <tool_response>...</tool_response> blocks
    and convert them into [{"role": "tool", "content": "..."}].
    """
    matches = re.findall(r"<tool_response>\s*(.*?)\s*</tool_response>", text, flags=re.DOTALL)
    results = []
    for m in matches:
        # Keep raw string content (trim whitespace/newlines)
        cleaned = m.strip()
        results.append({
            "role": "tool",
            "content": cleaned
        })
    return results

def format_messages(example: dict):
    # Format chat_template_kwargs
    chat_template_kwargs = example.get("chat_template_kwargs") or {}
    if "xml_tools" in chat_template_kwargs:
        # Smoltalk2 was created specifically for training SmolLM and is aligned with its chat_template_kwargs format.
        # To train models other than SmolLM, we need to convert each example into a format that can directly utilize the datasets available on Hugging Face.
        xml_tools = chat_template_kwargs.pop("xml_tools")
        if type(xml_tools) == list:
            xml_tools = xml_tools[0]
            match = re.search(r"<tools>\s*(\[.*?\])\s*</tools>", xml_tools, flags=re.DOTALL)
            if match:
                xml_tools = match.group(1).strip()
            try:
                # xml_tools in smoltalk2 forms like: "<tools>\n[...]\n</tools>"
                # Change it as list of dictionary (standard tool calling format)
                tools = ast.literal_eval(strip_tool_tags(xml_tools)) 
                # Function "_prepare_dataset" of trl SFTTrainer expects that the `tools` parameter to be explicitly fed into the "apply_chat_template" function.
                # ex. tokenizer.apply_chat_template(..., tools=example.get("tools"), **example["chat_template_kwargs"])
                # Therefore, here to separate "tools" from "chat_template_kwargs".
                xml_tools_type = "python_dict"
                example["tools"] = tools
            except:
                try:
                    tools = [json.loads(line) for line in strip_tool_tags(xml_tools).splitlines()]
                    xml_tools_type = "json"
                    example["tools"] = tools
                except:
                    xml_tools_type = "others"
                    example["tools"] = []
        else:
            xml_tools_type = "others"
            example["tools"] = []
    else:
        xml_tools_type = "others"
        example["tools"] = []
    
    example["xml_tools_type"] = xml_tools_type
    
    # Reformat tool response messages
    messages = []
    for message in example["messages"]:
        if message["role"] == "assistant":
            tool_calls = extract_tool_calls(message["content"])
            if not tool_calls["tool_calls"]:
                messages.append(message)
            else:
                messages.append(tool_calls)
        elif message["role"] == "tool":
            # [{"role": "tool", "content": "..."}, ...]
            tool_responses = extract_tool_responses(message["content"])
            messages.extend(tool_responses)
        else:
            messages.append(message)
    example["messages"] = messages       
    
    # It will be feed into the `apply_chat_template` function (**example["chat_template_kwargs"])
    chat_template_kwargs = {"enable_thinking": chat_template_kwargs["enable_thinking"]}
    example["chat_template_kwargs"] = chat_template_kwargs 
    return example

from tqdm import tqdm
examples = []
for example in tqdm(ds):
    examples += [format_messages(example)]
    
python_dict_examples = [example for example in examples if example["xml_tools_type"] == "python_dict"]

def find_invalid_tool_calls(examples):
    invalid_cases = {}

    for i, example in enumerate(examples):
        tools = example.get("tools", [])
        messages = example.get("messages", [])

        # Build a mapping: tool_name -> allowed argument keys
        tool_specs = {}
        for tool in tools:
            fn = tool.get("function", {})
            name = fn.get("name")
            if not name:
                invalid_reasons = invalid_cases.get(str(i), [])
                invalid_reasons += [f"No name in tools `{fn}`"]
                invalid_cases[str(i)] = invalid_reasons
            props = fn.get("parameters", {}).get("properties", {})
            if name in tool_specs:
                invalid_reasons = invalid_cases.get(str(i), [])
                invalid_reasons += [f"Duplicated function name in tools `{name}`"]
                invalid_cases[str(i)] = invalid_reasons
                
            tool_specs[name] = set(props.keys())

        # Scan messages for assistant tool_calls
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            tool_calls = msg.get("tool_calls")
            if not tool_calls:
                continue
            
            for call in tool_calls:
                name = call.get("name")
                args = call.get("arguments", {})

                # Case 1 — Tool name not defined
                if name not in tool_specs:
                    invalid_reasons = invalid_cases.get(str(i), [])
                    invalid_reasons += [f"Undefined tool '{name}'"]
                    invalid_cases[str(i)] = invalid_reasons
                    continue

                # Case 2 — Arguments not a dict
                if not isinstance(args, dict):
                    invalid_reasons = invalid_cases.get(str(i), [])
                    invalid_reasons += [f"Arguments must be a dict, got {type(args).__name__}"]
                    invalid_cases[str(i)] = invalid_reasons

                # Case 3 — Argument key mismatch
                allowed_keys = tool_specs[name]
                invalid_keys = set(args.keys()) - allowed_keys
                if invalid_keys:
                    invalid_reasons = invalid_cases.get(str(i), [])
                    invalid_reasons += [f"Invalid argument(s) for '{name}': {invalid_keys}"]
                    invalid_cases[str(i)] = invalid_reasons
    return invalid_cases


# 🔧 Example usage
invalid = find_invalid_tool_calls(python_dict_examples)

if invalid:
    print(f"Found {len(invalid)} invalid tool calls:\n")
else:
    print("✅ All tool_calls are valid!")


final_examples = []
for i, example in enumerate(python_dict_examples):
    if str(i) not in invalid.keys():
        example["messages"] = json.dumps(example["messages"])
        example["tools"] = json.dumps(example["tools"])
        final_examples += [example]
        
from datasets import Dataset

ex_ds = Dataset.from_list(final_examples)
ex_ds.push_to_hub("eungizoa/agentic-collection", config_name="SFT", split="hermes_function_calling_v1_no_think")
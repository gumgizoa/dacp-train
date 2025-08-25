from __future__ import annotations

import json
import logging
import re
import os
import aiofiles
from typing import Any, Callable, List, Union

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


class OutputParserException(Exception):
    pass


class JsonRepairError(Exception):
    def init(self, e, text):
        message = "Don't know how to fix '%s', position %s (-->%s<--)" % (e.msg, e.pos, text[e.pos - 5 : e.pos + 5])
        super().init(message)
        self.text = text


def save_json(obj: dict, file_path: str, write_mode:str="w", **kwargs):
    with open(file_path, write_mode) as file:
        json.dump(obj, file, **kwargs)
    
def read_jsonl(file_path, to_pandas: bool = False) -> Union[List[dict], pd.DataFrame]:
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except Exception:
        logger.warning("Attempting to read as JSON Lines format. You may safely ignore this warning if this is expected.")
        with open(file_path, "r") as f:
            data = []
            for line in f.readlines():
                data += [json.loads(line)]
    if to_pandas:
        data = pd.DataFrame(data)
    return data

def save_as_jsonl(data: pd.DataFrame, file_path: str, write_mode:str ="a"):
    with open(file_path, write_mode) as file:
        records = data.to_dict("records")
        for record in records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

def _replace_new_line(match: re.Match[str]) -> str:
    value = match.group(2)
    value = re.sub(r"\n", r"\\n", value)
    value = re.sub(r"\r", r"\\r", value)
    value = re.sub(r"\t", r"\\t", value)
    value = re.sub(r'(?<!\\)"', r"\"", value)

    return match.group(1) + value + match.group(3)


def _custom_parser(multiline_string: str) -> str:
    """The LLM response for `action_input` may be a multiline
    string containing unescaped newlines, tabs or quotes. This function
    replaces those characters with their escaped counterparts.
    (newlines in JSON must be double-escaped: `\\n`).
    """
    if isinstance(multiline_string, (bytes, bytearray)):
        multiline_string = multiline_string.decode()

    multiline_string = re.sub(
        r'("action_input"\:\s*")(.*?)(")',
        _replace_new_line,
        multiline_string,
        flags=re.DOTALL,
    )

    return multiline_string


# Adapted from https://github.com/KillianLucas/open-interpreter/blob/5b6080fae1f8c68938a1e4fa8667e3744084ee21/interpreter/utils/parse_partial_json.py
# MIT License


def parse_partial_json(s: str, *, strict: bool = False) -> Any:
    """Parse a JSON string that may be missing closing braces.

    Args:
        s: The JSON string to parse.
        strict: Whether to use strict parsing. Defaults to False.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    # Attempt to parse the string as-is.
    try:
        return json.loads(s, strict=strict)
    except json.JSONDecodeError:
        pass

    # Initialize variables.
    new_chars = []
    stack = []
    is_inside_string = False
    escaped = False

    # Process each character in the string one at a time.
    for char in s:
        if is_inside_string:
            if char == '"' and not escaped:
                is_inside_string = False
            elif char == "\n" and not escaped:
                char = "\\n"  # Replace the newline character with the escape sequence.
            elif char == "\\":
                escaped = not escaped
            else:
                escaped = False
        else:
            if char == '"':
                is_inside_string = True
                escaped = False
            elif char == "{":
                stack.append("}")
            elif char == "[":
                stack.append("]")
            elif char == "}" or char == "]":
                if stack and stack[-1] == char:
                    stack.pop()
                else:
                    # Mismatched closing character; the input is malformed.
                    return None

        # Append the processed character to the new string.
        new_chars.append(char)

    # If we're still inside a string at the end of processing,
    # we need to close the string.
    if is_inside_string:
        new_chars.append('"')

    # Reverse the stack to get the closing characters.
    stack.reverse()

    # Try to parse mods of string until we succeed or run out of characters.
    while new_chars:
        # Close any remaining open structures in the reverse
        # order that they were opened.
        # Attempt to parse the modified string as JSON.
        try:
            return json.loads("".join(new_chars + stack), strict=strict)
        except json.JSONDecodeError:
            # If we still can't parse the string as JSON,
            # try removing the last character
            new_chars.pop()

    # If we got here, we ran out of characters to remove
    # and still couldn't parse the string as JSON, so return the parse error
    # for the original string.
    return json.loads(s, strict=strict)


_json_markdown_re = re.compile(r"```(json)?(.*)", re.DOTALL)


def _parse_json_markdown(json_string: str, *, parser: Callable[[str], Any] = parse_partial_json) -> dict:
    """Parse a JSON string from a Markdown string.

    Args:
        json_string: The Markdown string.

    Returns:
        The parsed JSON object as a Python dictionary.
    """
    try:
        return _parse_json(json_string, parser=parser)
    except json.JSONDecodeError:
        # Try to find JSON string within triple backticks
        match = _json_markdown_re.search(json_string)

        # If no match found, assume the entire string is a JSON string
        # Else, use the content within the backticks
        json_str = json_string if match is None else match.group(2)
    return _parse_json(json_str, parser=parser)


_json_strip_chars = " \n\r\t`"


def _parse_json(json_str: str, *, parser: Callable[[str], Any] = parse_partial_json) -> dict:
    # Strip whitespace,newlines,backtick from the start and end
    json_str = json_str.strip(_json_strip_chars)

    # handle newlines and other special characters inside the returned value
    json_str = _custom_parser(json_str)

    # Parse the JSON string into a Python dictionary
    return parser(json_str)


def _strip_trailing_json_commas(json_str: str) -> str:
    """Find and remove trailing comma in json string which would raise JSONDecoding Error.
    Example:
      {
          "title": "title",
          "content": "content", <- TRAILING_JSON_COMMA
      }
    """
    find_trailing_comma = r"(,)(?!\s*?[\{\[\"'\w])"
    return re.sub(find_trailing_comma, "", json_str)


def _repair_json(text):
    while True:
        try:
            return json.loads(text)
        except json.decoder.JSONDecodeError as e:
            if e.msg == "Expecting ',' delimiter":
                if text[e.pos - 1] == '"':
                    text = text[: e.pos - 1] + "\\" + text[e.pos - 1 :]
                    continue
                elif text[e.pos - 2] == '"':
                    text = text[: e.pos - 2] + "\\" + text[e.pos - 2 :]
                    continue
            elif e.msg == "Invalid control character at":
                if text[e.pos] == "\n":
                    text = text[: e.pos] + "\\n" + text[e.pos + 1 :]
                    continue
            raise JsonRepairError(e, text) from None


def parse_json_markdown(json_str: str, *, parser: Callable[[str], Any] = parse_partial_json) -> dict:
    """Parse json markdown, especially generated from llm."""    
    def _inner_function(json_str, parser):
        try:
            parsed_json = _parse_json_markdown(json_str, parser=parser)
        except Exception:
            json_regex = r"```json\n(.*?)\n```"
            match = re.search(json_regex, json_str, re.DOTALL)
            parsed_json = _repair_json(match.group(1))
        return parsed_json
    
    json_str = _strip_trailing_json_commas(json_str)
    try:
        parsed_json = _inner_function(json_str, parser)
    except Exception as e:
        json_str = json_str.replace("\\", "\\\\")
        parsed_json = _inner_function(json_str, parser)
    return parsed_json


def make_serializable(obj):
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(i) for i in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj
    
def make_json_serializable(value):
    if isinstance(value, dict):
        # If the value is a dictionary, we need to go through each key-value pair recursively
        return {k: make_json_serializable(v) for k, v in value.items()}
    elif isinstance(value, list):
        # If the value is a list, we need to process each element recursively
        return [make_json_serializable(item) for item in value]
    else:
        # Try to serialize the value directly, and if it fails, convert it to a string
        try:
            json.dumps(value)
            return value
        except (TypeError, ValueError):
            return str(value)

def write_jsonl(file_path: str, record: dict):
    record_dict = make_serializable(record)
    with open(file_path, "a") as file:
        file.write(json.dumps(record_dict, ensure_ascii=False) + "\n")

async def awrite_jsonl(file_path: str, record: dict):
    record_dict = make_serializable(record)
    async with aiofiles.open(file_path, "a") as file:
        await file.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
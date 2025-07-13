import contextlib
from typing import Union

import torch
from IPython.display import display
from PIL import Image


@contextlib.contextmanager
def set_default_tensor_type(dtype: torch.dtype):
    """Sets the default torch dtype to the given dtype."""
    torch.set_default_dtype(dtype)
    yield
    torch.set_default_dtype(torch.float)


def apply_chat_template(prompt_parts: list[Union[str, Image.Image]]) -> str:
    formatted_prompt = ["<start_of_turn>user\n"]
    for part in prompt_parts:
        if isinstance(part, str):
            formatted_prompt.append(part)
        elif isinstance(part, Image.Image):
            formatted_prompt.append("<start_of_image><end_of_image>")
    formatted_prompt.append("<end_of_turn>\n<start_of_turn>model")
    return "\n".join(formatted_prompt)


def format_prompt(prompt_parts: list[Union[str, Image.Image]]) -> list[str | Image.Image]:
    formatted_prompt = ["<start_of_turn>user\n"]
    formatted_prompt.extend(prompt_parts)
    formatted_prompt.append("<end_of_turn>\n<start_of_turn>model")
    return formatted_prompt


def truncate_model_output(text: str, clip_start: bool = False, clip_end: bool = True) -> str:
    if not (clip_start or clip_end):
        return text

    model_turn_marker: str = "<end_of_turn>\n<start_of_turn>model"
    try:
        start_index = text.index(model_turn_marker)
        if clip_end:
            search_start = start_index + len(model_turn_marker)
            end_index = text.index("<end_of_turn>", search_start)
        else:
            end_index = -1

        if not clip_start:
            search_start = 0
        return text[search_start:end_index]
    except ValueError:
        return text


def display_prompt_and_result(prompt: list[str | Image.Image], result: str, width: int = 80):
    print("=" * width)
    print("INPUT")
    for part in prompt:
        if isinstance(part, str):
            print(part, end="")
        elif isinstance(part, Image.Image):
            img_copy = part.copy()
            img_copy.thumbnail((256, 256))
            display(img_copy)
    print("\n", "-" * width)
    print("GENERATED")
    print(truncate_model_output(result))
    print("=" * width)

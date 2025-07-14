import logging
from pathlib import Path

import pytest
import torch
from huggingface_hub import snapshot_download
from PIL import Image

from lxmls.multimodal.gemma3 import config, utils
from lxmls.multimodal.gemma3 import model as gemma3_model

logger = logging.getLogger(__name__)
formatter = logging.Formatter("%(asctime)s - %(name)24s - %(levelname)7s - %(message)s")
sh = logging.StreamHandler()
sh.setFormatter(formatter)
logger.addHandler(sh)
logger.setLevel(logging.DEBUG)


def test_set_default_tensor_type():
    with utils.set_default_tensor_type(torch.float16):
        assert torch.get_default_dtype() == torch.float16
    # After context manager, default dtype is reset to float (torch.float32)
    assert torch.get_default_dtype() == torch.float


def test_format_prompt():
    prompt_parts = ["Hello, world!\n", "How are you?\n"]
    expected = ["<start_of_turn>user\n"] + prompt_parts + ["<end_of_turn>\n<start_of_turn>model"]
    formatted = utils.format_prompt(prompt_parts)
    assert formatted == expected


def test_truncate_model_output_clip_end():
    # Create a string with the model turn marker and another <end_of_turn> later
    marker = "<end_of_turn>\n<start_of_turn>model"
    text = "Keep this part " + marker + "Remove this part<end_of_turn> and more"
    # If clip_end is True, the output should end at the first <end_of_turn> after marker
    truncated = utils.truncate_model_output(text, clip_start=False, clip_end=True)
    # Expecting text before the second marker to be removed
    assert marker in text
    assert "<end_of_turn>" not in truncated[len(marker) :]


def test_truncate_model_output_no_clip():
    text = "No clipping here."
    # Without clipping, should return original text
    assert utils.truncate_model_output(text, clip_start=False, clip_end=False) == text


def test_display_prompt_and_result(capsys):
    # Create a prompt list with both text and an image.
    prompt = [
        "Test prompt.\n",
        Image.new("RGB", (100, 100), color="red"),
    ]
    prompt = utils.format_prompt(prompt)
    result: str = "{utils.apply_chat_template(prompt)}Generated text here."

    # Call the function (which prints to stdout)
    utils.display_prompt_and_result(prompt, result)

    captured = capsys.readouterr().out
    # Check that output contains some expected keywords.
    assert "INPUT" in captured
    assert "GENERATED" in captured
    # Check that truncated output does not include the extra text after the marker.
    assert "Extra text" not in captured


@pytest.fixture(scope="module")
def model_data_path():
    """Fixture to provide the path to the model data."""
    path = Path(__file__).parent.parent / "data" / "vlm" / "gemma3"
    assert path.exists(), f"Model directory not found at {path}."
    assert (path / "model.ckpt").exists(), f"Checkpoint not found at {path / 'model.ckpt'}."
    assert (path / "tokenizer.model").exists(), f"Tokenizer not found at {path / 'tokenizer.model'}."
    return path


@pytest.fixture(scope="module")
def image_data_path():
    """Fixture to provide the path to the images."""
    path = Path(__file__).parent.parent / "data" / "vlm" / "images"
    assert path.exists(), f"Image directory not found at {path}."
    return path


@pytest.fixture(scope="module")
def gemma3_model_instance(model_data_path):
    """Fixture to initialize and load the Gemma3 model."""

    logger.debug("Downloading model data if not present")
    _ = snapshot_download("rshwndsz/gemma-3-4b-it-ckpt", local_dir=str(model_data_path))
    model_config = config.get_model_config()
    model_config.dtype = "float32"
    model_config.quant = False
    model_config.tokenizer = str(model_data_path / "tokenizer.model")

    ckpt_path = Path(model_data_path) / "model.ckpt"
    logger.debug(f"Loading model from {ckpt_path}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with utils.set_default_tensor_type(model_config.get_dtype()):
        model = gemma3_model.Gemma3ForMultimodalLM(model_config)
        model.load_state_dict(torch.load(ckpt_path)["model_state_dict"])
        model = model.to(device).eval()
    logger.debug("Model loading done")

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        peak_memory = torch.cuda.max_memory_allocated(device)
    else:
        peak_memory = -1

    return model, device, peak_memory


@pytest.mark.gpu
def test_text_generation(gemma3_model_instance):
    """Tests text-only generation."""
    model, device, peak_memory = gemma3_model_instance
    if peak_memory != -1:
        print(f"Peak memory usage after model load: {peak_memory / 1e9:.2f} GB")
    prompt = [
        ["<start_of_turn>user The capital of Italy is?<end_of_turn>\n<start_of_turn>model"],
        ["<start_of_turn>user What is your purpose?<end_of_turn>\n<start_of_turn>model"],
    ]
    results = model.generate(prompt, device, output_len=20)
    assert isinstance(results, list)
    assert len(results) == 2
    assert all(isinstance(res, str) and len(res) > 0 for res in results)
    assert "rome" in results[0].lower()


@pytest.mark.gpu
def test_single_image_generation(gemma3_model_instance, image_data_path):
    """Tests generation with a single image."""
    torch.cuda.empty_cache()
    model, device, peak_memory = gemma3_model_instance
    if peak_memory != -1:
        print(f"Peak memory usage after model load: {peak_memory / 1e9:.2f} GB")
    prompt = [
        [
            "<start_of_turn>user\n",
            Image.open(image_data_path / "cow_in_beach.jpg"),
            "The name of the animal in the image is?",
            "<end_of_turn>\n<start_of_turn>model",
        ]
    ]
    result = model.generate(prompt, device, output_len=20)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str) and len(result[0]) > 0
    assert "cow" in result[0].lower()


@pytest.mark.gpu
def test_interleaved_image_generation(gemma3_model_instance, image_data_path):
    """Tests generation with multiple interleaved images."""
    torch.cuda.empty_cache()
    model, device, peak_memory = gemma3_model_instance
    if peak_memory != -1:
        print(f"Peak memory usage after model load: {peak_memory / 1e9:.2f} GB")
    prompt = [
        [
            "<start_of_turn>user\n",
            "This image",
            Image.open(image_data_path / "lilly.jpg"),
            "and this image",
            Image.open(image_data_path / "sunflower.jpg"),
            "are similar because? Give me the main reason.",
            "<end_of_turn>\n<start_of_turn>model",
        ]
    ]
    result = model.generate(prompt, device, output_len=120)
    assert isinstance(result, list)
    assert len(result) == 1
    assert isinstance(result[0], str) and len(result[0]) > 0
    assert "flower" in result[0].lower()

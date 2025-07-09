from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig

model_id = "google/gemma-3-4b-it-qat-q4_0-unquantized"

calibration_dataset = load_dataset(
    "allenai/c4",
    data_files="en/c4-train.00001-of-01024.json.gz",
    split="train",
).select(range(1024))["text"]

quant_config = QuantizeConfig(
    bits=4,
    group_size=128,
    v2=True,
    dynamic={
        r"+.*\language_model.*\.mlp.*": {"bits": 4, "group_size": 128},
        r"+.*\language_model.*\.self_attn.*": {"bits": 4, "group_size": 128},
    },
)

model = GPTQModel.load(model_id, quant_config)
model.quantize(calibration_dataset, batch_size=8, auto_gc=False)
model.save("gemma-3-4b-it-qat-q4_0")

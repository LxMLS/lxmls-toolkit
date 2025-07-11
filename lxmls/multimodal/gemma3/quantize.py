from argparse import ArgumentParser

from datasets import load_dataset
from gptqmodel import GPTQModel, QuantizeConfig


def main(n: int, bs: int, v2: bool = False):
    if v2:
        print("Using GPTQ v2: Could take 2x VRAM")

    model_id = "google/gemma-3-4b-it-qat-q4_0-unquantized"

    calibration_dataset = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(n))["text"]

    quant_config = QuantizeConfig(
        bits=4,
        group_size=128,
        v2=v2,
        dynamic={
            r".*language_model.*\.mlp.*": {"bits": 4, "group_size": 128},
            r".*language_model.*\.self_attn.*": {"bits": 4, "group_size": 128},
        },
    )

    model = GPTQModel.load(model_id, quant_config)
    model.quantize(calibration_dataset, batch_size=bs, auto_gc=True, buffered_fwd=False)
    model.save("gemma-3-4b-it-qat-4_128")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--n", type=int, default=1024)
    parser.add_argument("--bs", type=int, default=4)
    parser.add_argument("--v2", action="store_true")
    args = parser.parse_args()

    main(args.n, args.bs, args.v2)

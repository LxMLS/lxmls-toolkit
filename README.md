![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Frshwndsz%2Flxmls-toolkit%2Fmaster%2Fpyproject.toml)

# LxMLS 2025

Machine learning toolkit for natural language processing. Written for [LxMLS - Lisbon Machine Learning Summer School](http://lxmls.it.pt)

* Scientific Python and Mathematical background
* Linear Classifiers (Gradient Descent)
* Feed-forward models in deep learning (Backpropagation)
* Sequence models in deep learning
* Attention Models (Transformers)

Bear in mind that the main purpose of the toolkit is educational. You may resort
to other toolboxes if you are looking for efficient implementations of the
algorithms described.

## Instructions for Students

> [!IMPORTANT] 
> This is the **student branch**. You are on the **correct** branch!

Download the code. If you are used to git just clone the student branch. For
example from the command line in do

```bash
git clone git@github.com:LxMLS/lxmls-toolkit.git lxmls-toolkit-student
cd lxmls-toolkit-student
```

### Install [uv](https://astral.sh/uv) 

**Linux and MacOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```


### Set up environment

[Reference](https://docs.astral.sh/uv/guides/install-python)<br>
If you do not have the proper python version, install it with 
```bash
uv python install 3.12
```

[Reference](https://docs.astral.sh/uv/guides/integration/pytorch) <br>
Choose the torch index based on your system and setup the environment:
```bash
uv sync --extra {cpu, cu118, cu124, cu126}
```

Activate the virtual environment with

**Linux and MacOS**
```bash
source ./.venv/bin/activate
```

**Windows**
```powershell
.venv\Scripts\activate
```

> [!IMPORTANT]
> Remember to run scripts from the root directory `lxmls-toolkit-student`

### Development

Install the `ruff` linter & `ty` type-checker with 
```bash
uv sync --extra dev 
```

To run all tests install `pytest`

```bash
uv sync --extra test
```

and run
```bash
pytest -m "not gpu" -n auto
```

Run tests that are GPU intensive with single worker using
```bash
pytest -m gpu -n 1
```

### Quantisation

Install `GPTQModel`

```bash
deactivate
mkdir 3p && cd 3p
git clone https://github.com/ModelScope/GPTQModel && cd GPTQModel
```

Install Python 3.13 with free-threading
[GPTQModel Issue #1660](https://github.com/ModelCloud/GPTQModel/issues/1610)

```console
$ uv python install 3.13t
$ uv venv .venv --python 3.13t
$ source .venv/bin/activate
$ python -VV
Python 3.13.3 experimental free-threading build (main, Apr  9 2025, 04:05:28) [Clang 20.1.0 ]
```

Build `GPTQModel` from source

```bash
uv pip install -r requirements.txt
uv pip install -v . --no-build-isolation
uv pip install datasets
```

Run `quantize.py`
```bash
python ../../lxmls/multimodal/gemma3/quantize.py
```
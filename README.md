![Python Version from PEP 621 TOML](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2Frshwndsz%2Flxmls-toolkit%2Fmaster%2Fpyproject.toml)

# LxMLS 2025

Machine Learning toolkit for Natural Language Processing.<br>
Written for [LxMLS - Lisbon Machine Learning Summer School](http://lxmls.it.pt)

* Scientific Python and Mathematical background
* Linear Classifiers (Gradient Descent)
* Feed-forward models in deep learning (Backpropagation)
* Sequence models in deep learning
* Attention Models (Transformers)
* Multimodal Models

> [!NOTE]
> Bear in mind that the main purpose of the toolkit is educational. 
> You may resort to other toolboxes if you are looking for efficient implementations of the algorithms described.

## Instructions for Students

> [!IMPORTANT] 
> Use the [student branch](https://github.com/LxMLS/lxmls-toolkit/tree/student) **not** this one 🚨!

Download the code. If you are used to git just clone the student branch. For
example from the command line in do

```bash
git clone https://github.com/LxMLS/lxmls-toolkit.git lxmls-toolkit-student
cd lxmls-toolkit-student
git checkout student
```

### Install [uv](https://astral.sh/uv) 

**Linux and MacOS**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows** <br>
Open Command Prompt (search for `cmd`) to run the following commands.

First, check if your system has `git` using 
```powershell
git  --version
```

If `git` isn't installed run the following command to install it
```powershell
winget install Git.Git
```

Then, install `uv` using
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```
If that errors out, try
```powershell
winget install astral-sh.uv
```

### Set up environment

[Reference](https://docs.astral.sh/uv/guides/install-python)<br>
If you do not have the proper python version, install it with 
```bash
uv python install 3.12
```

If you have an Nvidia GPU, get the CUDA driver version with 
```nvidia-smi``` or ```nvcc --version```.

[Reference](https://docs.astral.sh/uv/guides/integration/pytorch) <br>
Choose the torch index based on your system and setup the environment:
```bash
uv sync --extra {cpu, cu118, cu124, cu126}
```

For example, if you're on MacOS you'd use
```bash
uv sync --extra cpu
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

> [!NOTE]
> The following instructions are for developers building the toolkit.

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


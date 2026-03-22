<div align="center">

# CrediPred

Model architectures based on graph neural networks to learn credibility signals on billion-scale webgraphs.

<img src="img/credipred.png" alt="CrediGraph Logo" style="width: 100px; height: auto;" />

</div>

## Getting Started

### Prerequisites

The project uses [uv](https://docs.astral.sh/uv/) to manage and lock project dependencies for a consistent and reproducible environment. If you do not have `uv` installed on your system, visit [this page](https://docs.astral.sh/uv/getting-started/installation/) for installation instructions.

**Note**: If you have `pip`, you can invoke:

```sh
pip install uv
```

### Installation

```sh
# Clone the repo

# Enter the repo directory
cd CrediPred

# Install core dependencies into an isolated environment
uv sync

# The isolated env is .venv, you may source it like so:
source .venv/bin/activate
```

## Usage

### Running GNN Experiments

Given the size of our datasets we must leverage mini-batching in our GNN experiments. To do this we use PyG's `neighbor_loader`,
which requires additional libraries having undocumented build-time dependencies. As such, users are required to install them in their
own venv. seperate from `uv sync`.

pyg-lib:

```
uv pip install pyg-lib -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
```

PyTorch Sparse:

```
uv pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.7.0+${CUDA}.html
```

For information on installations of these additional libraries see [pyg-lib](https://github.com/pyg-team/pyg-lib) and [PyTorch Sparse](https://github.com/rusty1s/pytorch_sparse).

To run our baseline static experimentation:

```sh
uv run tgrag/experiments/main.py
```

Alternatively, you can design your own configuration, updating the model parameters:

```sh
uv run tgrag/experiments/main.py --config configs/your_config.yaml
```

To learn more about making a contribution to CrediGraph see our [contribution guide](./.github/CONTRIBUTION.md)

______________________________________________________________________

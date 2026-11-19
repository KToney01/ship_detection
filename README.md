# ship_detection
Computer vision project for classifying ships in satellite imagery.

## Project Objectives: 
This project is being developed as a learning project focused on the following:
- Computer vision
- `PyTorch`
- Neural network architecture
- Image representation and manipulation
- Git and version control
- Experimental design
- Research documentation


## Current Status

- Completed a reproducible baseline CNN experiment with held-out test evaluation.
- Refactored core functionality into a reusable Python package with automated tests.
- Implemented tested single-image inference with command-line device selection and confidence output.

## Baseline Results

The baseline CNN was selected using validation F1 and evaluated once on the held-out test set.

| Metric | Result |
| --- | ---: |
| Accuracy | 92.29% |
| Precistion | 87.60% |
| Recall | 79.58% |
| F1 Score | 0.8339 |

The selected checkpoint occured at epoch 17/20 with a validation F1 of 0.8954.

## Dataset / Methodology

- Uses the [ShipsNet satellite imagery dataset](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery), containing 4,000 RGB image chips at 80 x 80 pixels.
- The dataset contains:
 - 1,000 ship images
 - 3,000 no-ship images
- Labels and source-scene identifiers are parsed from the filenames.
- Data is split using `StratifiedGroupKFold` to preserve class balance while preventing images from the same source scene from appearing across training, validation, and test sets.
- The baseline split contains approximately:
 - 70% training data
 - 15% validation data
 - 15% test data
- Images are converted to PyTorch tensors, transformed from HWC to CHW format, and normalized from '[0, 255]' to '[0, 1]'
- The held-out test set is reserved for final evaluation and is not used for checkpoint selection.

## Features

- `PyTorch` convolutional neural network for binary ship/no-ship classification.
- Reusable Python package structure for:
 - Dataset loading and preprocessing
 - Scene-aware train/validation/test splitting
 - DataLoader construction
 - Model creation and checkpoint loading
 - Training and validation
 - Evaluation metrics
 - Experiment result persistence
- Tracks accuracy, precision, recall, and F1 score during evaluation.
- Selects and saves the best model checkpoint using validation F1.
- Stores experiment configuration and final results as JSON.
- Supports persistent epoch-by-epoch training history in CSV format.
- Uses configurable experiment directories and checkpoint paths to keep runs isolated.
- Includes automated tests with `pytest` for data handling, model behavior, training, evaluation, persistence, and inference.
- Notebook workflow is designed as an orchestration layer over reusable package code.

## Inference 

The trained baseline model can classify individual 80 x 80 RGB satellite image chips from the command line.
```bash
python -m ship_detection.cli \
    ~/data/shipsnet/example.png \
    --checkpoint checkpoints/EXP-001-baseline/best_model.pt
```
Example output:
```bash
Prediction: ship
Class ID: 1
Confidence: 65.18%
```

## Setup

Clone the repository and create a virtual environment:
```bash
git clone <repository-url>
cd ship_detection

python -m venv .venv
source .venv/bin/activate
```

Install the project and dependencies:
```bash
python -m pip install -r requirements.txt
python -m pip install -e .
```

Place the ShipsNet dataset in a local directory such as "~/data/shipsnet/"
The dataset directory can be overridden with:
```bash
export SHIPSNET_DATA_DIR=path/to/shipsnet
```

Model checkpoints are not tracked in Git. Download the baseline checkpoint from the project release [v0.1.0 project release](https://github.com/KToney01/ship_detection/releases/tag/v0.1.0) and place it at: `checkpoints/EXP-001-baseline/best_model.pt`


## Testing

The project includes automated tests for data handling, dataset splitting, model behavior, training, evaluation, persistence, inference, and CLI utilities.

Run the full test suite from the repository root:
```bash
pytest -v
```

The CUDA-unavailable error-path test is skipped on systems where CUDA is available.

## Repository Structure
```text
ship_detection/
├── experiments/
│   └── EXP-001-baseline/
│       ├── config.json
│       └── results.json
├── notebooks/
│   ├── 00_eda.ipynb
│   └── 01_baseline_experiment.ipynb
├── src/
│   └── ship_detection/
        ├──__init__.py
│       ├── cli.py
│       ├── data.py
│       ├── evaluate.py
│       ├── model.py
│       ├── predict.py
│       ├── train.py
│       └── utils.py
├── tests/
│   ├── test_cli.py
│   ├── test_data.py
│   ├── test_evaluate.py
│   ├── test_model.py
│   ├── test_predict.py
│   ├── test_train.py
│   └── test_utils.py
├── pyproject.toml
├── requirements.txt
├── requirements-gpu.txt
└── README.md
```
- 'notebooks/' contains exploratory analysis and the reproducible baseline experiment.
- 'scr/ship_detection/' contains the reusable Python package.
- 'tests/' contains the automated pytest suite.
- 'experiments/' stores the reproducible experiment configuration and results,
- 'checkpoints/' is created locally and excluded from Git.

## Dataset Attribution

This project uses the [ShipsNet satellite imagery dataset](https://www.kaggle.com/datasets/rhammell/ships-in-satellite-imagery), created from Planet satellite imagery made available through Planet's Open California dataset.

ShipsNet is distributed under the [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0/) license. Dataset files are not redistributed in this repository.
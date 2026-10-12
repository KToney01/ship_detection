# ship_detection
*Computer vision research project for classifying ships in satellite imagery.*

## Project Objectives: 
*This project is being develiped as a learning project focused on the following:*
- Computer vision
- PyTorch
- Neural network architecture
- Image representation and manipulation
- Git and version control
- Experimental design
- Research documentation


## Current Status

- Refactored core logic into reusable Python package modules, with baseline notebook serving as the experiment orchestration layer.
- Implemented reproducible experiment tracking, model checkpointing, and automated testing with 'pytest'.

## Dataset / Methodology

- Uses the ShipsNet satellite imagery dataset, containing 4,000 RGB image chips at 80 x 80 pixels.
- The dataset contains:
 - 1,000 ship images
 - 3,000 no-ship images
- Labels and source-scene identifiers are parsed from the filenames.
- Data is split using 'StratifiedGroupKFold' to preserve class balance while preventing images from the same source scene from appearing across training, validation, and test sets.
- The baseline split contains approximately:
 - 70% training data
 - 15% validation data
 - 15% test data
- Images are converted to PyTorch tensors, transformed from HWC to CHW format, and normalized from '[0, 255]' to '[0, 1]'
- The held-out test set is reserved for final evaluation and is not used for checkpoint selection.


## Features

- PyTorch convolutional neural network for binary ship/no-ship classification.
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
- Includes automated tests with 'pytest' for model behavior and utility functions.
- Notebook workflow is designed as an orchestration layer over reusable package code.

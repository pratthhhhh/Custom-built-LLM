# Custom-built-LLM

A custom-built Large Language Model implementation from scratch, featuring training, evaluation, and inference capabilities with LoRA (Low-Rank Adaptation) support.

## Overview

This project implements a complete LLM pipeline including model architecture, tokenization, training with LoRA fine-tuning, and evaluation metrics. It provides a flexible framework for building and experimenting with language models.

## Project Structure

The repository contains the following Python scripts:

- **main.py** - Entry point for the application; orchestrates the LLM workflow
- **config.py** - Configuration management for model parameters and training settings
- **model.py** - Core LLM model architecture and forward pass implementation
- **tokenizer.py** - Text tokenization and vocabulary management
- **train.py** - Training loop and optimization logic
- **lora.py** - Low-Rank Adaptation implementation for efficient fine-tuning
- **evaluate.py** - Model evaluation and performance metrics
- **score.py** - Scoring and inference utilities

## Setup Instructions

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/pratthhhhh/Custom-built-LLM.git
   cd Custom-built-LLM
   ```

2. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (If requirements.txt is not available, install common dependencies manually)

## Usage

### Basic Usage

Run the main entry point:
```bash
python main.py
```

### Configuration

Modify `config.py` to customize model parameters such as:
- Model size (hidden dimensions, number of layers)
- Tokenizer vocabulary size
- Training hyperparameters (learning rate, batch size, epochs)
- LoRA configuration for fine-tuning

### Training

To train the model:
```bash
python train.py --config config.py
```

### Evaluation

To evaluate model performance:
```bash
python evaluate.py --model_path <path_to_model>
```

### Inference

To generate predictions:
```bash
python score.py --input "Your input text here"
```

## Key Features

- **Custom Model Architecture** - Fully implemented LLM from scratch
- **Efficient Fine-tuning** - LoRA support for memory-efficient adaptation
- **Flexible Tokenization** - Custom tokenizer for text processing
- **Comprehensive Evaluation** - Built-in metrics for model assessment
- **Modular Design** - Clean separation of concerns for easy experimentation

## Future Enhancements

- Multi-GPU training support
- Advanced evaluation metrics
- Pre-trained model checkpoints
- API endpoint for inference

## Contributing

Feel free to submit issues and enhancement requests!

## Author

pratthhhhh

# CIFAR-10 CNN: Architecture Design, Feature Analysis, and Optimization

A PyTorch computer-vision project exploring convolutional neural networks (CNNs) on the CIFAR-10 image-classification dataset. The notebook moves beyond a single baseline model: it inspects intermediate representations, implements a configurable CNN, compares architectural choices, and develops a custom multi-path CNN with an ablation study.

## Project highlights

- CIFAR-10 image classification across 10 object categories.
- Reproducible train/validation/test workflow with a fixed seed.
- Training-only channel normalization and data augmentation.
- Baseline CNN implementation and training diagnostics.
- Feature-map, convolution-filter, channel-activation, and class-selectivity analysis.
- Configurable CNN supporting variable convolution blocks, activation functions, batch normalization, pooling, and standard or depthwise-separable convolutions.
- Controlled architecture experiments, including model depth/width, BatchNorm, activation choice, and separable convolutions.
- A custom **Multi-Path CNN** with Fine, Wide, and Separable feature-extraction paths.
- Ablation experiments and comparison of parameter counts, validation performance, MACs, training time, and inference time.

## Dataset

The notebook uses **CIFAR-10**, which contains 60,000 RGB images of size 32×32 across the following classes:

`airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck`

The data is organized as:

| Split | Samples |
|---|---:|
| Training | 45,000 |
| Validation | 5,000 |
| Test | 10,000 |

The validation set is separated from the official training data. Channel-wise normalization statistics are calculated using the training subset only. Random cropping and horizontal flipping are used for training augmentation; validation and test transformations do not apply augmentation.

## Models and experiments

### 1. Baseline CNN

A three-block CNN with convolution, batch normalization, ReLU, and max pooling, followed by adaptive average pooling and a fully connected classifier.

### 2. Configurable CNN

A reusable CNN class whose architecture is described by a configuration dictionary. It supports:

- Variable numbers of convolutional blocks
- Configurable channels, kernel size, stride, and padding
- ReLU, GELU, SiLU, and LeakyReLU activations
- Optional batch normalization and pooling
- Standard and depthwise-separable convolutions
- Adaptive average pooling and a configurable classifier head

### 3. Architecture comparison

The notebook compares a baseline with deeper/wider, depthwise-separable, no-BatchNorm, and GELU variants. The reported selection rule is the highest validation Macro-F1, with validation loss used as a tie-breaker.

### 4. Multi-Path CNN

The custom architecture uses three parallel paths:

- **Fine:** standard 3×3 convolutions for local details
- **Wide:** a 5×5 convolution followed by dilated convolutions for a wider receptive field
- **Separable:** depthwise-separable convolutions

The path outputs are concatenated and combined through a 1×1 fusion convolution before global average pooling and classification. The notebook compares the full architecture with a parameter-matched simple model and ablation variants.

## Results

The following values are transcribed from the saved notebook outputs. They refer to two separate experiments and should not be compared as if they were the same model-selection run.

### Main configurable-CNN experiment (Part 7)

| Metric | Test result |
|---|---:|
| Accuracy | **84.03%** |
| Macro precision | 84.03% |
| Macro recall | 84.03% |
| Macro-F1 | **84.01%** |
| Test loss | 0.4626 |
| Trainable parameters | 423,082 |

The selected model was the deeper/wider four-block CNN (32–64–128–256 channels), selected using validation Macro-F1.

### Multi-Path / ablation experiment (Part 9)

| Model | Validation accuracy | Validation Macro-F1 | Parameters |
|---|---:|---:|---:|
| Multi-Path (Fine + Wide + Separable) | 81.58% | 81.40% | 127,677 |
| Parameter-matched simple model | 76.30% | 75.97% | 128,510 |
| Ablation: without Fine path | 79.58% | 79.38% | 128,441 |
| Ablation: without Wide path | 80.08% | 79.85% | 123,909 |
| Ablation: without Separable path | 82.38% | **82.17%** | 130,954 |

The selected Part 9 ablation model (without the Separable path) achieved **80.63% test accuracy** and **80.53% test Macro-F1**.

## Repository structure

```text
cifar10-cnn-github/
├── README.md
├── requirements.txt
├── .gitignore
└── notebooks/
    └── cifar10-cnn-architecture-analysis.ipynb
```

During execution, the notebook creates local folders for generated artifacts:

- `data/` — downloaded CIFAR-10 data
- `results/` — experiment summaries and reports
- `checkpoints/` — model checkpoints
- `figures/` — training curves, confusion matrices, and feature visualizations

These generated files are not included in this repository package; they are recreated when the notebook is run.

## How to run

### Option A: Google Colab

1. Upload `notebooks/cifar10-cnn-architecture-analysis.ipynb` to Google Colab.
2. Select a GPU runtime if available (the original experiments were run on a T4 GPU).
3. Run the notebook from top to bottom.
4. The notebook downloads CIFAR-10 automatically and creates the output folders.

### Option B: Local environment

```bash
python -m venv .venv
```

Activate the environment, then install the dependencies:

```bash
pip install -r requirements.txt
```

Launch Jupyter:

```bash
jupyter notebook
```

Open the notebook under `notebooks/` and run all cells in order.

For GPU-enabled PyTorch, install the build appropriate for your CUDA setup using the official PyTorch installation selector: https://pytorch.org/get-started/locally/

## Reproducibility and evaluation notes

- The notebook uses a fixed random seed (`SEED = 42`) to make experiments more comparable.
- The test set is held out from training and validation-based model selection within each experiment.
- Architecture comparisons use a shared split and a common training protocol where specified in the notebook.
- GPU operations may not be perfectly deterministic across hardware and software environments.
- Runtime, training time, and inference time can vary by hardware and software versions.
- The reported results are the outputs saved in this notebook; rerunning the experiments may produce small differences.

## Limitations

- CIFAR-10 images are small (32×32), which limits fine-grained visual detail.
- The experiments use a single principal seed; multi-seed evaluation would provide stronger evidence about the stability of small performance differences.
- The custom multi-path model is an experimental architecture, not a claim of state-of-the-art performance.
- Results depend on the selected training budget and hardware.

## Technologies

`Python` · `PyTorch` · `torchvision` · `NumPy` · `Pandas` · `scikit-learn` · `Matplotlib` · `Seaborn` · `Jupyter Notebook`

## Author

**Nahid Hatami**  
Data Science  
Shahid Beheshti University

---

This repository is an educational and experimental deep-learning project focused on CNN architecture design, analysis, and comparison.

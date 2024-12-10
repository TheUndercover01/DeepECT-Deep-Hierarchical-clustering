# DeepECT: Deep Embedded Cluster Tree

This is an unofficial implementation of the paper "DeepECT: The Deep Embedded Cluster Tree" published in Data Science and Engineering.
[Link to the paper](https://link.springer.com/article/10.1007/s41019-020-00134-0)

## Overview

DeepECT is a novel deep learning-based hierarchical clustering method that combines the power of deep neural networks with traditional clustering approaches. The algorithm learns a hierarchical tree structure of clusters while simultaneously learning the feature representation of the data.

## Key Features

- Deep autoencoder-based feature learning
- Hierarchical clustering tree construction
- Joint optimization of reconstruction and clustering objectives
- Support for unsupervised learning on image datasets
- Implementation tested on Fashion-MNIST dataset

## Requirements

- Python 3.10+
- PyTorch
- torchvision
- numpy
- scikit-learn

## Project Structure

```
.
├── ClusterTree.ipynb     # Implementation of the hierarchical clustering tree
├── losses.ipynb         # Loss functions implementation
├── main.ipynb          # Main training and evaluation code
├── data/               # Directory for datasets
└── model_weights/      # Pretrained model weights
```

## Implementation Details

The implementation consists of three main components:

1. **Autoencoder Network**: Implemented in `main.ipynb`, responsible for learning the latent representation of the input data.

2. **Cluster Tree**: Implemented in `ClusterTree.ipynb`, manages the hierarchical structure of clusters and provides methods for:
   - Tree construction
   - Node management
   - Cluster assignment
   - Loss calculation (NC and DC losses)

3. **Training Process**:
   - Pre-training phase for autoencoder initialization
   - Joint training phase combining reconstruction and clustering objectives
   - Dynamic tree structure updates during training

## Usage

1. Clone the repository:
```bash
git clone https://github.com/TheUndercover01/DeepECT-Deep-Hierarchical-clustering.git
cd DeepECT-Deep-Hierarchical-clustering
```

2. Install dependencies:
```bash
pip install torch torchvision numpy scikit-learn
```

3. Run the notebooks in the following order:
   - First run `ClusterTree.ipynb` to set up the clustering infrastructure
   - Then run `main.ipynb` to train the model and perform clustering

## Pre-trained Models

The repository includes several pre-trained models:
- `model_weight_pretrained.pth`: Base pre-trained model
- `model_weight_pretrained_4000.pth`: Model trained on 4000 samples
- `model_weight_pretrained_tree_5000.pth`: Model with tree structure trained on 5000 samples

## Contributors

- [Aayush Deshmukh](https://github.com/TheUndercover01)
- [Kirubakaran M G](https://github.com/Kiruba061003)

## Citation

If you use this implementation in your research, please cite the original paper:

```bibtex
@article{deepect2020,
  title={DeepECT: the deep embedded cluster tree},
  author={Chen, Guoqing and Liu, San and Zhang, Yang and Liu, Xiang},
  journal={Data Science and Engineering},
  year={2020}
}
```

## License

This project is open-source and available under the MIT License.

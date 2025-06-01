# YZM304 Deep Learning Course Projects

## Overview

This repository contains all assignments and laboratory work for the **YZM304 Deep Learning** course at Ankara University, Department of Artificial Intelligence and Data Engineering (2024-2025 Spring Semester). The project explores various deep learning architectures and techniques through hands-on implementations and experiments.

## 📚 Course Structure

The course covers fundamental to advanced topics in deep learning through four main assignments and laboratory exercises:

- **Assignment 1**: Multi-Layer Perceptron Networks
- **Assignment 2**: Convolutional Neural Networks (CNNs)
- **Assignment 3**: Recurrent Neural Networks (RNNs)
- **Assignment 4**: Deep Reinforcement Learning
- **Lab Files**: Practical implementations and experiments

## 🗂️ Project Structure

```
├── README.md                           # This file
├── requirements.txt                    # Python dependencies for virtual environment
├── venv/                              # Virtual environment (not tracked in git)
├── firstAssignment/                    # Neural Network Fundamentals
│   ├── BankNote_Authentication.csv     # Dataset for binary classification
│   ├── threeLayerNetwork.py           # 3-layer neural network implementation
│   ├── twoLayerNetwork.py             # 2-layer neural network implementation
│   ├── twoLayerNetworkScikitLearn.py  # Scikit-learn comparison
│   └── README.md                      # Detailed assignment report
├── secondAssignment/                   # Convolutional Neural Networks
│   ├── model1_lenet.py                # LeNet-5 inspired architecture
│   ├── model2_improved.py             # Enhanced CNN with batch normalization
│   ├── model3_pretrained.py           # Transfer learning with ResNet18
│   ├── model4_hybrid.py               # Hybrid CNN + SVM approach
│   ├── model5_comparison.py           # Comparative analysis
│   ├── utils.py                       # Utility functions
│   ├── data/                          # MNIST and CIFAR-10 datasets
│   └── README.md                      # Detailed assignment report
├── thirdAssignment/                    # Recurrent Neural Networks
│   ├── customRnn.py                   # Custom RNN implementation from scratch
│   ├── libraryRnn.py                  # TensorFlow/Keras RNN implementation
│   ├── data.py                        # Data preprocessing utilities
│   ├── metrics.txt                    # Performance metrics
│   ├── images/                        # Training plots and confusion matrices
│   └── README.md                      # Detailed assignment report
├── fourthAssignment/                   # Deep Reinforcement Learning
│   ├── car_racing_ppo.py              # Proximal Policy Optimization
│   ├── car_racing_sac.py              # Soft Actor-Critic
│   ├── car_racing_td3.py              # Twin Delayed DDPG
│   ├── evaluate.py                    # Model evaluation script
│   ├── visualize.py                   # Results visualization
│   ├── models/                        # Trained model weights
│   ├── blog/                          # Detailed analysis blog
│   └── README.md                      # Detailed assignment report
└── labFiles/                          # Laboratory Notebooks
    ├── mlp.ipynb                      # Multi-layer perceptron experiments
    └── perceptron.ipynb               # Basic perceptron implementation
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+ (Python 3.9 or 3.10 recommended)
- pip (Python package manager)
- Virtual environment (recommended for isolation)

### Virtual Environment Setup

This project uses a Python virtual environment to manage dependencies and ensure consistent execution across different systems. Using a virtual environment prevents conflicts between project dependencies and system-wide packages.

#### Setting up the Virtual Environment

1. Clone the repository:
```bash
git clone <repository-url>
cd yzm304
```

2. Create a virtual environment:
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On macOS/Linux:
source venv/bin/activate

# On Windows:
venv\Scripts\activate
```

3. Install all project dependencies:
```bash
# Install all requirements from requirements.txt
pip install -r requirements.txt
```

4. Verify installation:
```bash
# Check if packages are installed correctly
pip list
```

#### Managing the Virtual Environment

- **Activate** the environment before working on the project:
  ```bash
  source venv/bin/activate  # macOS/Linux
  venv\Scripts\activate     # Windows
  ```

- **Deactivate** when done:
  ```bash
  deactivate
  ```

- **Update dependencies** (if needed):
  ```bash
  pip install -r requirements.txt --upgrade
  ```

### Dependencies Overview

The project uses several key libraries organized by assignment type:

- **Core Libraries**: NumPy, Pandas, Matplotlib, SciPy
- **Machine Learning**: Scikit-learn
- **Deep Learning**: PyTorch, TensorFlow
- **Reinforcement Learning**: Stable-Baselines3, Gymnasium
- **Development**: Jupyter Notebook, IPython

See `requirements.txt` for the complete list with version specifications.

## 📋 Assignment Details

### Assignment 1: Multi-Layer Perceptron Networks
**Objective**: Implement and compare 2-layer and 3-layer neural networks for binary classification

**Key Features**:
- Custom implementation using NumPy
- Bank Note Authentication dataset
- Comparison with Scikit-learn MLPClassifier
- Activation function analysis (tanh, ReLU)
- Performance metrics and convergence analysis

**Dataset**: Bank Note Authentication (4 features, binary classification)

### Assignment 2: Convolutional Neural Networks
**Objective**: Explore various CNN architectures for image classification

**Key Features**:
- LeNet-5 inspired architecture for MNIST
- Enhanced CNN with batch normalization
- Transfer learning with pre-trained ResNet18
- Hybrid CNN + SVM approach
- Comparative performance analysis

**Datasets**: MNIST (handwritten digits), CIFAR-10 (natural images)

### Assignment 3: Recurrent Neural Networks
**Objective**: Implement RNNs for sentiment analysis on text data

**Key Features**:
- Custom RNN implementation from scratch (NumPy)
- TensorFlow/Keras RNN implementation
- Text preprocessing and tokenization
- Sentiment classification (positive/negative)
- Performance comparison and analysis

**Dataset**: Custom sentiment analysis dataset with positive/negative text samples

### Assignment 4: Deep Reinforcement Learning
**Objective**: Compare state-of-the-art RL algorithms on continuous control tasks

**Key Features**:
- Three RL algorithms: PPO, SAC, TD3
- CarRacing-v3 environment (vision-based control)
- Comprehensive performance evaluation
- Statistical analysis and visualization
- Detailed algorithmic comparison

**Environment**: OpenAI Gym CarRacing-v3 (continuous control, image observations)

## 📊 Results Summary

### Neural Networks (Assignment 1)
- 3-layer network achieved better performance than 2-layer
- Custom implementation competitive with Scikit-learn
- Tanh activation showed good convergence properties

### CNNs (Assignment 2)
- Transfer learning (ResNet18) achieved highest accuracy
- Batch normalization significantly improved training stability
- Hybrid approaches showed potential for feature extraction tasks

### RNNs (Assignment 3)
- Both custom and library implementations achieved good sentiment classification
- Custom RNN demonstrated understanding of backpropagation through time
- TensorFlow implementation showed better optimization convergence

### Reinforcement Learning (Assignment 4)
- SAC achieved highest mean performance (6.94 ± 0.47)
- PPO showed most consistent results (6.77 ± 0.37)
- TD3 demonstrated competitive performance with higher variance

## 🔧 Usage Examples

**Note**: Always activate the virtual environment before running any scripts:
```bash
source venv/bin/activate  # macOS/Linux
# or
venv\Scripts\activate     # Windows
```

### Running Neural Network Training
```bash
cd firstAssignment
python threeLayerNetwork.py
python twoLayerNetwork.py
python twoLayerNetworkScikitLearn.py
```

### Training CNN Models
```bash
cd secondAssignment
python model1_lenet.py          # LeNet-5 on MNIST
python model2_improved.py       # Enhanced CNN with batch normalization
python model3_pretrained.py     # Transfer learning with ResNet18
python model4_hybrid.py         # Hybrid CNN + SVM approach
python model5_comparison.py     # Comparative analysis
```

### RNN Sentiment Analysis
```bash
cd thirdAssignment
python customRnn.py             # Custom implementation from scratch
python libraryRnn.py            # TensorFlow/Keras implementation
```

### Reinforcement Learning Training and Evaluation
```bash
cd fourthAssignment
python car_racing_ppo.py        # Train PPO agent
python car_racing_sac.py        # Train SAC agent
python car_racing_td3.py        # Train TD3 agent
python evaluate.py              # Evaluate all trained models
python visualize.py             # Visualize agent performance
```

### Working with Jupyter Notebooks
```bash
# Start Jupyter Notebook server
jupyter notebook

# Navigate to labFiles/ directory and open:
# - mlp.ipynb for multi-layer perceptron experiments
# - perceptron.ipynb for basic perceptron implementation
```

## 📈 Performance Metrics

Each assignment includes comprehensive evaluation metrics:
- **Classification Tasks**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Training Dynamics**: Loss curves, convergence analysis
- **Model Comparison**: Statistical significance testing
- **Visualization**: Training plots, performance comparisons

## 🎓 Learning Outcomes

Through these assignments, the following deep learning concepts are explored:

1. **Neural Network Fundamentals**: Backpropagation, gradient descent, activation functions
2. **Convolutional Networks**: Feature extraction, pooling, transfer learning
3. **Recurrent Networks**: Sequential processing, BPTT, text analysis
4. **Reinforcement Learning**: Policy optimization, value functions, continuous control
5. **Implementation Skills**: Both from-scratch and library-based implementations
6. **Evaluation Methodology**: Proper train/test splits, statistical analysis

## 🔧 Troubleshooting

### Common Virtual Environment Issues

1. **Virtual environment not activating**:
   ```bash
   # Make sure you're in the project directory
   cd /path/to/yzm304
   # Try recreating the virtual environment
   rm -rf venv
   python -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   ```

2. **Package installation errors**:
   ```bash
   # Update pip first
   pip install --upgrade pip
   # Then install requirements
   pip install -r requirements.txt
   ```

3. **PyTorch/TensorFlow compatibility issues**:
   - Ensure you have compatible versions of CUDA if using GPU
   - Check system requirements for your operating system
   - Consider using CPU-only versions for development

4. **Jupyter notebook kernel issues**:
   ```bash
   # Install kernel in virtual environment
   python -m ipykernel install --user --name=yzm304
   # Select 'yzm304' kernel when running notebooks
   ```

### Platform-Specific Notes

- **macOS**: May require Xcode command line tools: `xcode-select --install`
- **Windows**: Ensure Python is added to PATH during installation
- **Linux**: Some packages may require system-level dependencies (e.g., `python3-dev`)

## 📚 References

- Course materials from YZM304 Deep Learning, Ankara University
- Deep Learning by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- Hands-On Machine Learning by Aurélien Géron
- Stable-Baselines3 Documentation
- PyTorch and TensorFlow Official Documentation

## 📞 Contact

**Course**: YZM304 Deep Learning  
**Institution**: Ankara University, Department of AI and Data Engineering  
**Semester**: 2024-2025 Spring

---

*This repository represents a comprehensive exploration of deep learning techniques, from fundamental neural networks to advanced reinforcement learning algorithms, providing both theoretical understanding and practical implementation experience.*

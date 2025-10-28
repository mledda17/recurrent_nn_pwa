# Recurrent Neural Networks with ReLU activations are Continuous Piecewise Affine Systems Dynamical Systems and Viceversa

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)

Official code for the paper: **"Recurrent Neural Networks with ReLU activations are Continuous Piecewise Affine Systems Dynamical Systems and Viceversa"**

Authors:
- Marco Ledda, Student, IEEE
- Diego Deplano, Member, IEEE
- Alessandro Giua, Fellow, IEEE
- Mauro Franceschelli, Senior, IEEE

_All authors are with the DIEE, University of Cagliari, 09123 Cagliari, Italy._

---

## 📝 Abstract
> This paper proves the exact equivalence between ReLU-based recurrent neural networks (RNNs) and discrete-time continuous piecewise-affine (CPWA) dynamical systems. We show that every ReLU–RNN induces a finite polyhedral partition of the state–input space with affine dynamics on each region, and we provide a constructive procedure to derive the corresponding CPWA representation. Conversely, we prove that any CPWA system admits an exact ReLU–RNN realization, although only the existence of such a mapping is established without an explicit construction. The equivalence is numerically validated by comparing the trajectories of a two-layer ReLU–RNN and its corresponding CPWA model, confirming perfect agreement.


## 🚀 Getting Started

### Prerequisites

- **Python** 3.10 or higher  
- **Matplotlib** 3.10.1 or higher  
- **NumPy** 2.1.3 or higher  
- **SciPy** 1.15.2 or higher

### 1. Clone the repository
First of all, you will need to clone this repository. To do this, run the following command:

```bash
git clone https://github.com/mledda17/recurrent_nn_pwa.git
```

### 2. Install requirements and dependecies
Install dependencies via:

```bash
pip install -r requirements.txt
```

### 3. Run numerical validation
All experiments can be executed using the following command:
```bash
python scripts/simulate_compare.py
```

The results of the experiments will be put in a new folder "scripts/plots".

Official code for the paper: **"Recurrent Neural Networks with ReLU activations are Continuous Piecewise Affine Systems Dynamical Systems and Viceversa"**  

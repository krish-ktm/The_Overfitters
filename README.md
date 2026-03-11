# U-Net: Implementation, Reproduction, and Empirical Analysis

Welcome to the central repository for **The Overfitters**' Machine Learning project (CS 6140). This project focuses on implementing, reproducing, and analyzing the U-Net architecture for biomedical image segmentation.

## 📖 About The Project
See [`PROJECT_ABSTRACT.md`](PROJECT_ABSTRACT.md) for a detailed overview, objectives, and motivation behind this project.

## 🗂️ Project Structure
*(Note: As the project develops, the directory structure below should be updated to match the actual layout)*

```text
The_Overfits/
├── Proposal/                 # Project proposal documents (LaTeX, PDFs)
├── PROJECT_ABSTRACT.md       # High-level project overview and goals
├── README.md                 # Project documentation and setup instructions
├── src/                      # Source code for the project
│   ├── model.py              # U-Net PyTorch architecture implementation
│   ├── dataset.py            # Data loading and preprocessing pipelines
│   ├── train.py              # Training loops and model saving routines
│   └── evaluate.py           # Evaluation scripts and metric calculation
├── data/                     # Directory for storing dataset files
│   ├── kaggle_2018/          # Kaggle 2018 Data Science Bowl dataset
│   └── drive/                # DRIVE retinal vessel segmentation dataset
└── notebooks/                # Jupyter notebooks for visual analysis/EDA
```

## ⚙️ Getting Started

### Prerequisites
To run this project, you will need to have Python installed along with the following primary libraries:
- `torch` (PyTorch)
- `torchvision`
- `numpy`
- `matplotlib`
- `scikit-learn`
- *(Add other requirements such as `opencv-python`, `albumentations` etc. as they are added)*

### Installation

1. Clone the repository:
   ```bash
   git clone <repository-url>
   cd The_Overfits
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   *(Ensure you create a `requirements.txt` file as project dependencies are solidified)*

## 🚀 Usage

### Data Preparation
Download the datasets and place them in their respective folders under the `data/` directory:
1. **Kaggle 2018 Data Science Bowl**: [Link/Instructions]
2. **DRIVE Dataset**: [Link/Instructions]

### Training
To train the standard U-Net or baseline FCN, run:
```bash
python src/train.py --model unet --dataset kaggle
```
*(Update arguments based on your implementation)*

### Evaluation
To evaluate a trained model and generate metrics (Dice, IoU, Accuracy, Precision, Recall):
```bash
python src/evaluate.py --weights path/to/weights.pth
```

## 👥 Team
**The Overfitters**
- Ramyashree
- Ashwini Vitekar
- Krishkumar Patel

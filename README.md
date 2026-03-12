# 🏎️ F1 Race Predictor - Your Project

## Project Structure
```
f1-predictor/
├── README.md               ← You are here
├── setup.py                ← Run this FIRST to install everything
├── data/
│   └── fetch_data.py       ← Pulls F1 data using FastF1
├── models/
│   └── train_model.py      ← Trains the prediction model
├── visualizations/
│   └── visualize.py        ← Charts & track visualizations
└── notebooks/
    └── explore.ipynb       ← (optional) Jupyter notebook
```

## ⚡ Quick Start (Run in order)

### Step 1 - Install dependencies
```bash
pip install fastf1 pandas numpy matplotlib seaborn scikit-learn
```

### Step 2 - Fetch F1 data
```bash
cd data
python fetch_data.py
```

### Step 3 - Train the model
```bash
cd models
python train_model.py
```

### Step 4 - Visualize results
```bash
cd visualizations
python visualize.py
```

## 📦 Libraries Used
- **FastF1** - Official F1 data (lap times, telemetry, positions, weather)
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning model
- **Matplotlib / Seaborn** - Charts and visualizations
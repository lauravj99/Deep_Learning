# Cyclist Traffic Forecasting in Copenhagen
### Deep Learning Project – Google Colab Implementation

This repository contains the full implementation of our deep learning models developed to forecast hourly cyclist traffic counts in Copenhagen using time series data and historical weather conditions. The project was completed as part of coursework at Aarhus University.

**Authors:**  
Antía Bardelás Avilleira (202402718),  
Jacob Borg (202104887),  
Laura Vibeke Jacobsen (202105071)

---

## Repository structure

```
├── Deep_learning.ipynb       # Complete Google Colab notebook (full pipeline)
├── README.md            # Project documentation (this file)
```

Large files such as `.pt` and `.pkl` model outputs are not included due to GitHub's file size limits.

---

## Project objective

The goal of the project is to model and predict hourly cyclist counts across Copenhagen by combining:

- Historical bicycle count data  
- Weather time series (temperature, rain, snowfall, cloud cover, wind speed, humidity, etc.)  
- Road-specific embeddings  
- Temporal sequences of 168 hours (one week)

Two deep learning architectures were designed and compared:

1. **DeepLSTM** – Two-layer LSTM followed by fully connected layers  
2. **ConvBiLSTM** – CNN feature extractor followed by a bidirectional LSTM and MLP head

---

## Pipeline overview

### 1. Data acquisition
- Weather data collected using the Open-Meteo API  
- Bicycle count data from Kaggle (“Bike Traffic Counts in Copenhagen”)

### 2. Preprocessing
- Time alignment of weather and traffic data  
- Sliding window sequence generation (`T = 168`)  
- Standardization of numeric features  
- Road ID and weekday ID encoding  
- Train/validation/test split:  
  - Train: 2007–2010  
  - Test: 2011  
  - Validation: 2012

### 3. Model design
- Learnable embeddings (road, weekday, weather code)  
- LSTM and BiLSTM sequence modeling  
- CNN layers for feature extraction in ConvBiLSTM  
- Dropout regularization  
- Fully connected regression layers

### 4. Hyperparameter search
Explored parameters:

- Hidden layer sizes  
- Dropout rates  
- Learning rates  
- Weight decay (ConvBiLSTM only)

Validation MSE was used to identify the best configuration.

### 5. Evaluation
The notebook includes:

- Final MSE on train/validation/test  
- Learning curves for both models  
- Time-series predictions vs. true cyclist counts  
- Predictions plotted in the original (inverse-scaled) count units

---

## Results

Both models provide meaningful forecasts of hourly cyclist traffic.  
A detailed comparison and visual inspection of predictions are included in the notebook.  
Performance was evaluated on the road segment with the largest number of test samples.

---

## Missing files

Due to GitHub’s 100 MB upload limit, the following files are not included:

- `best_model1_state.pt`  
- `best_model2_state.pt`  
- `results_model1.pkl`  
- `results_model2.pkl`

These can be recreated by running the notebook and are available upon request.

---

## Running the notebook

The notebook is optimized for Google Colab.

To run locally:

```bash
pip install numpy pandas torch scikit-learn matplotlib openmeteo-requests kagglehub requests_cache retry_requests
```

---

## Authors

Antía Bardelás Avilleira (202402718)  
Jacob Borg (202104887)  
Laura Vibeke Jacobsen (202105071)  
Aarhus University

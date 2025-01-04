# Energy Consumption Prediction Project

## Overview

This project predicts energy consumption using variables such as temperature, humidity, and time of day. The model leverages multiple datasets to analyze and forecast energy usage patterns effectively.

---

## Environment Setup

To set up the environment for this project, follow these steps:

1. **Create a virtual environment**:

   ```bash
   python -m venv energy-consumption-env
   ```

2. **Activate the virtual environment**:

   - On Windows:
     ```bash
     energy-consumption-env\Scripts\activate
     ```
   - On macOS/Linux:
     ```bash
     source energy-consumption-env/bin/activate
     ```

3. **Install the required dependencies**:

   ```bash
   pip install -r requirements.txt
   ```

4. **Verify the installation**:

   ```bash
   python -c "import pandas; import matplotlib; import numpy; print('Environment setup complete!')"
   ```

---

## Datasets

### Primary Dataset: PJM Hourly Energy Consumption

- **Description**: Hourly energy consumption data in megawatts (MW) from PJM Interconnection LLC.
- **File**: `hourly_energy_consumption_pjme.csv`
- **Source**: [Kaggle](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)

### Complementary Dataset: Energy and Weather Data (Spain)

- **Description**: Historical energy consumption and weather data for Spain with hourly granularity.
- **Files**:
  - `energy_consumption_generation_spain.csv`: Energy consumption and generation by source.
  - `weather_data_spain.csv`: Weather data including temperature and humidity.
- **Source**: [Kaggle](https://www.kaggle.com/datasets/nicholasjhana/energy-consumption-generation-prices-and-weather)

---

## Data Collection and Preparation

### Goals

1. Ensure raw data is collected and organized in the repository structure.
2. Perform an initial inspection to identify issues (e.g., missing values, inconsistencies).
3. Prepare processed data for use in Jupyter notebooks.

### File Renaming and Organization

To maintain clarity, the files were renamed and organized as follows:

- **Renamed Files**:

  - `PJME_hourly.csv` → `hourly_energy_consumption_pjme.csv`
  - `energy_dataset.csv` → `energy_consumption_generation_spain.csv`
  - `weather_features.csv` → `weather_data_spain.csv`

- **Repository Structure**:

  ```
  data/
  ├── raw/
  │   ├── hourly_energy_consumption_pjme.csv
  │   ├── energy_consumption_generation_spain.csv
  │   ├── weather_data_spain.csv
  ├── processed/  # Cleaned and ready-to-use data
  ```

### Initial Data Inspection

Use the `notebooks/eda.ipynb` notebook to load and inspect the renamed datasets:

```python
import pandas as pd

# Load the data
energy_data = pd.read_csv("data/raw/hourly_energy_consumption_pjme.csv")
print(energy_data.head())

# Check data types and missing values
print(energy_data.info())
```

Perform an initial analysis to identify patterns or inconsistencies in the data.

---

## Project Structure

The project is organized as follows:

```
energy-consumption-prediction/
├── data/
│   ├── raw/
│   │   ├── hourly_energy_consumption_pjme.csv
│   │   ├── energy_consumption_generation_spain.csv
│   │   ├── weather_data_spain.csv
│   ├── processed/  # Cleaned and ready-to-use data
├── notebooks/
│   ├── data_exploration.ipynb  # Initial exploration of datasets
├── src/
│   ├── preprocess.py  # Data preprocessing scripts
│   ├── utils.py       # Utility functions
├── requirements.txt   # Python dependencies
├── LICENSE            # MIT license
└── README.md          # Project overview (this file)
```

---

## Key Features

1. **Initial Data Cleaning**:

   - Inspection and handling of missing values.
   - Preparation of cleaned datasets for downstream tasks.

2. **Exploratory Data Analysis (EDA)**:

   - Preliminary visualizations of energy consumption patterns.
   - Initial summary statistics and data quality checks.

---

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/your-username/energy-consumption-prediction.git
   cd energy-consumption-prediction
   ```

2. **Set up the environment** (as described above).

3. **Run Jupyter notebooks** for data exploration:

   ```bash
   jupyter notebook
   ```

4. **Inspect the results** to identify necessary next steps.

---

## Future Improvements

1. Perform detailed feature engineering to enhance the predictive power of the datasets.
2. Develop initial machine learning models for baseline predictions.
3. Incorporate additional data sources, such as weather APIs, for more robust analysis.
4. Extend the analysis to time-series forecasting techniques.
5. Build a Streamlit application for visualizing results interactively.

---

## Acknowledgments

- PJM Interconnection LLC for providing hourly energy consumption data.
- Nicholas J. Hanas for sharing comprehensive energy and weather data on Kaggle.


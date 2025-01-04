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

  ### Advanced Data Preparation

#### **Steps Performed**
1. **Handling Missing Values**: 
   - The dataset analyzed did not contain missing values. However, an example of mean imputation for numerical variables was included for future reference.
   - Example code:
     ```
     numerical_cols = ['PJME_MW']
     for col in numerical_cols:
         if energy_data[col].isnull().sum() > 0:
             energy_data[col].fillna(energy_data[col].mean(), inplace=True)
     ```

2. **Feature Engineering**:
   - New features were created to capture seasonal and temporal patterns in energy consumption:
     - `hour`: Hour of the day.
     - `day_of_week`: Day of the week.
   - Example code:
     ```
     energy_data['Datetime'] = pd.to_datetime(energy_data['Datetime'])
     energy_data.set_index('Datetime', inplace=True)
     energy_data['hour'] = energy_data.index.hour
     energy_data['day_of_week'] = energy_data.index.dayofweek
     ```

3. **Processed Data Saved**:
   - The cleaned and processed dataset was saved as `cleaned_energy_consumption.csv` for further analysis and modeling.
   - Example code:
     ```
     processed_path = "../data/processed/cleaned_energy_consumption.csv"
     energy_data.to_csv(processed_path, index=False)
     ```

4. **Visualizations**:
   - Time series plots were created to analyze daily and weekly trends in energy consumption.
   - Histograms were generated to explore the distribution of energy usage.

#### **Insights**
- Energy consumption shows clear daily and weekly patterns, with higher usage observed during weekdays compared to weekends.
- The new features (`hour` and `day_of_week`) provide valuable insights into temporal trends, which can improve model performance.


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

1. **Validate Processed Data**:
   - Perform additional exploratory data analysis (EDA) on the processed dataset to confirm the quality of new features and validate data transformations.
   - Ensure that the new temporal features (`hour`, `day_of_week`) provide meaningful insights for predictive modeling.

2. **Baseline Predictive Modeling**:
   - Implement initial predictive models using algorithms such as linear regression to establish a baseline performance.
   - Evaluate model quality using metrics such as \( R^2 \), RMSE, and MAE.

3. **Feature Engineering Refinement**:
   - Explore additional feature engineering opportunities to enhance model performance.
   - Incorporate external data sources, such as weather APIs, to enrich the dataset.

4. **Model Optimization**:
   - Experiment with more advanced machine learning models (e.g., Random Forest, Gradient Boosting) to improve predictions.
   - Perform hyperparameter tuning to optimize model configurations.

5. **Results Communication**:
   - Generate comprehensive visualizations (e.g., time series plots, scatter plots) to communicate key findings effectively.
   - Prepare a detailed report summarizing insights and model performance.

6. **Interactive Application Development**:
   - Build an interactive dashboard or application (e.g., using Streamlit) to visualize energy consumption predictions and allow user interaction with the results.

---

## Acknowledgments

- PJM Interconnection LLC for providing hourly energy consumption data.
- Nicholas J. Hanas for sharing comprehensive energy and weather data on Kaggle.


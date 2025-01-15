# Energy Consumption Prediction Project

## Overview

This project predicts energy consumption using temporal features such as the hour of the day and the day of the week. A baseline Linear Regression model was developed to analyze and forecast energy usage patterns effectively. The project involves data cleaning, feature engineering, predictive modeling, and visualization to understand energy consumption trends.

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

## Data Preparation

### Goals

1. Load raw data from PJM Hourly Energy Consumption dataset.
2. Perform an initial inspection to identify missing values or inconsistencies.
3. Engineer new features to capture temporal patterns in energy consumption.

### Steps Performed

1. **Loading Raw Data**:
The dataset is loaded into a pandas DataFrame for analysis:

   ```python
   energy_data = pd.read_csv("../data/raw/hourly_energy_consumption_pjme.csv")
   print(energy_data.head())
   ```

2. **Initial Inspection**:
- Checked for missing values:

  ```python
  print(energy_data.isnull().sum())
  ```

- Verified data types and basic statistics:

  ```python
  print(energy_data.info())
  print(energy_data.describe())
  ```

3. **Feature Engineering**:

Created new features to capture temporal patterns:
- `hour`: Hour of the day.
- `day_of_week`: Day of the week.

  ```python
  energy_data['Datetime'] = pd.to_datetime(energy_data['Datetime'])
  energy_data.set_index('Datetime', inplace=True)
  energy_data['hour'] = energy_data.index.hour
  energy_data['day_of_week'] = energy_data.index.dayofweek
  ```

4. **Saving Processed Data**:
The cleaned dataset was saved for further analysis:

  ```python
  processed_path = "../data/processed/cleaned_energy_consumption.csv"
  energy_data.to_csv(processed_path)
  ```

---

## Model Development

### Data Splitting

The dataset was split into training (70%), validation (15%), and test (15%) subsets while preserving its temporal structure.

   ```python
   from sklearn.model_selection import train_test_split
   X = energy_data[['hour', 'day_of_week']]
   y = energy_data['PJME_MW']
   X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, shuffle=False)
   X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, shuffle=False)
   ```

---

## Model Training

A Linear Regression model was trained using the training dataset:

   ```python
   from sklearn.linear_model import LinearRegression

   model = LinearRegression()
   model.fit(X_train, y_train)
   print("Model training complete!")
   ```

---

## Model Evaluation and Metrics

The model's performance was evaluated on the validation set using metrics such as MAE, MSE, RMSE, and \( R^2 \):

   ```python
   from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
   import numpy as np

   y_pred = model.predict(X_val)

   mae = mean_absolute_error(y_val, y_pred)
   mse = mean_squared_error(y_val, y_pred)
   rmse = np.sqrt(mse)
   r2 = r2_score(y_val, y_pred)

   print("Evaluation Metrics:")
   print(f"MAE: {mae}")
   print(f"MSE: {mse}")
   print(f"RMSE: {rmse}")
   print(f"R²: {r2}")
   ```

### Results Summary
- MAE: 4362.43 MW
- RMSE: 5331.62 MW
- \( R^2 \): 0.23


## Visualization of Model Predictions

A comparison plot between actual and predicted values was created to assess model performance visually:

   ```python
   import matplotlib.pyplot as plt

   plt.figure(figsize=(10, 6))
   plt.plot(y_val.values[:100], label="Actual Values", marker='o')
   plt.plot(y_pred[:100], label="Predicted Values", marker='x')
   plt.legend()
   plt.title("Actual vs Predicted Energy Consumption")
   plt.xlabel("Samples")
   plt.ylabel("Energy Consumption (MW)")
   plt.show()
   ```

---

## Key Features

1. **Initial Data Cleaning**:
   - Addressed missing values in numerical columns.
   - Prepared a cleaned dataset for downstream tasks.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized temporal patterns in energy consumption.
   - Generated summary statistics to understand data distributions.

3. **Baseline Predictive Modeling**:
   - Developed a Linear Regression model as a baseline for forecasting.

---

## Usage

1. **Clone the repository**:

   ```bash
   git clone https://github.com/rennanreis/energy-consumption-prediction.git
   cd energy-consumption-prediction
   ```

2. **Set up the environment** (as described above).

3. **Run Jupyter notebooks** for data exploration:

   ```bash
   jupyter notebook
   ```

4. **Inspect the results** to identify necessary next steps.

---

## Project Structure

The project is organized as follows:

```text
energy-consumption-prediction/
├── data/
│   ├── raw/
│   │   ├── hourly_energy_consumption_pjme.csv
│   ├── processed/
│       ├── cleaned_energy_consumption.csv
├── notebooks/
│   ├── eda.ipynb  # Exploratory Data Analysis notebook
│   ├── modeling.ipynb  # Predictive modeling notebook
├── src/
│   ├── preprocess.py  # Data preprocessing scripts
│   ├── utils.py       # Utility functions
├── requirements.txt    # Python dependencies
├── LICENSE             # MIT license
└── README.md           # Project overview (this file)
```

---

### Future Improvements

1. **Feature Engineering**:
   - Add external variables such as temperature or holidays to improve predictions.

2. **Advanced Modeling**:
   - Experiment with advanced algorithms like Random Forest or Gradient Boosting.

3. **Hyperparameter Tuning**:
   - Use Grid Search or Random Search to optimize model performance.

4. **Time Series-Specific Models**:
   - Explore models like ARIMA or LSTM for better handling of temporal dependencies.

5. **Cross-Validation**:
   - Implement cross-validation to ensure robust evaluation of model generalizability.

---

## Acknowledgments

- PJM Interconnection LLC for providing hourly energy consumption data.
- Nicholas J. Hanas for sharing comprehensive energy and weather data on Kaggle.
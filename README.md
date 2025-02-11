# Incident Classification Model

## Overview

This project implements an **XGBoost classifier** for predicting the state of incidents based on historical data. The model uses **SMOTE** and **Random Undersampling** to balance the dataset, ensuring that classifications are accurate and reliable. The script processes raw incident data, extracts key features, trains an optimized machine learning model, and generates an **incident report** summarizing predictions and recommended actions.

## Dataset

The model is trained on an **incident event log dataset** that contains:

- **Time-based features** (day of the week, hour of the day, weekend indicator)
- **Categorical encodings** for priority, impact, urgency, category, and assignment group
- **Incident state labels** that the model aims to predict

The dataset is loaded from a CSV file, with automatic detection of column separators (`comma` or `semicolon`).

## Features

- **Automated Data Preprocessing**: Converts timestamps, extracts time-based features, and encodes categorical variables.
- **Class Balancing**: Uses **undersampling** for overrepresented classes and **SMOTE** for underrepresented ones.
- **XGBoost Training**: Trains a multi-class classifier with **300 estimators**, **max depth of 7**, and **learning rate of 0.05**.
- **Overfitting Detection**: Compares training and test accuracy, issuing a warning if overfitting is detected.
- **Incident Report Generation**: Predicts incident states for test cases and recommends actions, saving results to `incident_report.txt`.

## Requirements

To run this project, you need the following dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost
```

## Running the Model

To execute the script and train the model:

```bash
python incident_classifier.py
```

This will:

1. Load and preprocess the dataset from `incident_event_log.csv`.
2. Train an XGBoost classifier on the processed dataset.
3. Evaluate model performance and detect overfitting.
4. Generate `incident_report.txt` with predicted states and recommended actions.

## Output

The script generates:

- **Console Output**: Training/Test accuracy and overfitting warnings.
- **Incident Report** (`incident_report.txt`):
  ```
  Incident Report
  ===============================
  Training Accuracy: 0.85
  Test Accuracy: 0.82
  Model does not appear to be overfitting.
  -----------------------------------
  Incident 1:
    - Predicted State: Active
    - Recommended Action: Ensure assigned team is actively investigating.
  -----------------------------------
  ```

## Author

Developed by Fanny Nyberg for predictive incident classification using machine learning.

## License

This project is licensed under the MIT License. This means you are free to use, modify, and distribute the software, but it comes without warranty. For more details, see the [MIT License](https://opensource.org/licenses/MIT).


# Incident Classification Model

This project was a contribution to the case week *Tech Talent of the Year 2025* at SEB. This solution won 2nd place in the final competition.

## Overview

This project implements an **XGBoost classifier** for predicting the state of incidents based on historical data. The model uses **SMOTE** and **Random Undersampling** to balance the dataset, ensuring that classifications are accurate and reliable. The script processes raw incident data, extracts key features, trains an optimized machine learning model, and generates an **AI-driven incident report** using **Google Vertex AI**.

## Dataset

The model is trained on an **incident event log dataset** that contains:

- **Time-based features** (day of the week, hour of the day, weekend indicator)
- **Categorical encodings** for priority, impact, urgency, category, and assignment group
- **Incident state labels** that the model aims to predict
- **Balanced class distribution** using **undersampling** for overrepresented classes and **SMOTE** for underrepresented ones

The dataset is loaded from a CSV file, with automatic detection of column separators (`comma` or `semicolon`).

## Features

- **Automated Data Preprocessing**: Converts timestamps, extracts time-based features, and encodes categorical variables.
- **Class Balancing**: Uses **undersampling** for overrepresented classes and **SMOTE** for underrepresented ones.
- **XGBoost Training**: Trains a multi-class classifier with **300 estimators**, **max depth of 7**, and **learning rate of 0.05**.
- **Overfitting Detection**: Compares training and test accuracy, issuing a warning if overfitting is detected.
- **Incident Report Generation**: Uses **Google Vertex AI (Gemini 1.5 Flash)** to analyze incidents and generate detailed incident reports.

## Requirements

To run this project, you need the following dependencies:

```bash
pip install pandas numpy matplotlib scikit-learn imbalanced-learn xgboost vertexai
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
4. Generate `incident_report.md` with AI-generated insights on predicted incidents.

## Output

The script generates:

- **Console Output**: Training/Test accuracy and overfitting warnings.
- **Incident Report** (`incident_report.md`):
  ```
  ### Incident Reports
  
  **Training Accuracy:** 0.89
  **Test Accuracy:** 0.85
  
  ### Incident 1: INC0000045
  **State:** Awaiting Vendor
  **Priority:** High
  **Impact:** Critical
  **Urgency:** High
  **Reopened:** 2 times
  **Reassigned:** 3 times
  **Resolution Time:** 2025-02-12 14:30:00
  **Possible Cause:** Network Failure
  
  **Analysis:**
  The incident was awaiting a vendor response and had high priority. The root cause appears to be a network failure affecting multiple users. Recommended actions:
  
  1. **Vendor Follow-Up** – Ensure the vendor provides a status update.
  2. **Temporary Workaround** – Consider alternative connectivity solutions.
  3. **Post-Incident Review** – Conduct a review to prevent recurrence.
  
  --------------------------------------------------
  ```

## Author

Developed by Fanny Nyberg for predictive incident classification using machine learning and AI-driven reporting.

## License

This project is licensed under the MIT License. This means you are free to use, modify, and distribute the software, but it comes without warranty. For more details, see the [MIT License](https://opensource.org/licenses/MIT).

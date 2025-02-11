import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb

# Function to load the dataset from a CSV file
def load_data(file_path):
    df = pd.read_csv(file_path, sep=",", encoding="utf-8")
    if 'opened_at' not in df.columns:
        df = pd.read_csv(file_path, sep=";", encoding="utf-8")
    df.columns = df.columns.str.strip()
    return df

# Function to preprocess data and encode categorical features
def preprocess_data(df):
    if 'opened_at' in df.columns:
        df['opened_at'] = pd.to_datetime(df['opened_at'], format="%d/%m/%Y %H:%M", errors='coerce', dayfirst=True)
    if 'resolved_at' in df.columns:
        df['resolved_at'] = pd.to_datetime(df['resolved_at'], format="%d/%m/%Y %H:%M", errors='coerce', dayfirst=True)
    
    df = df.dropna(subset=['opened_at'])
    df['day_of_week'] = df['opened_at'].dt.dayofweek
    df['hour'] = df['opened_at'].dt.hour
    df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    df = df.dropna(subset=['incident_state'])
    df['incident_state'] = df['incident_state'].astype(str).str.strip()
    df['incident_state'] = df['incident_state'].replace({"": "Unknown", None: "Unknown"})
    df['incident_state'] = df['incident_state'].astype('category')
    
    incident_state_mapping = df['incident_state'].cat.categories
    df['incident_state_encoded'] = df['incident_state'].cat.codes
    
    df['priority_encoded'] = df['priority'].astype('category').cat.codes
    df['impact_encoded'] = df['impact'].astype('category').cat.codes
    df['urgency_encoded'] = df['urgency'].astype('category').cat.codes
    df['category_encoded'] = df['category'].astype('category').cat.codes
    df['assignment_group_encoded'] = df['assignment_group'].astype('category').cat.codes
    
    return df[['day_of_week', 'hour', 'is_weekend', 'priority_encoded', 'impact_encoded', 'urgency_encoded',
               'category_encoded', 'assignment_group_encoded', 'incident_state_encoded']], incident_state_mapping

# Function to balance dataset using undersampling and SMOTE
def balance_data(X, y):
    class_counts = y.value_counts()
    max_class_count = class_counts.max()
    min_class_count = class_counts.min()

    if max_class_count > 4 * min_class_count:
        undersample_strategy = {cls: min(count, 4 * min_class_count) for cls, count in class_counts.items()}
        undersample = RandomUnderSampler(sampling_strategy=undersample_strategy, random_state=42)
        X, y = undersample.fit_resample(X, y)
    
    min_class_size = max(500, int(max_class_count * 0.1))
    smote_strategy = {cls: min_class_size for cls, count in class_counts.items() if count < min_class_size}
    
    for cls in smote_strategy.keys():
        if smote_strategy[cls] < class_counts[cls]:
            smote_strategy[cls] = class_counts[cls]
    
    if smote_strategy:
        smote = SMOTE(sampling_strategy=smote_strategy, k_neighbors=1, random_state=42)
        X, y = smote.fit_resample(X, y)
    
    return X, y

# Function to train XGBoost classifier and generate report
def train_model(df, incident_state_mapping):
    X = df.drop(columns=['incident_state_encoded'])
    y = df['incident_state_encoded']
    
    X_balanced, y_balanced = balance_data(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(
        objective='multi:softmax',
        num_class=len(set(y)),
        eval_metric='mlogloss',
        n_estimators=300,
        max_depth=7,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        gamma=0.1,
        random_state=42
    )
    model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_test, y_test)], verbose=False)

    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    overfitting_warning = "Warning: Model may be overfitting!" if train_accuracy - test_accuracy > 0.05 else "Model does not appear to be overfitting."
    
    y_pred = model.predict(X_test[:10])
    report_lines = [
        "Incident Report", "===============================",
        f"Training Accuracy: {train_accuracy:.2f}",
        f"Test Accuracy: {test_accuracy:.2f}",
        overfitting_warning,
        "-----------------------------------"
    ]
    
    recommended_actions = {
        "Active": "Ensure assigned team is actively investigating.",
        "Awaiting User Info": "Review incident details for next steps.",
        "New": "Review incident details for next steps.",
        "Resolved": "Verify resolution and close the incident if confirmed.",
        "Closed": "No further action needed. Incident is closed.",
        "Awaiting Evidence": "Follow up for required documentation or proof.",
        "Awaiting Vendor": "Contact the vendor for an update on progress.",
        "Awaiting Problem Resolution": "Coordinate with the problem management team.",
        "Awaiting Change": "Monitor the related change request for updates.",
        "Awaiting Approval": "Follow up with approvers for decision-making.",
        "Canceled": "Incident has been canceled, no further action required.",
        "Unknown": "Review details and determine next steps."
    }
    
    for i, pred in enumerate(y_pred):
        state = incident_state_mapping[pred]
        action = recommended_actions.get(state, "Review incident details and determine next steps.")
        report_lines.append(f"Incident {i+1}:")
        report_lines.append(f"  - Predicted State: {state}")
        report_lines.append(f"  - Recommended Action: {action}")
        report_lines.append("-----------------------------------")
    
    with open("incident_report.txt", "w") as file:
        file.write("\n".join(report_lines))
    
    print("Incident report has been generated: incident_report.txt")
    return model, X_test, y_test

if __name__ == "__main__":
    file_path = "incident_event_log.csv"
    df, incident_state_mapping = preprocess_data(load_data(file_path))  
    model, X_test, y_test = train_model(df, incident_state_mapping)

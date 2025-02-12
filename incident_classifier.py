import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import xgboost as xgb
import vertexai
from vertexai.generative_models import GenerativeModel

# Initialize Vertex AI
vertexai.init(project="tech-talent-3-57e19185", location="europe-west1")

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
        df['opened_at'] = pd.to_datetime(df['opened_at'], errors='coerce', dayfirst=True)
        df['day_of_week'] = df['opened_at'].dt.dayofweek
        df['hour'] = df['opened_at'].dt.hour
        df['is_weekend'] = df['day_of_week'].apply(lambda x: 1 if x >= 5 else 0)
    
    df = df.dropna(subset=['incident_state'])
    df['incident_state'] = df['incident_state'].astype(str).str.strip()
    df['incident_state'] = df['incident_state'].replace({"": "Unknown", None: "Unknown"})
    df['incident_state'] = df['incident_state'].astype('category')
    
    incident_state_mapping = df['incident_state'].cat.categories
    df['incident_state_encoded'] = df['incident_state'].cat.codes
    
    return df, incident_state_mapping

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

# Function to train XGBoost classifier and return accuracy
def train_model(df):
    X = df[['day_of_week', 'hour', 'is_weekend']]
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
    model.fit(X_train, y_train)
    
    train_accuracy = accuracy_score(y_train, model.predict(X_train))
    test_accuracy = accuracy_score(y_test, model.predict(X_test))
    
    print(f"\nTraining Accuracy: {train_accuracy:.2f}")
    print(f"Test Accuracy: {test_accuracy:.2f}")
    
    return model, train_accuracy, test_accuracy

# Function to generate an AI-driven incident report
def generate_incident_report(incident):
    model = GenerativeModel("gemini-1.5-flash-002")
    
    prompt_text = (
        f"### Incident Analysis Report\n"
        f"**Incident ID:** {incident['number']}\n"
        f"**State:** {incident['incident_state']}\n"
        f"**Priority:** {incident.get('priority', 'Unknown')}\n"
        f"**Impact:** {incident.get('impact', 'Unknown')}\n"
        f"**Urgency:** {incident.get('urgency', 'Unknown')}\n"
        f"**Reopened:** {incident['reopen_count']} times\n"
        f"**Reassigned:** {incident['reassignment_count']} times\n"
        f"**Resolution Time:** {incident.get('resolved_at', 'Unknown')}\n"
        f"**Possible Cause:** {incident.get('caused_by', 'Unknown')}\n\n"
        "Analyze the given incident details and provide a clear summary with potential causes and recommended actions."
    )
    
    response = model.generate_content(prompt_text)
    return response.text

# Main execution
if __name__ == "__main__":
    file_path = "incident_event_log.csv"
    df, incident_state_mapping = preprocess_data(load_data(file_path))  
    model, train_acc, test_acc = train_model(df)
    
    sample_incident = df.iloc[0].to_dict()
    incident_report = generate_incident_report(sample_incident)
    
    # Save the report to a file
    report_filename = "incident_report.md"
    with open(report_filename, "w") as file:
        file.write(f"### Incident Report: {sample_incident['number']}\n\n")
        file.write(f"**Training Accuracy:** {train_acc:.2f}\n")
        file.write(f"**Test Accuracy:** {test_acc:.2f}\n\n")
        file.write(incident_report)
    
    print(f"Incident report saved to {report_filename}")

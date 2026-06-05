import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from data_loader import load_stroke_data

def build_and_train_model():
    """
    Loads data, builds a Scikit-Learn preprocessing & training pipeline, 
    evaluates it, and exports the final pipeline model file.
    """
    # 1. Load Data
    df = load_stroke_data()
    
    # Drop 'id' right away as it has no predictive power
    df = df.drop(columns=['id'])
    
    # 2. Split Features (X) and Target (y)
    X = df.drop(columns=['stroke'])
    y = df['stroke']
    
    # 3. Separate feature types for distinct pipeline transformations
    categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
    numerical_cols = ['age', 'hypertension', 'heart_disease', 'avg_glucose_level', 'bmi']
    
    # 4. Define Preprocessing transformers
    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')), # Handles missing 'bmi' values safely
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols),
            ('cat', categorical_transformer, categorical_cols)
        ]
    )
    
    # 5. Create complete ML Pipeline (Preprocessing + Model Model)
    # Using balanced class weight since stroke datasets are heavily imbalanced
    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42))
    ])
    
    # 6. Train-Test Split & Fit Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("Training the Random Forest pipeline...")
    model_pipeline.fit(X_train, y_train)
    
    # Evaluate score
    score = model_pipeline.score(X_test, y_test)
    print(f"Model trained successfully! Test Accuracy Score: {score:.4f}")
    
    # 7. Save the trained pipeline file
    joblib.dump(model_pipeline, 'stroke_model.pkl')
    print("Saved pipeline to 'stroke_model.pkl'")
    return model_pipeline

def get_or_train_model():
    """Helper function to load the model file or train it if it's missing."""
    try:
        model = joblib.load('stroke_model.pkl')
        return model
    except FileNotFoundError:
        return build_and_train_model()

if __name__ == "__main__":
    build_and_train_model()
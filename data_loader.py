import os
import glob
import kagglehub
import pandas as pd

def load_stroke_data():
    """
    Downloads the latest version of the fedesoriano/stroke-prediction-dataset 
    via kagglehub and loads the main healthcare CSV file into a pandas DataFrame.
    """
    print("Downloading dataset from Kaggle...")
    # Download latest version of the dataset
    path = kagglehub.dataset_download("fedesoriano/stroke-prediction-dataset")
    print("Path to dataset files:", path)
    
    # Search for the CSV file in the downloaded directory path
    csv_files = glob.glob(os.path.join(path, "*.csv"))
    
    if not csv_files:
        raise FileNotFoundError("No CSV file found in the downloaded Kagglehub path.")
        
    # Read the dataset (matches healthcare-dataset-stroke-data.csv)
    df = pd.read_csv(csv_files[0])
    return df

if __name__ == "__main__":
    # Test block to ensure downloading works locally
    try:
        data = load_stroke_data()
        print(f"Successfully loaded data! Shape: {data.shape}")
        print(data.columns)
    except Exception as e:
        print(f"Error loading data: {e}")
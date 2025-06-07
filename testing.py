import streamlit as st
import pandas as pd
import os
from pathlib import Path

def load_dataset():
    """
    Load dataset with multiple fallback options
    """
    # List of possible dataset file paths to try
    possible_paths = [
        "data/thyroid_data.csv",
        "dataset/thyroid_data.csv", 
        "thyroid_data.csv",
        "data.csv",
        "thyroid_dataset.csv",
        "Data/thyroid_data.csv",  # Case variations
        "Dataset/thyroid_data.csv"
    ]
    
    # Also check for other common formats
    possible_extensions = ['.csv', '.xlsx', '.json', '.parquet']
    base_names = ['thyroid_data', 'data', 'dataset', 'thyroid_dataset']
    
    # Add all combinations
    for base_name in base_names:
        for ext in possible_extensions:
            possible_paths.extend([
                f"data/{base_name}{ext}",
                f"dataset/{base_name}{ext}",
                f"{base_name}{ext}"
            ])
    
    # Try to load from each possible path
    for path in possible_paths:
        try:
            if os.path.exists(path):
                st.write(f"✅ Found dataset at: {path}")
                
                # Load based on file extension
                if path.endswith('.csv'):
                    df = pd.read_csv(path)
                elif path.endswith('.xlsx'):
                    df = pd.read_excel(path)
                elif path.endswith('.json'):
                    df = pd.read_json(path)
                elif path.endswith('.parquet'):
                    df = pd.read_parquet(path)
                else:
                    # Default to CSV
                    df = pd.read_csv(path)
                
                st.write(f"📊 Dataset loaded successfully! Shape: {df.shape}")
                return df, path
                
        except Exception as e:
            st.write(f"❌ Failed to load {path}: {str(e)}")
            continue
    
    # If no dataset found, show debug information
    st.error("Dataset file not found. Please check the file path.")
    show_debug_info()
    return None, None

def show_debug_info():
    """
    Show debug information to help identify the issue
    """
    st.write("**🔧 Debug Information**")
    
    # Show current working directory
    current_dir = os.getcwd()
    st.write(f"Current working directory: `{current_dir}`")
    
    # List all files in current directory
    st.write("Files in current directory:")
    try:
        files = os.listdir('.')
        for file in sorted(files):
            if os.path.isfile(file):
                st.write(f"📄 {file}")
            else:
                st.write(f"📁 {file}/")
    except Exception as e:
        st.write(f"Error listing files: {e}")
    
    # Check for data directories
    data_dirs = ['data', 'dataset', 'Data', 'Dataset']
    for dir_name in data_dirs:
        if os.path.exists(dir_name) and os.path.isdir(dir_name):
            st.write(f"Contents of `{dir_name}/` directory:")
            try:
                files = os.listdir(dir_name)
                for file in sorted(files):
                    file_path = os.path.join(dir_name, file)
                    if os.path.isfile(file_path):
                        st.write(f"📄 {dir_name}/{file}")
                    else:
                        st.write(f"📁 {dir_name}/{file}/")
            except Exception as e:
                st.write(f"Error listing {dir_name}: {e}")

# Usage in your main Streamlit app
def main():
    st.title("🧠 Thyroid Cancer Outcome Predictor")
    st.write("This app predicts the **outcome** of differentiated thyroid cancer based on patient information.")
    
    # Try to load the dataset
    df, dataset_path = load_dataset()
    
    if df is not None:
        # Dataset loaded successfully
        st.success(f"Dataset loaded from: {dataset_path}")
        
        # Show basic dataset info
        st.write("### Dataset Overview")
        st.write(f"- **Rows**: {df.shape[0]}")
        st.write(f"- **Columns**: {df.shape[1]}")
        
        # Show first few rows
        st.write("### Sample Data")
        st.dataframe(df.head())
        
        # Show column names
        st.write("### Column Names")
        st.write(df.columns.tolist())
        
        # Continue with your app logic here...
        # Add your prediction model code here
        
    else:
        st.error("Failed to load dataset. Please check the file path and format.")
        st.write("**Next Steps:**")
        st.write("1. Make sure your dataset file is uploaded to your GitHub repository")
        st.write("2. Check that the file name matches what your code expects")
        st.write("3. Ensure the file is not listed in .gitignore")
        st.write("4. Verify the file format (CSV, Excel, etc.)")

if __name__ == "__main__":
    main()

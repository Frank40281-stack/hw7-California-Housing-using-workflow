# save_model.py
import joblib
import pandas as pd
from feature_selection import download_data, preprocess_data, get_rankings

def train_and_save_pipeline(output_filename="california_housing_pipeline.joblib"):
    # 1. Download and load data
    df = download_data()
    
    # 2. Preprocess data (impute, one-hot encode, split, scale)
    X_train, X_test, y_train, y_test = preprocess_data(df)
    
    # 3. Calculate feature rankings
    rankings = get_rankings(X_train, y_train)
    
    # 4. Select the best algorithm (e.g., Pearson or SFS) and get the Sweet Spot (k=7) features
    # SFS (Forward) is a robust wrapper method. Let's use SFS top 7 features.
    selected_algorithm = 'SFS (Forward)'
    top_7_features = rankings[selected_algorithm][:7]
    print(f"\nTraining final model on top 7 features of {selected_algorithm}:")
    print(top_7_features)
    
    # 5. Fit the final model on scaled training data
    from sklearn.linear_model import LinearRegression
    model = LinearRegression()
    model.fit(X_train[top_7_features], y_train)
    
    # 6. Re-load original data to fit scaler and save it
    # To make the saved pipeline fully independent, we need:
    # - The trained model
    # - The fitted StandardScaler (to scale user inputs of all 13 features)
    # - The list of features used in the model
    # Let's fit the scaler on the original training set before scaling
    df_clean = df.copy().dropna(subset=['total_bedrooms'])
    df_clean = pd.get_dummies(df_clean, columns=['ocean_proximity'], dtype=float)
    X_raw = df_clean[[col for col in df_clean.columns if col != 'median_house_value']]
    y_raw = df_clean['median_house_value']
    
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
        X_raw, y_raw, test_size=0.2, random_state=42
    )
    
    # Impute missing bedrooms
    bedrooms_median = X_raw_train['total_bedrooms'].median()
    X_raw_train = X_raw_train.copy()
    X_raw_train['total_bedrooms'] = X_raw_train['total_bedrooms'].fillna(bedrooms_median)
    
    scaler = StandardScaler()
    scaler.fit(X_raw_train)
    
    # Save everything in a pipeline dictionary
    pipeline = {
        'model': model,
        'scaler': scaler,
        'features': top_7_features,
        'bedrooms_imputation_value': bedrooms_median,
        'all_rankings': rankings
    }
    
    joblib.dump(pipeline, output_filename)
    print(f"\nSuccessfully saved pipeline dictionary to {output_filename}")
    
    # Verify we can load and use it
    loaded_pipeline = joblib.load(output_filename)
    print("\nVerification: Loaded pipeline keys:")
    print(list(loaded_pipeline.keys()))
    print("Top 7 features in loaded pipeline:")
    print(loaded_pipeline['features'])

if __name__ == "__main__":
    train_and_save_pipeline()

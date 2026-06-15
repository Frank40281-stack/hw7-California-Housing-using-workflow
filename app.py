# app.py
import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

from feature_selection import (
    download_data, 
    preprocess_data, 
    get_rankings, 
    evaluate_stepwise, 
    generate_plots, 
    NAME_MAP
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

st.set_page_config(
    page_title="California Housing Feature Selection stepwise Evaluation",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for premium glassmorphism and modern UI feel
st.markdown("""
<style>
    .main {
        background-color: #F8FAFC;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding-top: 10px;
        padding-bottom: 10px;
        font-size: 16px;
        font-weight: 600;
        background-color: transparent;
        border-radius: 4px;
        color: #64748B;
    }
    .stTabs [aria-selected="true"] {
        color: #1E3A8A !important;
        border-bottom: 3px solid #1E3A8A !important;
    }
    .css-1r6g72h {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 24px;
        box-shadow: 0 4px 6px -1px rgb(0 0 0 / 0.05), 0 2px 4px -2px rgb(0 0 0 / 0.05);
    }
    h1 {
        color: #0F172A;
        font-weight: 800;
    }
    h2 {
        color: #1E293B;
        font-weight: 700;
    }
    h3 {
        color: #334155;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# Cache data loading
@st.cache_data
def load_raw_data():
    return download_data()

# Cache full feature selection and evaluation pipeline
@st.cache_data
def run_pipeline(test_size, random_state, imputer_strategy):
    df = load_raw_data()
    df_clean = df.copy()
    
    # One-hot encode ocean_proximity
    df_clean = pd.get_dummies(df_clean, columns=['ocean_proximity'], dtype=float)
    
    target_col = 'median_house_value'
    features = [col for col in df_clean.columns if col != target_col]
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    
    # Imputation strategy based on sidebar input
    if imputer_strategy == "Median":
        train_bedrooms_val = X_train['total_bedrooms'].median()
    else:
        train_bedrooms_val = X_train['total_bedrooms'].mean()
        
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['total_bedrooms'] = X_train['total_bedrooms'].fillna(train_bedrooms_val)
    X_test['total_bedrooms'] = X_test['total_bedrooms'].fillna(train_bedrooms_val)
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    # Run rankings
    rankings = get_rankings(X_train_scaled, y_train.reset_index(drop=True))
    
    # Run evaluation
    r2_results, mse_results = evaluate_stepwise(
        X_train_scaled, y_train.reset_index(drop=True),
        X_test_scaled, y_test.reset_index(drop=True),
        rankings
    )
    
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True), rankings, r2_results, mse_results

def main():
    st.title("CRISP-DM Step 4: Feature Selection stepwise Evaluation Dashboard")
    st.markdown("Replicating the publication-quality stepwise evaluation of 9 feature selection algorithms for the **California Housing** dataset.")
    
    # Sidebar
    st.sidebar.header("Pipeline Configurations")
    test_size = st.sidebar.slider("Test Set Size Ratio", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    random_state = st.sidebar.number_input("Random State Seed", min_value=0, max_value=9999, value=42)
    imputer_strategy = st.sidebar.selectbox("Missing Value Imputation", ["Median", "Mean"])
    
    evaluator_type = st.sidebar.selectbox(
        "Evaluator Model Type", 
        ["Linear Regression", "Ridge Regression", "Lasso Regression", "Random Forest"]
    )
    
    st.sidebar.markdown("""
    ---
    ### 9 Algorithms Evaluated:
    1. Pearson Correlation
    2. Spearman Correlation
    3. F-test Regression
    4. Mutual Information
    5. RFE (Recursive Feature Elimination)
    6. SFS (Sequential Forward Selection)
    7. SBS (Sequential Backward Selection)
    8. Lasso (L1) Regularization
    9. Random Forest Importance
    """)
    
    # Run pipeline (cached)
    with st.spinner("Executing pipeline and feature selection algorithms..."):
        X_train, X_test, y_train, y_test, rankings, r2_results, mse_results = run_pipeline(
            test_size, random_state, imputer_strategy
        )
        
    # Implement custom evaluator model dynamically if changed in sidebar
    if evaluator_type != "Linear Regression":
        # Recalculate stepwise scores for different evaluator
        r2_results = {}
        mse_results = {}
        with st.spinner(f"Re-evaluating stepwise performance using {evaluator_type}..."):
            for alg, ranking in rankings.items():
                r2_list = []
                mse_list = []
                for k in range(1, 14):
                    features_k = ranking[:k]
                    
                    if evaluator_type == "Ridge Regression":
                        model = Ridge(alpha=1.0)
                    elif evaluator_type == "Lasso Regression":
                        model = Lasso(alpha=1.0)
                    elif evaluator_type == "Random Forest":
                        model = RandomForestRegressor(n_estimators=30, random_state=42, n_jobs=-1)
                    else:
                        model = LinearRegression()
                        
                    model.fit(X_train[features_k], y_train)
                    preds = model.predict(X_test[features_k])
                    
                    r2_list.append(r2_score(y_test, preds))
                    mse_list.append(mean_squared_error(y_test, preds))
                r2_results[alg] = r2_list
                mse_results[alg] = mse_list

    # Main Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 stepwise Evaluation", 
        "🔍 Algorithm Details", 
        "🔮 Interactive Predictor", 
        "🗺️ Dataset Explorer"
    ])
    
    with tab1:
        st.header("Stepwise Evaluation & Feature Rankings")
        st.markdown("The chart below displays the evaluation metrics (Test $R^2$ and Test MSE) across varying feature subset sizes (1 to 13), alongside the full ranking table.")
        
        # Save output image
        img_path = "california_housing_feature_selection.png"
        generate_plots(rankings, r2_results, mse_results, img_path)
        
        if os.path.exists(img_path):
            st.image(img_path, use_container_width=True)
            
            with open(img_path, "rb") as file:
                btn = st.download_button(
                    label="Download High-Res Figure (PNG)",
                    data=file,
                    file_name="california_housing_feature_selection.png",
                    mime="image/png"
                )
        else:
            st.error("Error generating figures.")
            
    with tab2:
        st.header("Feature Selection Details")
        st.markdown("Explore individual feature rankings and metrics for each algorithm.")
        
        selected_alg = st.selectbox("Select Feature Selection Algorithm", list(rankings.keys()))
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader(f"Feature Ranking for {selected_alg}")
            df_rank = pd.DataFrame({
                'Rank': range(1, 14),
                'Feature (Original)': rankings[selected_alg],
                'Abbreviation': [NAME_MAP.get(f, f) for f in rankings[selected_alg]],
                'Test R2 (at k)': r2_results[selected_alg],
                'Test MSE (at k)': mse_results[selected_alg]
            })
            st.dataframe(df_rank.style.highlight_max(subset=['Test R2 (at k)'], color='#C6F6D5'), hide_index=True)
            
        with col2:
            st.subheader("Metrics Trend")
            # Draw line chart for selected algorithm
            fig_trend, (ax_r2, ax_mse) = plt.subplots(2, 1, figsize=(8, 7), sharex=True)
            x_vals = range(1, 14)
            
            ax_r2.plot(x_vals, r2_results[selected_alg], color='#1E3A8A', marker='o', linewidth=2)
            ax_r2.set_ylabel("Test R-squared ($R^2$)", color='#1E3A8A')
            ax_r2.grid(True, linestyle='--', alpha=0.5)
            ax_r2.set_title(f"{selected_alg} Performance Metrics", fontsize=12, fontweight='bold')
            
            ax_mse.plot(x_vals, mse_results[selected_alg], color='#B91C1C', marker='s', linewidth=2)
            ax_mse.set_ylabel("Test Mean Squared Error (MSE)", color='#B91C1C')
            ax_mse.set_xlabel("Number of Features in Model")
            ax_mse.set_xticks(x_vals)
            ax_mse.grid(True, linestyle='--', alpha=0.5)
            
            st.pyplot(fig_trend)
            
    with tab3:
        st.header("Interactive House Value Predictor")
        st.markdown("Use this tab to train a model using your selected algorithm and subset size $k$, then interactively predict house prices!")
        
        pred_alg = st.selectbox("Select Algorithm to use for Predictor", list(rankings.keys()), key="pred_alg")
        k_features = st.slider("Select Feature Subset Size (k)", min_value=1, max_value=13, value=4)
        
        features_to_use = rankings[pred_alg][:k_features]
        
        # Display features being used
        st.info(f"Model is trained using the top {k_features} features: {', '.join([NAME_MAP.get(f, f) for f in features_to_use])}")
        
        # Train model on original unscaled inputs (for easier user input sliding)
        # However, to be mathematically consistent, we can scale inputs inside the predictor
        df_raw = load_raw_data()
        df_raw_clean = df_raw.copy().dropna(subset=['total_bedrooms'])
        df_raw_clean = pd.get_dummies(df_raw_clean, columns=['ocean_proximity'], dtype=float)
        
        # Split original inputs
        X_raw = df_raw_clean[[col for col in df_raw_clean.columns if col != 'median_house_value']]
        y_raw = df_raw_clean['median_house_value']
        
        X_raw_train, X_raw_test, y_raw_train, y_raw_test = train_test_split(
            X_raw, y_raw, test_size=test_size, random_state=random_state
        )
        
        # Impute
        bedrooms_impute_val = X_raw_train['total_bedrooms'].median() if imputer_strategy == "Median" else X_raw_train['total_bedrooms'].mean()
        X_raw_train = X_raw_train.copy()
        X_raw_test = X_raw_test.copy()
        X_raw_train['total_bedrooms'] = X_raw_train['total_bedrooms'].fillna(bedrooms_impute_val)
        X_raw_test['total_bedrooms'] = X_raw_test['total_bedrooms'].fillna(bedrooms_impute_val)
        
        # Fits scaling internally
        scaler_pred = StandardScaler()
        X_raw_train_scaled = pd.DataFrame(scaler_pred.fit_transform(X_raw_train), columns=X_raw_train.columns)
        X_raw_test_scaled = pd.DataFrame(scaler_pred.transform(X_raw_test), columns=X_raw_test.columns)
        
        # Fit model
        if evaluator_type == "Ridge Regression":
            model_pred = Ridge(alpha=1.0)
        elif evaluator_type == "Lasso Regression":
            model_pred = Lasso(alpha=1.0)
        elif evaluator_type == "Random Forest":
            model_pred = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
        else:
            model_pred = LinearRegression()
            
        model_pred.fit(X_raw_train_scaled[features_to_use], y_raw_train)
        
        # Predict on test
        preds_test = model_pred.predict(X_raw_test_scaled[features_to_use])
        test_r2_val = r2_score(y_raw_test, preds_test)
        test_rmse_val = np.sqrt(mean_squared_error(y_raw_test, preds_test))
        
        # Display current metrics
        m_col1, m_col2 = st.columns(2)
        m_col1.metric("Model Test R-squared (R²)", f"{test_r2_val:.4f}")
        m_col2.metric("Model Test RMSE ($)", f"${test_rmse_val:,.2f}")
        
        # Generate inputs
        st.subheader("Input House Attributes:")
        input_values = {}
        
        # We group ocean proximity dummy columns if they are present in top k
        ocean_dummies_in_use = [f for f in features_to_use if f.startswith("ocean_proximity_")]
        numerical_in_use = [f for f in features_to_use if not f.startswith("ocean_proximity_")]
        
        # Render numerical inputs
        col_inputs = st.columns(min(3, len(numerical_in_use) + (1 if ocean_dummies_in_use else 0)))
        
        col_idx = 0
        for feat in numerical_in_use:
            with col_inputs[col_idx % len(col_inputs)]:
                # Get min, max, default from raw training set
                min_val = float(X_raw_train[feat].min())
                max_val = float(X_raw_train[feat].max())
                mean_val = float(X_raw_train[feat].mean())
                
                # Format slider cleanly
                label_name = NAME_MAP.get(feat, feat).replace("_", " ").title()
                input_values[feat] = st.slider(
                    label_name, 
                    min_value=min_val, 
                    max_value=max_val, 
                    value=mean_val,
                    format="%.4f" if feat == "median_income" else "%.1f"
                )
            col_idx += 1
            
        # If ocean proximity features are in use, let the user select a category
        if ocean_dummies_in_use:
            with col_inputs[col_idx % len(col_inputs)]:
                # Extract original category names
                available_categories = [cat.replace("ocean_proximity_", "") for cat in ocean_dummies_in_use]
                # Add default if some are missing in subset to make selection comprehensive
                all_cats = ["<1H OCEAN", "INLAND", "NEAR OCEAN", "NEAR BAY", "ISLAND"]
                selected_cat = st.selectbox("Ocean Proximity", all_cats)
                
                # Map categories to input variables
                for dummy_feat in [f"ocean_proximity_{c}" for c in all_cats]:
                    if dummy_feat in features_to_use:
                        input_values[dummy_feat] = 1.0 if dummy_feat == f"ocean_proximity_{selected_cat}" else 0.0
                        
        # Predict house value
        # Create input df matching the shape of train set, then scale it, then select features_to_use
        input_row = pd.DataFrame([X_raw_train.mean()], columns=X_raw_train.columns)
        for feat, val in input_values.items():
            input_row[feat] = val
            
        # Scale row using scaler_pred
        input_scaled = pd.DataFrame(scaler_pred.transform(input_row), columns=input_row.columns)
        input_features = input_scaled[features_to_use]
        
        prediction = model_pred.predict(input_features)[0]
        
        # Display prediction beautifully
        st.markdown("---")
        st.subheader("Predicted Median House Value:")
        st.markdown(f"<h1 style='color:#1E3A8A; text-align:center;'>${prediction:,.2f}</h1>", unsafe_allow_html=True)
        
    with tab4:
        st.header("Dataset Explorer")
        st.markdown("Summary statistics, correlation mapping, and geospatial visualization of the raw housing data.")
        
        df_raw = load_raw_data()
        
        col_stat1, col_stat2 = st.columns([1, 1])
        
        with col_stat1:
            st.subheader("Raw Data Sample (First 100 rows)")
            st.dataframe(df_raw.head(100))
            
        with col_stat2:
            st.subheader("Descriptive Statistics")
            st.dataframe(df_raw.describe())
            
        # Map of California House listings colored by price
        st.subheader("Geospatial Distribution of California Housing Listings")
        st.markdown("Each bubble represents a housing district. Color shows the house value, size represents population.")
        
        # Drop rows with missing lat/long
        map_df = df_raw.dropna(subset=['latitude', 'longitude']).copy()
        # Scale down size for better rendering
        map_df['size_population'] = map_df['population'] / 10.0
        
        # Display streamlit map
        st.map(
            map_df,
            latitude='latitude',
            longitude='longitude',
            size='size_population',
            color='median_house_value'
        )
        
        # Correlation Heatmap
        st.subheader("Correlation Heatmap of Features (Numerical)")
        fig_heat, ax_heat = plt.subplots(figsize=(10, 6))
        num_cols = df_raw.select_dtypes(include=[np.number]).columns
        corr_matrix = df_raw[num_cols].corr()
        sns.heatmap(corr_matrix, annot=True, cmap='RdBu_r', fmt='.2f', ax=ax_heat, vmin=-1, vmax=1)
        st.pyplot(fig_heat)

if __name__ == "__main__":
    main()

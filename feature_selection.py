# feature_selection.py
import os
import urllib.request
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV
from sklearn.feature_selection import f_regression, mutual_info_regression, RFE
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error

# Short name mapping for clean table representation
NAME_MAP = {
    'longitude': 'long',
    'latitude': 'lat',
    'housing_median_age': 'house_age',
    'total_rooms': 'rooms',
    'total_bedrooms': 'bedrooms',
    'population': 'population',
    'households': 'households',
    'median_income': 'income',
    'ocean_proximity_<1H OCEAN': 'prox_<1h',
    'ocean_proximity_INLAND': 'prox_inland',
    'ocean_proximity_ISLAND': 'prox_island',
    'ocean_proximity_NEAR BAY': 'prox_near_bay',
    'ocean_proximity_NEAR OCEAN': 'prox_near_ocean'
}

def download_data(filename="housing.csv"):
    url = "https://raw.githubusercontent.com/ageron/handson-ml2/master/datasets/housing/housing.csv"
    if not os.path.exists(filename):
        print(f"Downloading dataset from {url}...")
        urllib.request.urlretrieve(url, filename)
        print("Download complete.")
    else:
        print("Using cached dataset.")
    return pd.read_csv(filename)

def preprocess_data(df):
    print("Preprocessing data...")
    # Impute missing values in total_bedrooms with the median
    # We do median calculation on the whole dataset here for simplicity,
    # but in train/test split we'll ensure no leakage
    df_clean = df.copy()
    
    # One-hot encode ocean_proximity
    # This expands the feature space with 5 binary columns
    df_clean = pd.get_dummies(df_clean, columns=['ocean_proximity'], dtype=float)
    
    # Reorder columns so that target is at the end
    target_col = 'median_house_value'
    features = [col for col in df_clean.columns if col != target_col]
    
    X = df_clean[features]
    y = df_clean[target_col]
    
    # 80/20 Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Impute missing total_bedrooms using the training median
    train_bedrooms_median = X_train['total_bedrooms'].median()
    X_train = X_train.copy()
    X_test = X_test.copy()
    X_train['total_bedrooms'] = X_train['total_bedrooms'].fillna(train_bedrooms_median)
    X_test['total_bedrooms'] = X_test['total_bedrooms'].fillna(train_bedrooms_median)
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
    X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, y_train.reset_index(drop=True), y_test.reset_index(drop=True)

# Custom Forward Selection
def forward_selection(X, y):
    print("Running SFS (Forward)...")
    selected = []
    remaining = list(X.columns)
    ranking = []
    
    while remaining:
        best_score = -float('inf')
        best_feat = None
        for feat in remaining:
            candidate = selected + [feat]
            # Fast evaluation using 5-fold CV
            score = cross_val_score(LinearRegression(), X[candidate], y, cv=5, scoring='r2').mean()
            if score > best_score:
                best_score = score
                best_feat = feat
        selected.append(best_feat)
        remaining.remove(best_feat)
        ranking.append(best_feat)
    return ranking

# Custom Backward Elimination
def backward_elimination(X, y):
    print("Running SBS (Backward)...")
    selected = list(X.columns)
    ranking = []
    
    while len(selected) > 1:
        worst_score = -float('inf')
        worst_feat = None
        for feat in selected:
            candidate = [f for f in selected if f != feat]
            score = cross_val_score(LinearRegression(), X[candidate], y, cv=5, scoring='r2').mean()
            if score > worst_score:
                worst_score = score
                worst_feat = feat
        ranking.append(worst_feat)
        selected.remove(worst_feat)
    
    ranking.append(selected[0])
    ranking.reverse()
    return ranking

def get_rankings(X_train, y_train):
    rankings = {}
    
    # 1. Pearson Correlation
    print("Running Pearson Correlation...")
    pearson = X_train.corrwith(y_train).abs()
    rankings['Pearson Corr'] = pearson.sort_values(ascending=False).index.tolist()
    
    # 2. Spearman Correlation
    print("Running Spearman Correlation...")
    spearman = X_train.apply(lambda col: col.corr(y_train, method='spearman')).abs()
    rankings['Spearman Corr'] = spearman.sort_values(ascending=False).index.tolist()
    
    # 3. F-test Regression
    print("Running F-test Regression...")
    F, _ = f_regression(X_train, y_train)
    rankings['F-test Reg'] = pd.Series(F, index=X_train.columns).sort_values(ascending=False).index.tolist()
    
    # 4. Mutual Information
    print("Running Mutual Information...")
    mi = mutual_info_regression(X_train, y_train, random_state=42)
    rankings['Mutual Info'] = pd.Series(mi, index=X_train.columns).sort_values(ascending=False).index.tolist()
    
    # 5. RFE (Linear Regression)
    print("Running RFE...")
    rfe_selector = RFE(LinearRegression(), n_features_to_select=1)
    rfe_selector.fit(X_train, y_train)
    rankings['RFE'] = pd.Series(rfe_selector.ranking_, index=X_train.columns).sort_values().index.tolist()
    
    # 6. SFS (Forward)
    rankings['SFS (Forward)'] = forward_selection(X_train, y_train)
    
    # 7. SBS (Backward)
    rankings['SBS (Backward)'] = backward_elimination(X_train, y_train)
    
    # 8. Lasso (L1)
    print("Running Lasso (L1)...")
    lasso = LassoCV(cv=5, random_state=42).fit(X_train, y_train)
    coefs = pd.Series(np.abs(lasso.coef_), index=X_train.columns)
    # Tie-breaking with Pearson Correlation for robust 13-feature ranking
    pearson_corr = X_train.corrwith(y_train).abs()
    df_lasso = pd.DataFrame({'lasso': coefs, 'pearson': pearson_corr})
    rankings['Lasso (L1)'] = df_lasso.sort_values(by=['lasso', 'pearson'], ascending=[False, False]).index.tolist()
    
    # 9. Random Forest
    print("Running Random Forest...")
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    rankings['Random Forest'] = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False).index.tolist()
    
    return rankings

def evaluate_stepwise(X_train, y_train, X_test, y_test, rankings):
    print("Evaluating stepwise regression...")
    r2_results = {}
    mse_results = {}
    
    for alg, ranking in rankings.items():
        r2_list = []
        mse_list = []
        for k in range(1, 14):
            features_k = ranking[:k]
            model = LinearRegression()
            model.fit(X_train[features_k], y_train)
            preds = model.predict(X_test[features_k])
            
            r2 = r2_score(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            r2_list.append(r2)
            mse_list.append(mse)
            
        r2_results[alg] = r2_list
        mse_results[alg] = mse_list
        
    return r2_results, mse_results

def generate_plots(rankings, r2_results, mse_results, output_path="california_housing_feature_selection.png"):
    print("Generating combined evaluation plots and table...")
    # Compute Best (Frontier)
    best_r2 = []
    best_mse = []
    
    for k_idx in range(13):
        r2_at_k = [r2_results[alg][k_idx] for alg in r2_results]
        mse_at_k = [mse_results[alg][k_idx] for alg in mse_results]
        best_r2.append(max(r2_at_k))
        best_mse.append(min(mse_at_k))
        
    # Programmatic Sweet Spot: Smallest k where Frontier R2 is within 1.5% of max Frontier R2
    max_frontier_r2 = max(best_r2)
    sweet_spot = 1
    for k in range(1, 14):
        if best_r2[k-1] >= max_frontier_r2 - 0.015:
            sweet_spot = k
            break
    print(f"Sweet spot identified at k = {sweet_spot}")

    # Set up styling and gridspec
    plt.rcParams['font.sans-serif'] = 'Arial'
    plt.rcParams['font.family'] = 'sans-serif'
    
    fig = plt.figure(figsize=(16, 12), dpi=150)
    gs = gridspec.GridSpec(2, 2, height_ratios=[1.2, 1.0], hspace=0.3)
    
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    
    # 9 Distinct colors + Best Frontier
    colors = {
        'Pearson Corr': '#1f77b4',       # Muted Blue
        'Spearman Corr': '#ff7f0e',      # Muted Orange
        'F-test Reg': '#2ca02c',         # Muted Green
        'Mutual Info': '#d62728',        # Muted Red
        'RFE': '#9467bd',                # Muted Purple
        'SFS (Forward)': '#8c564b',      # Muted Brown
        'SBS (Backward)': '#e377c2',     # Muted Pink
        'Lasso (L1)': '#7f7f7f',         # Muted Gray
        'Random Forest': '#bcbd22',      # Muted Olive
        'Best (Frontier)': '#FF8F00'     # Bold Amber
    }
    
    markers = {
        'Pearson Corr': 'o',
        'Spearman Corr': 's',
        'F-test Reg': '^',
        'Mutual Info': 'D',
        'RFE': 'v',
        'SFS (Forward)': '<',
        'SBS (Backward)': '>',
        'Lasso (L1)': 'p',
        'Random Forest': '*',
        'Best (Frontier)': 'h'
    }
    
    x_vals = range(1, 14)
    
    # Plot R2
    for alg in r2_results:
        ax0.plot(x_vals, r2_results[alg], label=alg, color=colors[alg], marker=markers[alg], markersize=5, alpha=0.8, linewidth=1.5)
    ax0.plot(x_vals, best_r2, label='Best (Frontier)', color=colors['Best (Frontier)'], marker=markers['Best (Frontier)'], markersize=8, linewidth=3, zorder=10)
    ax0.axvline(x=sweet_spot, color='#8B0000', linestyle=':', linewidth=2, label=f'Sweet Spot (k={sweet_spot})')
    ax0.set_title("Test R-squared Score by Feature Subset Size", fontsize=14, fontweight='bold', pad=15)
    ax0.set_xlabel("Number of Features in Model", fontsize=12)
    ax0.set_ylabel("Test R-squared ($R^2$)", fontsize=12)
    ax0.set_xticks(x_vals)
    ax0.grid(True, linestyle='--', alpha=0.5)
    # Add Sweet Spot Text
    ax0.text(sweet_spot + 0.2, ax0.get_ylim()[0] + (ax0.get_ylim()[1] - ax0.get_ylim()[0])*0.05, f"Sweet Spot (k={sweet_spot})", color='#8B0000', fontsize=11, fontweight='bold')
    
    # Plot MSE
    for alg in mse_results:
        # Scale MSE to Billions of Dollars Squared or leave in numerical format. 
        # House prices in California dataset are in dollars (e.g. 450,000). 
        # Standard scaled target MSE is in standard units. Let's make it look clean.
        ax1.plot(x_vals, mse_results[alg], label=alg, color=colors[alg], marker=markers[alg], markersize=5, alpha=0.8, linewidth=1.5)
    ax1.plot(x_vals, best_mse, label='Best (Frontier)', color=colors['Best (Frontier)'], marker=markers['Best (Frontier)'], markersize=8, linewidth=3, zorder=10)
    ax1.axvline(x=sweet_spot, color='#8B0000', linestyle=':', linewidth=2, label=f'Sweet Spot (k={sweet_spot})')
    ax1.set_title("Test MSE by Feature Subset Size", fontsize=14, fontweight='bold', pad=15)
    ax1.set_xlabel("Number of Features in Model", fontsize=12)
    ax1.set_ylabel("Test Mean Squared Error (MSE)", fontsize=12)
    ax1.set_xticks(x_vals)
    ax1.grid(True, linestyle='--', alpha=0.5)
    ax1.text(sweet_spot + 0.2, ax1.get_ylim()[0] + (ax1.get_ylim()[1] - ax1.get_ylim()[0])*0.85, f"Sweet Spot (k={sweet_spot})", color='#8B0000', fontsize=11, fontweight='bold')
    
    # Put Legend on top right plot or center bottom
    ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10, frameon=True, facecolor='#f8f9fa')
    
    # Prepare ranking table data
    table_data = []
    columns_list = ['Rank', 'Pearson', 'Spearman', 'F-test', 'Mutual Info', 'RFE', 'SFS (Fwd)', 'SBS (Bwd)', 'Lasso (L1)', 'Random Forest']
    
    for rank_idx in range(13):
        row = [f"Rank {rank_idx + 1}"]
        for alg in rankings:
            feat_name = rankings[alg][rank_idx]
            short_name = NAME_MAP.get(feat_name, feat_name)
            row.append(short_name)
        table_data.append(row)
        
    # Draw Table in the bottom span of gridspec
    ax_table = fig.add_subplot(gs[1, :])
    ax_table.axis('off')
    
    # Matplotlib Table rendering
    tbl = ax_table.table(
        cellText=table_data,
        colLabels=columns_list,
        loc='center',
        cellLoc='center'
    )
    
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1.0, 1.8) # scale heights of cells
    
    # Style cells
    for (row_idx, col_idx), cell in tbl.get_celld().items():
        cell.set_edgecolor('#CBD5E1')
        if row_idx == 0:
            # Header Row
            cell.set_facecolor('#1A365D') # Dark blue
            cell.get_text().set_color('white')
            cell.get_text().set_weight('bold')
            cell.get_text().set_fontsize(11)
        else:
            # Body Rows alternating color
            if row_idx % 2 == 1:
                cell.set_facecolor('#F8FAFC') # Soft white/grey
            else:
                cell.set_facecolor('#EDF2F7') # Soft grey-blue
                
            # Highlight sweet spot features for SFS / RFE / Random Forest if desired,
            # or keep it standard. Let's keep it standard but clean.
            cell.get_text().set_color('#2D3748')
            
    fig.suptitle("CRISP-DM Step 4: 9 Feature Selection Algorithms stepwise Evaluation (California Housing)", fontsize=18, fontweight='bold', y=0.96)
    
    # Save high-res figure
    plt.savefig(output_path, bbox_inches='tight', dpi=150)
    print(f"Plot saved to {output_path}")
    plt.close()

def main():
    df = download_data()
    X_train, X_test, y_train, y_test = preprocess_data(df)
    rankings = get_rankings(X_train, y_train)
    r2_res, mse_res = evaluate_stepwise(X_train, y_train, X_test, y_test, rankings)
    generate_plots(rankings, r2_res, mse_res)
    print("Core feature selection execution finished successfully!")

if __name__ == "__main__":
    main()

# generate_workflow.py
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_workflow():
    steps = [
        "1. Data Ingestion\n(Download housing.csv from GitHub)",
        "2. Data Imputation\n(Impute total_bedrooms using Train Median)",
        "3. One-Hot Encoding\n(ocean_proximity dummy variables)",
        "4. Train/Test Split\n(80% Train, 20% Test)",
        "5. Standard Scaling\n(StandardScaler fit on Train, transform both)",
        "6. Feature Selection (9 Algorithms)\n(Pearson, Spearman, F-test, MI, RFE, SFS, SBS, Lasso, RF)",
        "7. Stepwise Subset Loop\n(For k = 1 to 13 features)",
        "8. Model Evaluation\n(Train Linear Reg on top k, Evaluate Test R2 & MSE)",
        "9. Sweet Spot & Plot Generation\n(Draw trend charts and ranking table)",
        "10. Streamlit Dashboard Server\n(Launch web app for interactive serving)"
    ]

    fig, ax = plt.subplots(figsize=(9, 13), dpi=150)
    ax.set_xlim(0, 10)
    ax.set_ylim(0.5, 11.5)
    ax.axis('off')

    # Add light-colored boxes flowing vertically
    for i, step in enumerate(steps):
        y = 11.0 - i * 1.05
        
        # Color coding based on process stage
        if i in [0, 1, 2, 3, 4]:
            color = '#EBF8FF'       # Soft Blue (Prep)
            edgecolor = '#3182CE'
        elif i == 5:
            color = '#E6FFFA'       # Soft Green (Feature Selection)
            edgecolor = '#319795'
        elif i == 6:
            color = '#FEFCBF'       # Soft Yellow (Loop)
            edgecolor = '#D69E2E'
        elif i == 7:
            color = '#FFF5F5'       # Soft Red (Evaluation)
            edgecolor = '#E53E3E'
        else:
            color = '#FAF5FF'       # Soft Purple (Output & Serve)
            edgecolor = '#805AD5'

        # Draw rounded rectangle
        box = patches.FancyBboxPatch(
            (1.5, y - 0.35), 7.0, 0.7, 
            boxstyle="round,pad=0.1", 
            linewidth=2, 
            edgecolor=edgecolor, 
            facecolor=color
        )
        ax.add_patch(box)
        
        # Draw centered text
        ax.text(5.0, y, step, ha='center', va='center', fontsize=9.5, fontweight='bold', color='#2D3748')
        
        # Draw arrow to the next box (if not the last one)
        if i < len(steps) - 1:
            ax.annotate(
                '', 
                xy=(5.0, y - 0.7), 
                xytext=(5.0, y - 0.35),
                arrowprops=dict(
                    arrowstyle="->", 
                    color='#4A5568', 
                    lw=2, 
                    mutation_scale=15
                )
            )

    plt.title("California Housing Feature Selection Pipeline Workflow", fontsize=14, fontweight='bold', pad=15, color='#1A202C')
    plt.savefig("california_housing_workflow.png", bbox_inches='tight', dpi=150)
    print("Workflow image saved successfully as california_housing_workflow.png")

if __name__ == "__main__":
    draw_workflow()

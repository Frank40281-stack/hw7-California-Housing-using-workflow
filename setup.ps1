# setup.ps1
Write-Host "Installing scikit-learn, matplotlib, and seaborn..." -ForegroundColor Cyan
pip install scikit-learn matplotlib seaborn

Write-Host "Verifying Python environment..." -ForegroundColor Cyan
python -c "
try:
    import sklearn
    import matplotlib
    import seaborn
    import pandas
    import numpy
    print('All required libraries imported successfully!')
except Exception as e:
    print('Verification failed:', e)
"

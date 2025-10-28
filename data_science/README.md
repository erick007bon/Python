# Data Science Algorithms

This directory contains implementations of common data science algorithms and techniques for educational purposes.

## Modules

### 1. Data Preprocessing

#### `normalization.py`
Techniques for scaling data to a fixed range:
- **Min-Max Normalization**: Scales data to [0, 1] range
- **Decimal Scaling Normalization**: Moves decimal point to scale data

#### `standardization.py`
Techniques for transforming data to have zero mean and unit variance:
- **Z-Score Standardization**: Centers data around zero with std=1
- **Robust Standardization**: Uses median and IQR (robust to outliers)

#### `label_encoding.py`
Methods for encoding categorical data:
- **Label Encoding**: Converts categories to integers
- **Label Decoding**: Converts integers back to categories
- **One-Hot Encoding**: Creates binary vectors for categories

### 2. Statistical Analysis

#### `statistical_analysis.py`
Common statistical measures and tests:
- **Descriptive Statistics**: mean, median, mode, variance, standard deviation
- **Quartiles and IQR**: Q1, Q2, Q3, interquartile range
- **Correlation**: Pearson correlation coefficient, covariance

### 3. Outlier Detection

#### `outlier_detection.py`
Methods for identifying anomalous data points:
- **IQR Method**: Uses interquartile range to detect outliers
- **Z-Score Method**: Uses standard deviations from mean
- **Modified Z-Score Method**: More robust version using median and MAD
- **Remove Outliers**: Utility function to clean data

### 4. Feature Engineering

#### `feature_selection.py`
Techniques for selecting relevant features:
- **Variance Threshold**: Removes low-variance features
- **Correlation-based Selection**: Selects features correlated with target
- **Mutual Information**: Measures dependence between variables

#### `dimensionality_reduction.py`
Methods for reducing feature dimensions:
- **PCA (Principal Component Analysis)**: Unsupervised linear transformation
- **LDA (Linear Discriminant Analysis)**: Supervised linear transformation

### 5. Time Series Analysis

#### `time_series_analysis.py`
Methods for analyzing temporal data:
- **Simple Moving Average (SMA)**: Smooths data with equal weights
- **Exponential Moving Average (EMA)**: Gives more weight to recent data
- **Weighted Moving Average (WMA)**: Custom weights for smoothing
- **Trend Analysis**: Calculate and remove linear trends
- **Seasonal Decomposition**: Separate trend, seasonal, and residual components

## Usage Examples

### Normalization
```python
from data_science.normalization import min_max_normalization

data = [10, 20, 30, 40, 50]
normalized = min_max_normalization(data)
print(normalized)  # [0.0, 0.25, 0.5, 0.75, 1.0]
```

### Statistical Analysis
```python
from data_science.statistical_analysis import mean, standard_deviation

data = [12, 15, 18, 20, 22, 25, 28, 30, 35, 40]
print(f"Mean: {mean(data)}")
print(f"Std Dev: {standard_deviation(data)}")
```

### Outlier Detection
```python
from data_science.outlier_detection import iqr_outlier_detection

data = [10, 12, 14, 15, 16, 18, 20, 22, 24, 200]
indices, values = iqr_outlier_detection(data)
print(f"Outliers: {values}")  # [200.0]
```

### Feature Selection
```python
from data_science.feature_selection import variance_threshold_selector

data = [[0, 2, 0, 3], [0, 1, 4, 3], [0, 1, 1, 3]]
selected, indices = variance_threshold_selector(data, threshold=0.0)
print(f"Selected features: {indices}")  # [1, 2, 3]
```

### Dimensionality Reduction
```python
from data_science.dimensionality_reduction import pca

data = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
transformed, components, variance = pca(data, n_components=2)
print(f"Reduced dimensions: {transformed.shape}")
```

### Time Series Analysis
```python
from data_science.time_series_analysis import simple_moving_average

data = [10, 12, 15, 14, 18, 20, 22, 25, 28, 30]
sma = simple_moving_average(data, window_size=3)
print(sma)
```

## Running Tests

Each module includes doctests that can be run individually:

```bash
python3 -m doctest data_science/normalization.py -v
python3 -m doctest data_science/standardization.py -v
python3 -m doctest data_science/statistical_analysis.py -v
# ... and so on
```

Or run the module directly to see example outputs:

```bash
python3 data_science/normalization.py
python3 data_science/outlier_detection.py
```

## Dependencies

- NumPy: For numerical computations and array operations

## Educational Purpose

These implementations are designed for learning and understanding the algorithms. For production use, consider using established libraries like:
- scikit-learn (feature selection, dimensionality reduction, preprocessing)
- pandas (data manipulation and analysis)
- statsmodels (statistical analysis)
- scipy (scientific computing)

## Contributing

When adding new algorithms:
1. Follow the existing code style and structure
2. Include comprehensive docstrings with parameter descriptions
3. Add doctests with examples
4. Include a `__main__` block with usage examples
5. Update this README with the new functionality

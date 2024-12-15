
# Instructions for Reproducing Results

To replicate the results presented in this work, please follow the steps outlined below:

## 1. Data Preparation
Ensure that the `data` folder is appropriately placed within the `code` directory.

## 2. Environment Setup
Create a Python virtual environment using Python version 3.11. Install the required dependencies specified in the `requirements.txt` file using the command:

```bash
pip install -r requirements.txt
```

## 3. Execution of the Notebook
Open the `reconstruction.ipynb` notebook and execute all sections sequentially to reproduce the results.

## 4. Modifying the Interpolation Method
To explore alternative interpolation techniques, update the interpolation method in the `create_standard_metrics` function. The available options include:
- `'interp1d'`: A piecewise linear interpolation.
- `'cubicspline'`: A cubic spline interpolation method.
- `'polyfit'`: A polynomial fitting method.

Choose the desired method by specifying it within the function.

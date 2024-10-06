import sys
print(sys.executable)


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import TimeSeriesSplit

df = pd.read_csv("../../data/Ohio2020_processed/train/540-ws-training_processed.csv")


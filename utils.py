import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib

FILE_PATH = "quantum_security_data.csv"
MODEL_PATH = "quantum_sentinel_model.pkl"
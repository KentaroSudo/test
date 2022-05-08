import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

sns.set()

# Display title
st.title('ボストン市の住宅価格の重回帰分析')

# Load dataset
dataset = load_boston()
# Create a explanately variables in pandas Dataframe type
df = pd.DataFrame(dataset.data)
# assigne lounm name from explanatery variables name
df.columns = dataset.feature_names
# Create the objective variable with the column name "PRICE"
df["PRICES"] = dataset.target 



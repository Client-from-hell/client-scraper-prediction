import pandas as pd
import numpy as np
from functools import reduce
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.externals import joblib
import streamlit as st
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold

df = pd.read_csv('C:/Users/Ironhack/Documents/Ironhack/Project clients from hell/dataframe_words.csv')
models = [LogisticRegression, RandomForestClassifier, MultinomialNB]

st.title("Balance of the target")
plt.hist(df['deadbeats'], bins = 2, rwidth = 0.5)
st.pyplot()

def pipe(obj, *fns):
    return reduce(lambda x, y: y(x), [obj] + list(fns))

def rescale_numbers(df, scaler):
    for col in df:
        if df[col].dtype in ['int64', 'float64']:
            numbers = df[col].astype(float).values.reshape(-1, 1)
            df[col] = scaler().fit_transform(numbers)
    return df

def preprocess(df):
    return (df.pipe(rescale_numbers, MinMaxScaler))

def train_test(df, target):
    return train_test_split(
        df[[col for col in df if col != target]],
        df['target'],
        test_size = .2,
        random_state = 42
    )

def evaluate_model(algorithm, train_test):
    train_X, test_X, train_y, test_y = train_test
    model = algorithm().fit(train_X, train_y)
    pred_proba_y = model.predict_proba(test_X)
    try:
        auc = roc_auc_score(test_y, pred_proba_y[:, 1])
        st.subheader('AUC Score')
        st.write(auc)
    except ValueError:
        pass
    f, t, _ = roc_curve(test_y, pred_proba_y[:, 1])
    st.subheader('AUC Graph')
    plt.plot(f, t)
    st.pyplot()
    score = model.score(test_X, test_y)
    st.write(f"Accuracy: {round(score, 2)}")
    return model, score

def k_fold(df, target):
    scores = []
    features = df[[col for col in df if col != target]]
    target = df[target]
    for model in models:
        st.title(model)
        for train_i, test_i in KFold(n_splits=5, random_state=42).split(df):
            scores.append(evaluate_model(
            model,
            (features.iloc[train_i], features.iloc[test_i], target.iloc[train_i], target.iloc[test_i])
        )[1])
        st.title("Average Model Score")
        st.write(sum(scores) / len(scores))
        #print(model, sum(scores)/len(scores)) 

k_fold(preprocess(df), target = 'deadbeats')



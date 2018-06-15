#!/usr/local/bin/env python3
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

KAGGLE_DIR = "/home/ubuntu/.kaggle/competitions/titanic"

def format_data(df):
    df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)
    return pd.get_dummies(df)

def main():
    df_org = pd.read_csv(f"{KAGGLE_DIR}/train.csv")
    test_df_org = pd.read_csv(f"{KAGGLE_DIR}/test.csv")
    
    all_df = pd.concat([df_org, test_df_org])
    all_df = format_data(all_df)
    
    df = all_df.iloc[:df_org.shape[0], :]
    test_df = all_df.iloc[df_org.shape[0]:, :]
    
    df = df.dropna()
    test_df = test_df.drop(["Survived"], axis=1).interpolate()
    
    x = df.drop(["Survived"], axis=1)
    y = df.Survived

    train_x, test_x, train_y, test_y = \
        train_test_split(x, y, test_size=0.2, random_state=None)

    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(train_x, train_y)
    print(clf.score(test_x, test_y) )
    
    pred_y = clf.predict(test_df)
    pred_y = np.concatenate([test_df_org.PassengerId.values.reshape(-1,1), pred_y.reshape(-1,1)], axis=1)
    np.savetxt("./answer.csv", pred_y, fmt="%.0f", delimiter=",", header="PassengerId,Survived")

if __name__=='__main__':
    main()
    
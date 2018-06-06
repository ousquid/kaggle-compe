#!/usr/local/bin/env python3
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

KAGGLE_DIR = "/home/ubuntu/.kaggle/competitions/titanic"

def format_data(df):
    df = df.drop(["PassengerId", "Name", "Ticket"], axis=1)
    df = df.dropna()
    return pd.get_dummies(df)

def main():
    df = pd.read_csv(f"{KAGGLE_DIR}/train.csv")
    df = format_data(df)
    
    x = df.drop(["Survived"], axis=1)
    y = df.Survived
    
    train_x, test_x, train_y, test_y = \
        train_test_split(x, y, test_size=0.2, random_state=None)

    clf = SGDClassifier(loss="hinge", penalty="l2")
    clf.fit(train_x, train_y)
    scr = clf.score(test_x, test_y)
    print(scr)
    
if __name__=='__main__':
    main()
    
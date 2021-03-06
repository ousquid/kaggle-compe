#!/usr/local/bin/env python3
import re
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split

KAGGLE_DIR = "/home/ubuntu/.kaggle/competitions/titanic"

def format_cabin(data):
    if pd.isnull(data):
        return "Z"
    else:
        return data[0]

def format_ticket(data):
    regex = re.compile('[^a-zA-Z]')
    ticket_str = regex.sub('',data)
    return ticket_str and ticket_str or "UNKNOWN"

def format_age(data):
    if pd.isnull(data):
        return 29
    else:
        return data

def format_sex(data):
    return data.split(",")[1].split(" ")[1]
    
def format_name(data):
    return ")" in data
    

def main():
    df_org = pd.read_csv(f"{KAGGLE_DIR}/train.csv")
    test_df_org = pd.read_csv(f"{KAGGLE_DIR}/test.csv")
    
    df = pd.concat([df_org, test_df_org])
    
    # -----
    
    # cabin -> alphabet or "unknown":Z
    df["Cabin"] = df["Cabin"].apply(format_cabin)
    
    # ticket -> sign or "unknown":Z
    df["Ticket"] = df["Ticket"].apply(format_ticket)
    
    # age -> "unknown":mean
    df["Age"] = df["Age"].apply(format_age)
    
    # sex -> name "mr, miss, mrs"
    df["Sex"] = df["Name"].apply(format_sex)
    
    # delete name
    df["Name"] = df["Name"].apply(format_name)
    
    df["Fare"] = df["Fare"].fillna(15)
    
    # -----
    
    df = pd.get_dummies(df)
    df = df.drop(["PassengerId"], axis=1)

    test_df = df.iloc[df_org.shape[0]:, :]
    df = df.iloc[:df_org.shape[0], :]
    
    test_df = test_df.drop(["Survived"], axis=1)
    
    x = df.drop(["Survived"], axis=1)
    y = df.Survived

    train_x, test_x, train_y, test_y = \
        train_test_split(x, y, test_size=0.0, random_state=666)

    from sklearn.svm import SVC
    #clf = SVC(kernel='linear', C=40.0, random_state=0)
    clf = SVC(kernel='rbf', C=6000, gamma=0.00005, random_state=0)
    clf.fit(train_x, train_y)
    
    #from sklearn.model_selection import GridSearchCV
    #from sklearn.pipeline import Pipeline

    #log_range = [0.0001, 0.01, 0.1, 1.0, 10.0, 100.0]
    #log_grid = [{'gamma': log_range, 'C': log_range}]
    #iris_gs = GridSearchCV(estimator = SVC(kernel='rbf', random_state=0),
    #              param_grid = log_grid, cv=10, n_jobs=-1)
    #iris_gs.fit(train_x, train_y)
    #print("cross validation accuracy:{}".format(iris_gs.best_score_) )
    #print("best gamma:{}".format(iris_gs.best_params_['gamma']) )
    #print("best C:{}".format(iris_gs.best_params_['C']) )


    #clf = SGDClassifier(loss="hinge", penalty="l2")
    #clf.fit(train_x, train_y)
    #print(clf.score(test_x, test_y) )
    
    pred_y = clf.predict(test_df)
    pred_y = np.concatenate([test_df_org.PassengerId.values.reshape(-1,1), pred_y.reshape(-1,1)], axis=1)
    np.savetxt("./answer.csv", pred_y, fmt="%.0f", delimiter=",", header="PassengerId,Survived")

if __name__=='__main__':
    main()
    
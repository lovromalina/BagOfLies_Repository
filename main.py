import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import random


df = pd.read_csv("gaze_df_1.csv")

X = df.drop('truth', axis=1)
y = df['truth']

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

avg_accuracy = 0
sum_accuracy = 0

for i in range(1, 2):

    random_int = random.randint(0, 1000)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_int)
    #print("X_train: ", X_train.shape, "y_train: ", y_train.shape)
    #print(X_train)


    rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=3, random_state=random_int)


    rf_classifier.fit(X_train, y_train)


    y_pred = rf_classifier.predict(X_test)

    # Calculate and print the accuracy
    accuracy = accuracy_score(y_test, y_pred)
    sum_accuracy += accuracy
    avg_accuracy = sum_accuracy / i
    print("Run: ", i)
    print("Accuracy:", accuracy)
    print("avg_accuracy:", avg_accuracy)
    print("____________________")

# Display the final DataFrame with gaze data
#print(final_np_dataframe)

#print("min rows:", min_row)


#print("Median original length of DataFrames:", median_original_length)


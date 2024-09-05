import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
import random



anno = pd.read_csv("Annotations.csv")

original_lengths = []
list_of_gaze_df = []
np_list_of_gaze_df = []
min_row = 1000


for gaze_path in anno['gaze']:
    
    gaze_dataframe = pd.read_csv(gaze_path)

    gaze_dataframe.drop("USER", axis=1, inplace=True)
    gaze_dataframe.drop("CS", axis=1, inplace=True)
    gaze_dataframe.drop("CY", axis=1, inplace=True)
    gaze_dataframe.drop("CX", axis=1, inplace=True)
    gaze_dataframe.drop("TIMETICK", axis=1, inplace=True)

    original_lengths.append(gaze_dataframe.shape[0])

    gaze_dataframe['CNT'] = pd.to_datetime(gaze_dataframe['CNT'], unit='s')
    gaze_dataframe.set_index('CNT', inplace=True)

    gaze_dataframe.interpolate(method='linear', inplace=True)

    target_rows = 540

    if gaze_dataframe.shape[0] < target_rows:
        resample_freq = pd.to_timedelta(gaze_dataframe.shape[0] / target_rows - 0.0015, unit='S')

        # Upsample and interpolate
        gaze_dataframe = gaze_dataframe.resample(resample_freq).ffill()
        # Limit to target_rows
        gaze_dataframe = gaze_dataframe.head(target_rows)
    else:
        resample_freq = pd.to_timedelta(gaze_dataframe.shape[0] / target_rows, unit='S')
        # Downsample
        gaze_dataframe = gaze_dataframe.resample(resample_freq).mean()
        gaze_dataframe = gaze_dataframe.head(target_rows)

    np_gaze_dataframe = gaze_dataframe.to_numpy()

    np_gaze_dataframe = np_gaze_dataframe.flatten()

    if gaze_dataframe.shape[0] < min_row:
        min_row = gaze_dataframe.shape[0]

    # Append the DataFrame to the list
    list_of_gaze_df.append(np_gaze_dataframe)
    

    column_names = gaze_dataframe.columns


final_dataframe = pd.DataFrame(data=list_of_gaze_df)

final_dataframe['truth'] = anno['truth']


median_original_length = np.median(original_lengths)



X = final_dataframe.drop('truth', axis=1)
y = final_dataframe['truth']

avg_accuracy = 0
sum_accuracy = 0

imputer = SimpleImputer(strategy='mean')
X = imputer.fit_transform(X)

random_int = random.randint(0, 1000)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_int)
#print("X_train: ", X_train.shape, "y_train: ", y_train.shape)
#print(X_train)



# Create a RandomForestClassifier
rf_classifier = RandomForestClassifier(n_estimators=300, max_depth=4, random_state=random_int)

# Train the classifier on the training data
rf_classifier.fit(X_train, y_train)

# Predict the labels on the testing data
y_pred = rf_classifier.predict(X_test)

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)
#print("avg_accuracy:", avg_accuracy)
print("____________________")

# Display the final DataFrame with gaze data
#print(final_dataframe)

#print("min rows:", min_row)

#X_train_imputed_pd = pd.DataFrame(X_train_imputed)

#print("Median original length of DataFrames:", median_original_length)

#!___________Za napraviti novi file nemoj dirati_________________
#!
#!final_dataframe.to_csv("gaze_df_1.csv", index=False)

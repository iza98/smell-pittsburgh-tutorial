# This is a reusable function to print the data
# (no need to modify this part)
def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")

# Import the "preprocessData" function in the "preprocessData.py" script for reuse
# (no need to modify this part)
from preprocessData import preprocessData

# Preprocess and print sensor and smell data
# (no need to modify this part)
df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/","dataset/smell_raw.csv"])
pretty_print(df_sensor, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")

# Select some variables, which means the columns in the data table.
# (you may want to modify this part to add more variables for experiments)
# (you can also comment out the following two lines to indicate that you want all variables)

wanted_cols = ["3.feed_24.PM10_UG_M3", "3.feed_28.SONICWS_MPH", "3.feed_28.SONICWD_DEG", "3.feed_27.CO_PPB", "3.feed_23.PM10_UG_M3",
               "3.feed_3.SO2_PPM", "3.feed_26.SONICWS_MPH", "3.feed_26.OZONE_PPM", "3.feed_28.SIGTHETA_DEG",
               "3.feed_3506.OZONE", "DateTime", "3.feed_23.CO_PPM", "3.feed_1.SONICWS_MPH", "3.feed_3.SONICWD_DEG",
               "3.feed_26.SIGTHETA_DEG", "3.feed_26.SONICWD_DEG", "3.feed_3.PM10B_UG_M3", "3.feed_1.SO2_PPM", 
               "3.feed_1.SONICWD_DEG", "3.feed_27.SO2_PPB", "3.feed_3.SONICWS_MPH", "3.feed_1.SIGTHETA_DEG",
               "3.feed_27.NOY_PPB", "3.feed_11067.NOX_PPB..3.feed_43.NOX_PPB", "3.feed_1.PM25B_UG_M3..3.feed_1.PM25T_UG_M3"]

# wanted_cols = ["3.feed_24.PM10_UG_M3", "3.feed_28.SONICWS_MPH", "3.feed_27.CO_PPB",
#                "3.feed_23.PM10_UG_M3", "3.feed_3.SO2_PPM", "3.feed_26.SONICWS_MPH",
#                "3.feed_26.OZONE_PPM", "DateTime", "3.feed_23.CO_PPM", "3.feed_1.SONICWS_MPH", 
#                "3.feed_3.SONICWD_DEG"]

# wanted_cols = ["3.feed_24.PM10_UG_M3", "DateTime", "3.feed_27.CO_PPB", "3.feed_26.OZONE_PPM", "3.feed_1.SONICWS_MPH"]

# wanted_cols = ["DateTime", "3.feed_24.PM10_UG_M3"]

df_sensor = df_sensor[wanted_cols]

# Print the selected sensor data
# (no need to modify this part)
pretty_print(df_sensor, "Display selected sensor data and column names")

# Import the "computeFeatures" function in the "computeFeatures.py" script for reuse
# (no need to modify this part)
from computeFeatures import computeFeatures

# Indicate the threshold to define a smell event
# (you may want to modify this parameter for experiments)
smell_thr = 40

# Indicate the number of future hours to predict smell events
# (you may want to modify this parameter for experiments)
smell_predict_hrs = 6

# Indicate the number of hours to look back to check previous sensor data
# (you may want to modify this parameter for experiments)
look_back_hrs = 0

# Indicate if you want to add interaction terms in the features (like x1*x2)
# (you may want to modify this parameter for experiments)
add_inter = False

# Compute and print features (X) and response (Y)
# (no need to modify this part)
df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
        f_hr=smell_predict_hrs, b_hr=look_back_hrs, thr=smell_thr, add_inter=add_inter)
pretty_print(df_X, "Display features (X) and column names")
pretty_print(df_Y, "Display response (Y) and column names")

# Import packages for reuse
# (you may want to import more models)
from util import scorer
from util import printScores
from util import createSplits

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

import matplotlib.pyplot as plt

def train_model(model, X_train, Y_train):
  # Train the model on the training data
  model.fit(X_train, Y_train)
  return model

# Calculate test size as 20% of the total data
test_size = 168

# Calculate train size as 80% of the total data
train_size = 336

"""For K-NN"""
# Set the range of k values to search over
param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]}

# Create a StandardScaler for scaling the data
scaler = StandardScaler()

# Scale the feature data using the scaler
X_scaled = scaler.fit_transform(df_X)

best_score = -1
best_params = None

# Build the cross validation splits
# (no need to modify this part)
splits = createSplits(test_size, train_size, df_X.shape[0])

"""
For K-NN
"""
# Set the range of k values to search over
model_param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}

# Create a GridSearchCV instance to find the best k value
model_grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=model_param_grid, cv=splits, scoring=scorer, refit = 'f1')

# For the grid search to your scaled training data
model_grid_search.fit(X_scaled, df_Y.squeeze())

# Get the best k value from the grid search results
best_k = model_grid_search.best_params_['n_neighbors']

# Create a K-NN model with the best k value
model = KNeighborsClassifier(n_neighbors=best_k)
print("Use model with best k value:", model)
print(best_k)

"""
For Logistic Regression
"""
# from sklearn.linear_model import LogisticRegression

# # Define hyperparameters for Logistic Regression
# param_grid = {
#     'C': [ 0.1, 1, 10, 100],  # Regularization strength
#     'penalty': ['l1', 'l2'],    # Regularization type
# }

# # Create the Logistic Regression model
# logistic_reg = LogisticRegression(solver='liblinear')

# # Create a GridSearchCV instance to find the best hyperparameters
# grid_search = GridSearchCV(estimator=logistic_reg, param_grid=param_grid, cv=splits, scoring=scorer, refit='f1')

# # Fit the grid search to your scaled training data
# grid_search.fit(X_scaled, df_Y.squeeze())

# # Get the best hyperparameters from the grid search results
# best_C = grid_search.best_params_['C']
# best_penalty = grid_search.best_params_['penalty']

# # Create a Logistic Regression model with the best hyperparameters
# model = LogisticRegression(C=best_C, penalty=best_penalty, solver='liblinear')
# print("Use model with best hyperparameters:", model)

"""
For Support Vector Machine
"""
# from sklearn.svm import SVC

# # Define hyperparameters for the SVM
# param_grid = {
#     'C': [1],  # Regularization strength
#     'kernel': ['linear', 'rbf', 'poly'],  # Kernel types
#     # Add other SVM hyperparameters here if needed
# }

# # Create the SVM model with default hyperparameters
# svm = SVC()

# # Create a GridSearchCV instance to find the best hyperparameters
# grid_search = GridSearchCV(estimator=svm, param_grid=param_grid, cv=splits, scoring=scorer, refit='f1')

# # Fit the grid search to your scaled training data
# grid_search.fit(X_scaled, df_Y.squeeze())

# # Get the best hyperparameters from the grid search results
# best_C = grid_search.best_params_['C']
# best_kernel = grid_search.best_params_['kernel']

# # Create an SVM model with the best hyperparameters
# model = SVC(C=best_C, kernel=best_kernel)
# print("Use model with best hyperparameters:", model)

"""
For Gradient Boosting
"""
# from xgboost import XGBClassifier

# # Define hyperparameters for the XGBoost
# param_grid = {
#     'n_estimators': [100],
#     'max_depth': [3, 4, 5, 6],
#     'learning_rate': [0.05, 0.1, 0.20, 0.3]
# }

# # Create a XGBoost model with default hyperparameters
# xgboost = XGBClassifier(objective='binary:logistic')

# # Create a GridSearchCV instance to find the best hyperparameters
# grid_search = GridSearchCV(estimator=xgboost, param_grid=param_grid, cv=splits, scoring=scorer, refit='f1')

# # Fit the grid search to the scaled training data
# grid_search.fit(X_scaled, df_Y.squeeze())

# # Get the best hyperparameters from the grid search results
# best_n_estimators = grid_search.best_params_['n_estimators']
# best_max_depth = grid_search.best_params_['max_depth']
# best_learning_rate = grid_search.best_params_['learning_rate']

# # Create a XGBoost model with the best hyperparameters
# model = XGBClassifier(
#     objective='binary:logistic',  # For binary classification
#     n_estimators=best_n_estimators,             # Number of boosting rounds
#     max_depth=best_max_depth,                  # Maximum tree depth
#     learning_rate=best_learning_rate           # Learning rate
# )

# Perform cross-validation to evaluate the model
# (no need to modify this part)
print("Use model", model)
print("Perform cross-validation, please wait...")
result = cross_validate(model, df_X, df_Y.squeeze(), cv=5, scoring=scorer)
model.predict
printScores(result)
      
# Import packages for reuse
# (no need to modify this part)
from util import computeFeatureImportance

# Compute and show feature importance weights
# (no need to modify this part)
# feature_importance = computeFeatureImportance(df_X, df_Y, scoring="f1")
# pretty_print(feature_importance, "Display feature importance based on f1-score")

# from plot_ConfusionMatrix import plot_confusion_matrix

# class_labels = ["Non-Odor", "Odor"]
# plot_confusion_matrix(y_true, y_pred, class_labels)

# from plot_ROCurve import plot_ROC_curve

# plot_ROC_curve(y_true, y_pred)

import numpy as np
from sklearn.model_selection import learning_curve
    
# def plot_learning_curve(model, X, y, train_sizes, cv=5):
     
#     skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=42)
#     skf.get_n_splits(X, y)

#     for i, (train_index, test_index) in enumerate(skf.split(X, y)):
#         X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#         y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
#     _, train_scores, test_scores = learning_curve(
#             model, X_train, y_train, train_sizes=train_sizes, cv=None, scoring='f1'
#         )

#     train_scores_mean = np.mean(train_scores, axis=1)
#     test_scores_mean = np.mean(test_scores, axis=1)

#     plt.figure()
#     plt.title("Learning Curve")
#     plt.xlabel("Training Size")
#     plt.ylabel("f1")
#     plt.grid()

#     plt.plot(train_sizes, train_scores_mean, 'o-', label="Training f1")
#     plt.plot(train_sizes, test_scores_mean, 'o-', label="Cross-validation f1")

#     plt.legend(loc="best")
#     plt.show()

# train_sizes = np.linspace(0.1, 0.8, 20) 

# plot_learning_curve(model, df_X, df_Y.squeeze(), train_sizes, cv=5)


# import matplotlib.pyplot as plt
# import numpy as np

# def plot_learning_curve(model, df_X, df_Y, train_size, test_sizes, scaler, param_grid):
#     f1_scores = []

#     for test_size in test_sizes:
#         print(f"Test Size: {test_size}, Train Size: {train_size}")

#         # Build the cross validation splits
#         splits = createSplits(test_size, train_size, df_X.shape[0])

#         # Create a GridSearchCV instance to find the best k value
#         grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=param_grid, cv=5, scoring=scorer, refit='f1')

#         # Scale the feature data using the scaler
#         X_scaled = scaler.fit_transform(df_X)

#         # For the grid search, use your scaled training data
#         grid_search.fit(X_scaled, df_Y.squeeze())

#         # Get the best k value from the grid search results
#         best_k = grid_search.best_params_['n_neighbors']

#         # Create a K-NN model with the best k value
#         model = KNeighborsClassifier(n_neighbors=best_k)

#         # Perform cross-validation to evaluate the model
#         result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
#         test_f1 = result['test_f1'].mean()
#         f1_scores.append(test_f1)

#     # Plot the learning curve
#     plt.figure()
#     plt.title("Learning Curve")
#     plt.xlabel("Training Size")
#     plt.ylabel("F1 Score")

#     plt.grid()

#     plt.plot(test_sizes, f1_scores, 'o-', label="F1 Score")

#     plt.legend(loc="best")
#     plt.show()

# # Define your parameters
# train_size = 10752
# test_sizes = np.linspace(100, 4000, 20, dtype=int) 

# # Call the function to plot the learning curve
# plot_learning_curve(model, df_X, df_Y, train_size, test_sizes, scaler, param_grid)

# def plot_learning_curve(model, df_X, df_Y, test_size, train_sizes, scaler):
#     f1_scores = []
#     for train_size in train_sizes:
#         print(f"Train Size: {train_size}, Test Size: {test_size}")

#         # Build the cross validation splits
#         splits = createSplits(test_size, train_size, df_X.shape[0])

#         # model = KNeighborsClassifier(n_neighbors=best_k)

#         # Perform cross-validation to evaluate the model
#         result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
#         test_f1 = round(np.mean(result["test_f1"]), 2)
#         print(round(np.mean(result["test_f1"]), 2), result['test_f1'].mean())
#         f1_scores.append(test_f1)

#     # Plot the learning curve
#     plt.figure()
#     plt.title("Learning Curve")
#     plt.xlabel("Training Size")
#     plt.ylabel("F1 Score")
#     plt.grid()
#     plt.plot(train_sizes, f1_scores, 'o-', label="F1 Score")
#     plt.legend(loc="best")
#     plt.show()

# # Define your parameters
# test_size = 168
# train_sizes = train_sizes = np.linspace(336, 16000, 20, dtype = int) 

# # Call the function to plot the learning curve
# plot_learning_curve(model, df_X, df_Y, test_size, train_sizes, scaler)

"""
# Test for sizes
# """
# test_size_values = [168, 336, 504]
# train_size_values = [336, 840, 1344, 2688, 5376, 8064, 8760, 10752, 12600]
# import numpy as np

# for test_size in test_size_values:
#     for train_size in train_size_values:
#         # Create cross-validation splits
#         splits = createSplits(test_size, train_size, df_X.shape[0])

#          # Grid search for the model's hyperparameters
#         model_param_grid = {"n_neighbors": [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21]}
#         model_grid_search = GridSearchCV(estimator=KNeighborsClassifier(), param_grid=model_param_grid, cv=splits, scoring=scorer, refit='f1')
#         model_grid_search.fit(df_X, df_Y.squeeze())
#         best_k = model_grid_search.best_params_['n_neighbors']
#         model = KNeighborsClassifier(n_neighbors=best_k)

#         # perform cross-validation to evaluate to model
#         print(f"Testing with test_size={test_size} and train_size={train_size}, model={model}...")

#         results = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer)
#         mean_score=round(np.mean(results["test_f1"]), 2)
#         print ("f1:", mean_score)

#         if mean_score > best_score:
#             best_score = mean_score
#             best_params = {'test_size': test_size, 'train_size': train_size}
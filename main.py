"""
This script contains the pipeline for predicting the presence of bad smell.

The dataset that we will use is from the following URL:
- https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction

For the background of this project, please check the following paper:
- https://arxiv.org/pdf/1912.11936.pdf

Our task is to train a model (denoted F) to predict presence of bad smell.
We will use the term "smell event" to indicate the "presence of bad smell".
The model (F) maps a set of features (X) to a response (Y), where Y=F(X).
The features are extracted from the raw data from air quality and weather sensors.
The response means if there will be a bad smell event in the future.

Now, how to define a bad smell event?
We define it as if the sum of smell ratings within a time range exceeds a threshold.
This reflects if many people reports bad smell ratings within a future time range.
Details will be explained as you move forward to read this script.

The following is a brief description of the pipeline:
- Step 1: Preprocess the raw data
- Step 2: Select variables from the preprocessed sensor data
- Step 3: Extract features (X) and the response (Y) from the preprocessed data
- Step 4: Train and evaluate a machine learning model (F) that maps X to Y
- Step 5: Investigate the importance of each feature
"""


# This is a reusable function to print the data
# (no need to modify this part)
def pretty_print(df, message):
    print("\n================================================")
    print("%s\n" % message)
    print(df)
    print("\nColumn names below:")
    print(list(df.columns))
    print("================================================\n")


"""
Step 1: Preprocess the raw data

We want to preprocess sensor and smell data and get the intermediate results.
This step does the following:
- Load the sensor and smell data in the dataset folder
- Merge the sensor data from different air quality and weather monitoring stations
- Align the timestamps in the sensor and smell data by resampling the data points
- Treat missing data

The returned variable "df_sensor" means the preprocessed sensor data.
The DateTime column means the timestamp.
Other columns mean the average value of the sensor data in the previous hour.
For example, "2016-10-31 06:00:00+00:00" means October 31 in 2016 at 6AM UTC time.
Column "3.feed_1.SO2_PPM" means the averaged SO2 values from 5AM to 6AM.
The column name suffix SO2 means sulfur dioxide, and PPM means the unit.
The prefix "3.feed_1." in the column name means a specific sensor (feed ID 1).
You can ignore the "3." at the begining of the column name.
Here is what it means for each feed ID:
- Feed 26: Lawrenceville ACHD
- Feed 28: Liberty ACHD
- Feed 23: Flag Plaza ACHD
- Feed 43: Parkway East ACHD
- Feed 11067: Parkway East Near Road ACHD
- Feed 1: Avalon ACHD
- Feed 27: Lawrenceville 2 ACHD
- Feed 29: Liberty 2 ACHD
- Feed 3: North Braddock ACHD
- Feed 3506: BAPC 301 39TH STREET BLDG AirNow
- Feed 5975: Parkway East AirNow
- Feed 3508: South Allegheny High School AirNow
- Feed 24: Glassport High Street ACHD

You can search the metadata of the feed by searching the feed ID in the following URL:
- https://environmentaldata.org/

Some column names look like "3.feed_11067.SIGTHETA_DEG..3.feed_43.SIGTHETA_DEG".
This means that the column has data from two sensor stations (feed ID 11067 and 43).
The reason is that some sensor stations are replaced by the new ones over time.
So in this case, we merge sensor readings from both feed ID 11067 and 43.
Here is a list of the explanation of column name suffix:
- SO2_PPM: sulfur dioxide in ppm (parts per million)
- SO2_PPB: sulfur dioxide in ppb (parts per billion)
- H2S_PPM: hydrogen sulfide in ppm
- SIGTHETA_DEG: standard deviation of the wind direction
- SONICWD_DEG: wind direction (the direction from which it originates) in degrees
- SONICWS_MPH: wind speed in mph (miles per hour)
- CO_PPM: carbon monoxide in ppm
- CO_PPB: carbon monoxide in ppb
- PM10_UG_M3: particulate matter (PM10) in micrograms per cubic meter
- PM10B_UG_M3: same as PM10_UG_M3
- PM25_UG_M3: fine particulate matter (PM2.5) in micrograms per cubic meter
- PM25T_UG_M3: same as PM25_UG_M3
- PM2_5: same as PM25_UG_M3
- PM25B_UG_M3: same as PM25_UG_M3
- NO_PPB: nitric oxide in ppb
- NO2_PPB: nitrogen dioxide in ppb
- NOX_PPB: sum of of NO and NO2 in ppb 
- NOY_PPB: sum of all oxidized atmospheric odd-nitrogen species in ppb
- OZONE_PPM: ozone (or trioxygen) in ppm
- OZONE: same as OZONE_PPM

More explanation about the suffix is in the following URL:
- https://tools.wprdc.org/pages/air-quality-docs.html

The returned variable "df_smell" means the preprocessed smell data.
[TODO: explain what each column mean]

Both "df_sensor" and "df_smell" use the pandas.DataFrame data structure.
More information about the data structure is in the following URL:
- https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe
"""
# Import the "preprocessData" function in the "preprocessData.py" script for reuse
# (no need to modify this part)
from preprocessData import preprocessData

# Preprocess and print sensor and smell data
# (no need to modify this part)
df_sensor, df_smell = preprocessData(in_p=["dataset/esdr_raw/","dataset/smell_raw.csv"])
pretty_print(df_sensor, "Display all sensor data and column names")
pretty_print(df_smell, "Display smell data and column names")


"""
Step 2: Select variables from the preprocessed sensor data

We may want to pick a subset of the sensor data instead of using all of them.
Our intuition is that H2S emissions near the pollution source may be a good variable.
So we first select "3.feed_28.H2S_PPM"
, which is the H2S measurement from a monitoring station at the south of Pittsburgh.

Can you improve the model performance by adding more variables?
You can play with the selection of variables and see how it affects model performance.

Hint 1: try adding the "3.feed_11067.SONICWD_DEG..3.feed_43.SONICWD_DEG" variable
It is the wind direction measured from a monitoring station at the east of Pittsburgh.

Hint 2: try adding the "3.feed_26.SONICWD_DEG" variable
It is the wind direction measured from a monitoring station at the middle part of Pittsburgh.

Hint 3: try using more variables
Does using more variables help increase model performance?
"""
# Select some variables, which means the columns in the data table.
# (you may want to modify this part to add more variables for experiments)
# (you can also comment out the following two lines to indicate that you want all variables)
wanted_cols = ["DateTime", "3.feed_28.H2S_PPM", "3.feed_11067.SONICWD_DEG..3.feed_43.SONICWD_DEG"]
df_sensor = df_sensor[wanted_cols]

# Print the selected sensor data
# (no need to modify this part)
pretty_print(df_sensor, "Display selected sensor data and column names")


"""
Step 3: Extract features (X) and the response (Y) from the preprocessed data

We want to extract features (X) for predicting smell events.
We also want to extract the response (Y) which indicates smell events.
This step does the following:
- Compute the features based on the "look_back_hrs" parameter
- Compute the response based on the "smell_predict_hrs" and "smell_thr" parameters

Parameter "smell_thr" is the threshold to define a smell event.
If the sum of smell ratings is larger than this threshold
, the model will think that there will be a smell event.
For example, if we set this threshold to 40
, this may mean that there are 10 people, and each of them reported smell rating 4.

Parameter "smell_predict_hrs" is the number of future hours to predict smell events.
For example, setting it to 8 means to predict the smell event in the future 8 hours.
If it is 12:00 now, the model predicts if smell events will happen between 12:00 and 20:00.
Keep in mind that the prediction is binary (yes or no).

Parameter "look_back_hrs" is the number of hours to look back to check previous sensor data
The sensor data includes air quality and weather data from deployed sensors.
For example, setting it to 2 means to check the sensor data in the previous 2 hours.
If it is 12:00 now, the model will use sensor data from 10:00 to 12:00 to predict smell events.

The returned variable "df_X" means the features (X).
[TODO: explain what the columns mean]

The returned variable "df_Y" means the response (Y).
Response value 0 means no smell events in the future.
[TODO: explain what the columns mean]

Both "df_X" and "df_Y" also use the pandas.DataFrame data structure.
If you forgot what is a DataFrame, check the following URL:
- https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe

Now, here are several questions for you, indicated below:

Is threshold 40 (sum of smell ratings) good for defining a smell event?
Specifically, can you figure out a better (while also reasonable) threshold?
You can play with the "smell_thr" parameter and see how it affects model performance.

Is it possible that we can predict the smell events more precisely?
Specifically, can you figure out if we can predict smell events in a shorter future?
You can play with the "smell_predict_hrs" parameter and see how it affects model performance.

Do we believe that we only need to look back for one hour?
Specifically, can you figure out how many hours we need to look back?
You can play with the "look_back_hrs" parameter and see how it affects model performance.
"""
# Import the "computeFeatures" function in the "computeFeatures.py" script for reuse
# (no need to modify this part)
from computeFeatures import computeFeatures

# Indicate the threshold to define a smell event
# (you may want to modify this parameter for experiments)
smell_thr = 40

# Indicate the number of future hours to predict smell events
# (you may want to modify this parameter for experiments)
smell_predict_hrs = 8

# Indicate the number of hours to look back to check previous sensor data
# (you may want to modify this parameter for experiments)
look_back_hrs = 1

# Indicate if you want to add interaction terms in the features (like x1*x2)
# (you may want to modify this parameter for experiments)
add_inter = False

# Compute and print features (X) and response (Y)
# (no need to modify this part)
df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
        f_hr=smell_predict_hrs, b_hr=look_back_hrs, thr=smell_thr, add_inter=add_inter)
pretty_print(df_X, "Display features (X) and column names")
pretty_print(df_Y, "Display response (Y) and column names")


"""
Step 4: Train and evaluate a machine learning model (F) that maps X to Y

Currently we use a very naive model which always predicts "no" smell events.
Can you make the prediction result better by using other models?
(You can play with different models and check their performance.)

Hint 1: try the Decision Tree model
Remember to import the "DecisionTreeClassifier" model as in the following URL:
- https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

Hint 2: try the Random Forest model (which is a collection of Decision Tree models)
Remember to import the "RandomForestClassifier" model as in the following URL:
- https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html

Parameter "test_size" is the number of samples for testing.
For example, setting it to 168 means using 168 samples for testing
, which also means 168 hours (or 7 days) of data.

Parameter "train_size" is the number of samples for training.
For example, setting it to 336 means using 336 samples for testing
, which also means 336 hours (or 14 days) of data.

So, this means we are using previous 14 days of sensor data to train the model
, and then use the model to predict the smell events in the next 7 days.
For example, every Sunday we can re-train the model with the updated data
, so that we have the updated model to predict smell events every week.

But, do we really believe that 14 days of data is enough to train the model?
Specifically, can you figure out how much data is needed to train the model?
You can play with the "train_size" parameter and see how it affects model performance.

Also, do we believe that the model can be used for only 7 days?
Specifically, can you figure out how long does the model remain effective?
You can play with the "test_size" parameter and see how it affects model performance.
"""
# Import packages for reuse
# (you may want to import more models)
from util import scorer
from util import printScores
from util import createSplits
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import cross_validate

# Indicate how much data you want to use to test the model
# (you may want to modify this parameter for experiments)
test_size = 168

# Indicate how much data you want to use to train the model
# (you may want to modify this parameter for experiments)
train_size = 336

# Build the cross validation splits
# (no need to modify this part)
splits = createSplits(test_size, train_size, df_X.shape[0])

# Indicate which model you want to use to predict smell events
# (you may want to modify this part to use other models)
model = DummyClassifier(strategy="constant", constant=0)

# Perform cross-validation to evaluate the model
# (no need to modify this part)
print("Perform cross-validation, please wait...")
result = cross_validate(model, df_X, df_Y.squeeze(), cv=splits, scoring=scorer, n_jobs=-2)
printScores(result)


"""
Step 5: Investigate the importance of each feature

[TODO: explain how this part works]
"""
# Import packages for reuse
# (no need to modify this part)
from util import computeFeatureImportance

# Compute feature importance weights
# (no need to modify this part)
computeFeatureImportance(df_X, df_Y)

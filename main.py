"""
This script contains the pipeline for predicting the presence of bad smell.

The dataset that we will use is from the following URL:
- https://github.com/CMU-CREATE-Lab/smell-pittsburgh-prediction

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
- Step 4: Train a machine learning model (F) that maps the features to the response
- Step 5: Evaluate the performance of the machine learning model
"""


"""
Import packages

We are going to reuse the code that others already wrote.
Syntax "from preprocessData" means using the code in the "preprocessData.py" script.
Syntax "import preprocessData" means importing the "preprocessData" function.
By combining these two syntax we can import a specific function in a python script.
"""
from preprocessData import preprocessData
from computeFeatures import computeFeatures


"""
The threshold to define a smell event

If the sum of smell ratings is larger than this threshold
, the model will think that there will be a smell event.

(You can play with this parameter in the assignments.)
"""
smell_thr = 40


"""
The number of hours to predict smell events in the future

For example, 8 means to predict the smell event in the future 8 hours.
If the the current time is 12:00
, the model will predict if smell events will happen between 12:00 and 20:00.
The prediction is binary (yes or no).

(You can play with this parameter in the assignments.)
"""
smell_predict_hrs = 8


"""
The number of hours to look back to check the sensor data in the past

The sensor data includes air quality and weather data from deployed sensors.
For example, 2 means to check the sensor data in the previous 2 hours.
If the current time is 12:00
, the model will use sensor data from 10:00 to 12:00 to predict smell events.

(You can play with this parameter in the assignments.)
"""
look_back_hrs = 2


# This is a reusable function to print the data
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
(Ignore the "3." at the begining of the column name)
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
More explanation is in the following URL:
- https://tools.wprdc.org/pages/air-quality-docs.html

The returned variable "df_smell" means the preprocessed smell data.
[TODO: Explain what each columns mean]

Both "df_sensor" and "df_smell" use the pandas.DataFrame data structure.
More information about the data structure is in the following URL:
- https://pandas.pydata.org/docs/user_guide/dsintro.html#dataframe
"""
p1 = "dataset/" # path to the folder that contains the dataset
df_sensor, df_smell = preprocessData(in_p=[p1+"esdr_raw/",p1+"smell_raw.csv"])

# Print the sensor data
pretty_print(df_sensor, "Display all sensor data and column names")

# Print the smell data
pretty_print(df_smell, "Display smell data and column names")


"""
Step 2: Select variables from the preprocessed sensor data

Now we want to just pick a subset of the sensor data.
"""
# IMPORTANT: you must select "DateTime" since it is necessary for the next step.
wanted_cols = ["DateTime", "3.feed_28.SO2_PPM", "3.feed_28.SONICWD_DEG"]
df_sensor = df_sensor[wanted_cols]

# Print the selected sensor data
pretty_print(df_sensor, "Display selected sensor data and column names")


"""
Step 3: Extract features (X) and the response (Y) from the preprocessed data

We want to extract features (X) for predicting smell events.
We also want to extract the response (Y) which indicates smell events.
This step does the following:
- Compute the features based on the "look_back_hrs" parameter
- Compute the response based on the "smell_predict_hrs" and "smell_thr" parameters

The returned variable "df_X" means the features (X).
[TODO: explain the columns]

The returned variable "df_Y" means the response (Y).
Response value 0 means no smell events in the future.
[TODO: explain the columns]

Both "df_X" and "df_Y" also use the pandas.DataFrame data structure.
"""
df_X, df_Y, _ = computeFeatures(df_esdr=df_sensor, df_smell=df_smell,
        f_hr=smell_predict_hrs, b_hr=look_back_hrs, thr=smell_thr)

# Print features (X)
pretty_print(df_X, "Display features (X) and column names")

# Print response (Y)
pretty_print(df_Y, "Display response (Y) and column names")


"""
Step 4: Train a machine learning model (F) that maps the features to the response
"""
p2 = "temp/" # path to the folder that stores intermediate and final results

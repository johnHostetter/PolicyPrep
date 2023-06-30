The pre-processing steps for the data are as follows:

First, we must download the training data, which is stored by semesters on Google Drive.

Second, we must transform the format of that training data, into a format that is compatible 
with the InferNet library. This is done by the `data_to_infernet.py` script.

Third, we must aggregate the data into a single file, which is done by the `aggregate_data.py` 
script. This is done because the InferNet library requires that all data be in a single file.

Fourth, we must run the `data_to_infernet.py` script again, to transform the aggregated data
into a format that is compatible with the InferNet library.

Finally, we must split the data into training and testing sets, which is done by the
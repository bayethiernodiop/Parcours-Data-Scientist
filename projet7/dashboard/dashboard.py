# TO RUN : $streamlit run dashboard.py
# Local URL: http://localhost:8501
# Network URL: http://192.168.0.50:8501



import streamlit as st
# To make things easier later, we're also importing numpy and pandas for
# working with sample data.
import numpy
import pandas

st.write('Hello, world!')

x = st.slider('Select a value')
st.write(x, 'squared is', x * x)


##################################################
# Selecting applicant ID
select_sk_id = st.selectbox('Select SK_ID from list:', [100001, 100005], key=1)

##################################################
# Requesting the API
import requests
import json

# URL of the scoring API
# SK_ID_CURR = 100005
SCORING_API_URL = "http://127.0.0.1:5000/api/scoring/?SK_ID_CURR=" + str(select_sk_id)

# save the response to API request
response = requests.get(SCORING_API_URL)

# convert from JSON format to Python dict
content = json.loads(response.content.decode('utf-8'))

# getting the values from the content
st.write(content)
SK_ID_CURR = content['SK_ID_CURR']
score = content['score']
st.write('Score:', score)
st.write('Applicant ID:', SK_ID_CURR)


#############################################################
import streamlit as st
import pandas as pd

# Reuse this data across runs!
read_and_cache_csv = st.cache(pd.read_csv)

BUCKET = "https://streamlit-self-driving.s3-us-west-2.amazonaws.com/"
data = read_and_cache_csv(BUCKET + "labels.csv.gz", nrows=1000)
desired_label = st.selectbox('Filter to:', ['car', 'truck'])
st.write(data[data.label == desired_label])




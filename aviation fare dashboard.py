import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.preprocessing import StandardScaler
import joblib
from PIL import Image
from datetime import datetime, date

# Setting page configuration
st.set_page_config(page_title="Aviation flights fare", page_icon="✈️", layout='wide')

# Loading data
data = pd.read_csv('cleaned_df.csv')

with st.sidebar:
    
    st.sidebar.image('R.jpg')
    st.sidebar.subheader("This dashboard for Indian Aviation Flights Fare aimed at predicting the prices of flight tickets")
    st.sidebar.write("")
    
    # Inputs 
    col1, col2 = st.columns((5,5))
    source = st.sidebar.selectbox("Departure City", ['All'] + list(data['Source'].unique()))
    # data after Source selection
    if source != 'All':
        data = data[data['Source'] == source]
    
    destination = st.sidebar.selectbox("Arrival City", ['All'] + list(data['Destination'].unique()))
    # data after Source and Destination selection
    if destination != 'All':
        data = data[data['Destination'] == destination]
      
    month = st.sidebar.selectbox("Month", ['All'] + list(data['Month'].unique()))
    # data after Source and Destination and Month selection
    if month != 'All':
        data = data[data['Month'] == month]
    
    day = st.sidebar.selectbox("Day", ['All'] + list(data['Day'].unique()))
    # data after Source and Destination and Month and Day selection
    if day != 'All':
        data = data[data['Day'] == day]

    dep_hour = st.sidebar.selectbox("Departure Hour", ['All'] + list(data['Dep_Hour'].unique()))
    # data after Source and Destination and Month and Day selection and departure hour selection
    if dep_hour != 'All':
        data = data[data['Dep_Hour'] == dep_hour]

    airline = st.sidebar.selectbox("Airline Carrier", ['All'] + list(data['Airline'].unique()))
    # data after Source and Destination and Month and Day selection and departure hour selection and airline selection
    if airline != 'All':
        data = data[data['Airline'] == airline]

    duration = data[(data['Airline'] == airline) & (data['Source'] == source) & (data['Destination'] == destination)]
    
    add_info = st.sidebar.selectbox("Additional Services", ['All'] + list(data['Additional_Info'].unique()))
    

    st.sidebar.write("")
    st.sidebar.markdown("Made by Omayma Ali")

    # filtering Function
def filter(airline, source, destination, add_info, month, day, dep_hour):
    if airline=='All' and source=='All' and destination=='All' and add_info=='All' and month=='All' and day=='All' and dep_hour=='All':
        filtered_data = data.copy()
    else:
        filtered_data = data

        if source != 'All':
            filtered_data = filtered_data[filtered_data['Source'] == source]

        if destination != 'All':
            filtered_data = filtered_data[filtered_data['Destination'] == destination]

        if month != 'All':
            filtered_data = filtered_data[filtered_data['Month'] == month]
        
        if day != 'All':
            filtered_data = filtered_data[filtered_data['Day'] == day]

        if dep_hour != 'All':
            filtered_data = filtered_data[filtered_data['Dep_Hour'] == dep_hour]

        if airline != 'All':
            filtered_data = filtered_data[filtered_data['Airline'] == airline]

        if add_info != 'All':
            filtered_data = filtered_data[filtered_data['Additional_Info'] == add_info]
  
    return filtered_data

# Information Cards
card1, card2, card3, card4 = st.columns((2,2,2,4))

# Filtered DataFrame
filtered_data = filter(airline, source, destination, add_info, month, day, dep_hour)

# Cards Values
flight_count = filtered_data['Airline'].count()
highest_Price = filtered_data['Price'].max()
lowest_Price = filtered_data['Price'].min()
top_airline = filtered_data['Airline'].value_counts().idxmax()
# Show The Cards
card1.metric("Flight Count", f"{flight_count}")
card2.metric("Highest Price", f"{highest_Price}")
card3.metric("Lowest Price", f"{lowest_Price}")
card4.metric("Top Airline", f"{top_airline}")

# Dashboard Tabs
tab1, tab2 = st.tabs(["📈 Analyze", "🤖 Predict"])

# Data Analysis
with tab1:   
    visual1, visual2 = st.columns((5, 5))
    with visual1:
        st.subheader('The Most Airline Count')
        most_airline = filtered_data['Airline'].value_counts().sort_values(ascending=False).head()
        fig = px.pie(data_frame=most_airline, 
                     names=most_airline.index, 
                     values=most_airline.values, 
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    with visual2:
        st.subheader('Airline Price')
        airline_price = filtered_data.groupby(['Airline'])['Price'].min().sort_values(ascending=False).head(15)
        fig = px.bar(airline_price, 
                     x=airline_price.index, 
                     y=airline_price.values, 
                     color=airline_price.values)
         # Customize x-axis and y-axis labels
        fig.update_xaxes(title='Airline')
        fig.update_yaxes(title='Price')
        st.plotly_chart(fig, use_container_width=True) 

# predicting Model
with tab2:

    model = joblib.load('aviation_flight_fare_prediction_model.p')    
    sc = StandardScaler()

    # Inputs 
    col1, col2 = st.columns((5,5))
    with col1:
        airline_pred = st.selectbox("Airline Carrier", list(data['Airline'].unique()))
        airline_pred = {'IndiGo':3, 'Air India':1, 'Jet Airways':4, 'SpiceJet':8,
                'Multiple carriers':6, 'GoAir':2, 'Vistara':10, 'Air Asia':0,
                'Vistara Premium economy':11, 'Jet Airways Business':5,
                'Multiple carriers Premium economy':7, 'Trujet':9}.get(airline_pred, airline_pred)
        
        source_pred= st.selectbox("Departure City", list(data['Source'].unique()))    
        source_pred = {'Banglore':0, 'Kolkata':3, 'Delhi':2, 'Chennai':1, 'Mumbai':4}.get(source_pred, source_pred)
        
        destination_pred = st.selectbox("Arrival City", list(data['Source'].unique())) 
        destination_pred = {'New Delhi':5, 'Banglore':0, 'Cochin':1, 'Kolkata':4,
                    'Delhi':2, 'Hyderabad':3}.get(destination_pred, destination_pred)
        
        stops_pred= int(st.selectbox("Stops", options= data['Total_Stops'].unique()))

        duration_pred_scaled = sc.fit_transform([[int(st.number_input("Flight Duration (in minutes)",
                                                                    min_value=0, step=10))]])
    
    with col2:
                
        add_info_pred= st.selectbox("Additional Services", list(data['Additional_Info'].unique()))
        add_info_pred = {'No info':7, 'In-flight meal not included':5, 'No check-in baggage included':6,
                    '1 Short layover':1, '1 Long layover':0, 'Change airports':4,
                    'Business class':3, 'Red-eye flight':8, '2 Long layover':2}.get(add_info_pred, add_info_pred)

        day_pred= int(st.selectbox("Day", options= data['Day'].unique()))
        month_pred= int(st.selectbox("Month", options= data['Month'].unique()))

        st.write('If the minutes more than 30, Please increase the hour by 1')
        dep_hour_pred_scaled = sc.fit_transform([[int(st.number_input("Departure Hour (24 format)",
                                                                    min_value=0))]])

    # Submit Button
    if st.button("Submit 👇"):
        input_data = np.array([[duration_pred_scaled[0][0], stops_pred, day_pred, month_pred, 
                                dep_hour_pred_scaled[0][0], airline_pred, source_pred,
                                  destination_pred, add_info_pred]])

        Price = model.predict(input_data)
        st.write('The price for this ticket = ', int(Price)) 


    



        
    

      
 
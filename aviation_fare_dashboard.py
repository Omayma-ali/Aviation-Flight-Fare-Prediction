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
df = pd.read_csv('cleaned_df.csv')

with st.sidebar:

    st.sidebar.image('R.jpg')
    st.sidebar.subheader("This dashboard for Indian Aviation Flights Fare aimed at predicting the prices of flight tickets")
    st.sidebar.write("")
    
    data = df.copy()
    source = st.sidebar.selectbox("Departure City", ['All'] + list(data['Source'].unique()))
    # data after Source selection
    if source != 'All':
        data = data[data['Source'] == source]
    
    destination = st.sidebar.selectbox("Arrival City", ['All'] + list(data['Destination'].unique()))
    # data after Source and Destination selection
    if destination != 'All':
        data = data[data['Destination'] == destination]

    duration = data[(data['Source'] == source) & (data['Destination'] == destination)]
    
    airline = st.sidebar.selectbox("Airline Carrier", ['All'] + list(data['Airline'].unique()))
    # data after Source and Destination and Month and Day selection and departure hour selection and airline selection
    if airline != 'All':
        data = data[data['Airline'] == airline]

    add_info = st.sidebar.selectbox("Additional Services", ['All'] + list(data['Additional_Info'].unique()))

    filter_box = st.sidebar.selectbox("Filter Prices on", [None, 'Day', 'Month', 'Dep_Hour'])



    st.sidebar.write("")
    st.sidebar.markdown("Made by [Omayma Ali](https://github.com/Omayma-ali)")

    # filtering Function
def filter(airline, source, destination, add_info):
    if airline=='All' and source=='All' and destination=='All' and add_info=='All':
        filtered_data = data.copy()
    else:
        filtered_data = data

        if source != 'All':
            filtered_data = filtered_data[filtered_data['Source'] == source]

        if destination != 'All':
            filtered_data = filtered_data[filtered_data['Destination'] == destination]

        if airline != 'All':
            filtered_data = filtered_data[filtered_data['Airline'] == airline]

        if add_info != 'All':
            filtered_data = filtered_data[filtered_data['Additional_Info'] == add_info]

    return filtered_data

# Information Cards
card1, card2, card3, card4 = st.columns((2,2,2,4))

# Filtered DataFrame
filtered_data = filter(airline, source, destination, add_info)

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
tab1, tab2, tab3 = st.tabs(["🏠 Home", "📈 Insights", "🤖 Prediction"])
# introduction
with tab1:
    st.write("If you are a traveler looking to plan your next trip, or you are an airline or travel agency, "
         "you need to know about ticket and service price variations.\n"
         "Airline ticket pricing has become increasingly complex due to factors such as demand fluctuations and seasonal trends.\n"
         "\n"
         "My project aims to help you make the right decision and buy the best ticket at the best price by developing a predictive model "
         "that can accurately estimate flight fares based on the given features.")
   
    im1 = Image.open('how airline ticketing system work.jpg')
    im2 = Image.open('images.png')
    im3 = Image.open('1686635188.webp')
    im4 = Image.open('images (1).jpg')
    im5 = Image.open('images1.jpg')
    # Images2 Cards for flight ticket booking stages
    img1, img2 = st.columns((5, 5))
    # Display the images in the columns
    img1.image(im1, 
               caption='flight ticket booking stages')
    img2.image(im2, 
               caption='Decide The best Time and Day to book Flight')
    
    img3, img4 = st.columns((5, 5))
    img3.image(im3, 
              caption='Choose the cheap Flight')
    img4.image(im4, 
                caption='Book the flight')
    img5 = st
    img5.image(im5, 
              caption='Have a nice flight')
    

# Data Analysis
with tab2:
   
    visual1, visual2 = st.columns((5, 5))
    with visual1:
        st.subheader('Top Airlines')
        most_airline = filtered_data['Airline'].value_counts().sort_values(ascending=False).head()
        fig = px.pie(data_frame=most_airline, 
                     names=most_airline.index, 
                     values=most_airline.values, 
                     hole=0.4)
        st.plotly_chart(fig, use_container_width=True)

    # Get the top airlines based on the most_airline index
    top_airlines = most_airline.index.tolist()
    with visual2:
        st.subheader('Airline Price')
        airline_price = filtered_data[filtered_data['Airline'].isin(top_airlines)].groupby('Airline')['Price'].min().sort_values(ascending=False)
        fig = px.bar(airline_price, 
                    x=airline_price.index, 
                    y=airline_price.values)
        # Customize x-axis and y-axis labels
        fig.update_xaxes(title='Airline')
        fig.update_yaxes(title='Price')
        st.plotly_chart(fig, use_container_width=True) 
    

    st.subheader('Duration vs Price')
    fig = px.scatter(filtered_data,
                    x='Duration',
                    y='Price',
                    color=filter_box,
                    )
    # Customize x-axis and y-axis labels
    fig.update_xaxes(title='Duration')
    fig.update_yaxes(title='Price')
    st.plotly_chart(fig, use_container_width=True)
    # Customize the width and placement of the legend
    fig.update_layout(
        legend=dict(
            borderwidth=2,  # Set the width of the legend border
            orientation='h',
            x=1,  # Set the x position of the legend (1 means right-aligned)
            y=1,  # Set the y position of the legend (1 means top-aligned)
            xanchor='right',  # Set the x anchor to 'right' for right alignment
            yanchor='top',  # Set the y anchor to 'top' for top alignment
            traceorder='normal'
        )
    )


# predicting Model
with tab3:

    model = joblib.load('aviation_flight_fare_prediction_model.p')    
    sc = StandardScaler()

    # Inputs 
    col1, col2 = st.columns((5,5))
    with col1:
        airline_pred = st.selectbox("Airline Carrier", list(df['Airline'].unique()))
        airline_pred = {'IndiGo':3, 'Air India':1, 'Jet Airways':4, 'SpiceJet':8,
                'Multiple carriers':6, 'GoAir':2, 'Vistara':10, 'Air Asia':0,
                'Vistara Premium economy':11, 'Jet Airways Business':5,
                'Multiple carriers Premium economy':7, 'Trujet':9}.get(airline_pred, airline_pred)
        
        source_pred= st.selectbox("Departure City", list(df['Source'].unique()))    
        source_pred = {'Banglore':0, 'Kolkata':3, 'Delhi':2, 'Chennai':1, 'Mumbai':4}.get(source_pred, source_pred)
        
        destination_pred = st.selectbox("Arrival City", list(df['Source'].unique())) 
        destination_pred = {'New Delhi':5, 'Banglore':0, 'Cochin':1, 'Kolkata':4,
                    'Delhi':2, 'Hyderabad':3}.get(destination_pred, destination_pred)
        
        stops_pred= int(st.selectbox("Stops", options= df['Total_Stops'].unique()))
        
        duration_pred_scaled = sc.fit_transform([[int(st.number_input("Flight Duration (in minutes)",
                                                                    min_value=0, step=10))]])
    
        
    
    with col2:
                
        add_info_pred= st.selectbox("Additional Services", list(df['Additional_Info'].unique()))
        add_info_pred = {'No info':7, 'In-flight meal not included':5, 'No check-in baggage included':6,
                    '1 Short layover':1, '1 Long layover':0, 'Change airports':4,
                    'Business class':3, 'Red-eye flight':8, '2 Long layover':2}.get(add_info_pred, add_info_pred)

        # day_pred= int(st.selectbox("Day", options= df['Day'].unique()))
        # month_pred= int(st.selectbox("Month", options= df['Month'].unique()))

        # Date Selection
        today = date.today()
        min_date = today + pd.DateOffset(days=1)
        max_date = today + pd.DateOffset(months=6)
        selected_date = st.date_input('Select a date', min_value=min_date, max_value=max_date, value=min_date)

        st.write(' ')
        st.write(' ')
        st.write('for Departure Hour, If the minutes more than 30, Please increase the hour by 1')
        
        dep_hour_pred_scaled = sc.fit_transform([[int(st.number_input("Departure Hour (24 format)",
                                                                    min_value=0))]])

    # Submit Button
    if st.button("Submit 👇"):
        input_data = np.array([[duration_pred_scaled[0][0], stops_pred, selected_date.day, selected_date.month, 
                                dep_hour_pred_scaled[0][0], airline_pred, source_pred,
                                  destination_pred, add_info_pred]])

        Price = model.predict(input_data)
        # Display the price as a metric
        st.metric("Ticket Price", int(Price))

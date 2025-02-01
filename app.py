import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt
from textblob import TextBlob
from wordcloud import WordCloud
import joblib
from statsmodels.tsa.statespace.sarimax import SARIMAX
import os
import requests  # Add this import to download files

# Title of the dashboard and other configurations
st.set_page_config(page_title="Jamaica's Travel Statistics", layout="wide")

# Add custom CSS for background image-still no loading
background_image = "background.jpg"
st.markdown(
    f"""
    <style>
    .stApp {{
        background: url({background_image});
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """,
    unsafe_allow_html=True
)

# Download large file from Google Drive (before loading datasets)
@st.cache_data
def download_large_file():
    url = "https://drive.google.com/uc?export=download&id=113zVfzZm9uB-lPlAZgVeHaTtx3yGAZmU"
    output_file = "streamlit_templates/updated_filtered_tourism_metrics_with_other.csv"

    try:
        response = requests.get(url)
        if response.status_code == 200:
            with open(output_file, "wb") as file:
                file.write(response.content)
            st.success(f"File downloaded successfully: {output_file}")
        else:
            st.error("Failed to download the file. Check the URL or permissions.")
    except Exception as e:
        st.error(f"An error occurred while downloading the file: {e}")

# Call the function to download the large file
download_large_file()

# Function to load data with error handling
@st.cache_data
def load_data(file_path):
    try:
        # Attempt to read the CSV file
        data = pd.read_csv(file_path, on_bad_lines='skip', engine='python')
        return data
    except FileNotFoundError:
        # Handle file not found error
        st.error(f"File not found: {file_path}")
        return None
    except pd.errors.ParserError as e:
        # Handle parsing errors
        st.error(f"Error parsing file: {e}")
        return None
    except Exception as e:
        # Handle any other errors
        st.error(f"An error occurred: {e}")
        return None

# Function to preprocess data
@st.cache_data
def preprocess_data(data):
    # Ensure month is treated as a category with correct order
    if 'month' in data.columns:
        data['month'] = pd.Categorical(data['month'], 
                                       categories=['January', 'February', 'March', 'April', 
                                                   'May', 'June', 'July', 'August', 
                                                   'September', 'October', 'November', 'December'], 
                                       ordered=True)
    # Round Length of Stay to 2 decimal places if it exists
    if 'LENGTH OF STAY' in data.columns:
        data['LENGTH OF STAY'] = data['LENGTH OF STAY'].round(2)
    
    # Convert DATE to datetime and extract year and month for filtering
    if 'DATE' in data.columns:
        data['DATE'] = pd.to_datetime(data['DATE'])
        data['Month-Year'] = data['DATE'].dt.strftime('%Y-%m')
    
    return data

# Load datasets
file_path_1 = 'streamlit_templates/future_visitor_arrivals_forecast_with_lagged_features.csv'
file_path_2 = 'streamlit_templates/forecasted_visitor_arrivals_by_country_region.csv'
file_path_metrics = 'streamlit_templates/updated_filtered_tourism_metrics_with_other.csv'
main_data_path = 'streamlit_templates/cleaned_corrected_filtered_dataset.csv'
forecast_data_path = 'streamlit_templates/sarima_forecast.csv'
nps_data_path = "streamlit_templates/aggregated_nps_data.csv"
sentiment_data_path = "streamlit_templates/cleaned_combined_data.csv"

with st.spinner("Loading datasets..."):
    pred_data = load_data(file_path_1)
    country_region_pred_data = load_data(file_path_2)
    metrics_data = load_data(file_path_metrics)
    main_data = load_data(main_data_path)
    forecast_data = load_data(forecast_data_path)
    nps_data = load_data(nps_data_path)
    sentiment_data = load_data(sentiment_data_path)

# Validate and preprocess data
if pred_data is not None and validate_data(pred_data, ['DATE', 'Forecast', 'Lower CI', 'Upper CI']):
    pred_data = preprocess_data(pred_data)
    pred_data.rename(columns={'Lower CI': 'Lowest possible count', 'Upper CI': 'Highest possible count'}, inplace=True)

if country_region_pred_data is not None and validate_data(country_region_pred_data, ['DATE', 'Forecast', 'Lower CI', 'Upper CI', 'COUNTRY/REGION']):
    country_region_pred_data = preprocess_data(country_region_pred_data)
    country_region_pred_data.rename(columns={'Lower CI': 'Lowest possible count', 'Upper CI': 'Highest possible count'}, inplace=True)

if metrics_data is not None and validate_data(metrics_data, ['year', 'month', 'visitor', 'COUNTRY/REGION']):
    metrics_data = preprocess_data(metrics_data)

# Ensure Date is in datetime format in main_data
if main_data is not None:
    if 'Date' in main_data.columns:
        main_data['Date'] = pd.to_datetime(main_data['Date'], errors='coerce')
    else:
        st.error("The 'Date' column is not present in the main dataset.")

    # Check if the conversion was successful and handle the error if not
    if main_data['Date'].isnull().all():
        st.error("The 'Date' column could not be converted to datetime format. Please check the data.")
else:
    st.error("Main data could not be loaded.")

# Ensure Date is in datetime format in forecast_data
if forecast_data is not None:
    if 'Date' in forecast_data.columns:
        forecast_data['Date'] = pd.to_datetime(forecast_data['Date'], errors='coerce')
    else:
        st.error("The 'Date' column is not present in the forecast dataset.")

    # Check if the conversion was successful and handle the error if not
    if forecast_data['Date'].isnull().all():
        st.error("The 'Date' column could not be converted to datetime format in forecast data. Please check the data.")
else:
    st.error("Forecast data could not be loaded.")

if forecast_data is not None and validate_data(forecast_data, ['Date', 'Forecasted Load Factor', 'Lower CI', 'Upper CI']):
    forecast_data = preprocess_data(forecast_data)

if nps_data is not None and validate_data(nps_data, ['Year', 'NPS']):
    nps_data = preprocess_data(nps_data)

if sentiment_data is not None and validate_data(sentiment_data, ['Year', 'Month', 'Comments']):
    sentiment_data['Sentiment'] = sentiment_data['Comments'].apply(lambda x: TextBlob(x).sentiment.polarity if pd.notna(x) else 0)

# Sidebar for navigation
st.sidebar.title("Navigation")
selected_dashboard = st.sidebar.radio("Choose Dashboard", ["Visitor Arrival Predictions", "Tourism Performance Metrics", "Load Factor Prediction", "Aviation Metrics", "NPS and Sentiment Analysis Dashboard"])

# Page 1: Visitor Arrival Predictions
if selected_dashboard == "Visitor Arrival Predictions":
    st.title("Visitor Arrival Predictions")
    
    selected_country = st.sidebar.selectbox("Select Country/Region", options=['All'] + sorted(country_region_pred_data['COUNTRY/REGION'].unique().tolist()))
    selected_month_years = st.sidebar.multiselect("Select Month-Year(s)", options=['All'] + sorted(pred_data['Month-Year'].unique().tolist()), default='All')

    # Filter data based on selections
    if selected_country == 'All':
        if 'All' in selected_month_years or not selected_month_years:
            filtered_pred_data = pred_data.copy()  # Show all data by default
        else:
            filtered_pred_data = pred_data[pred_data['Month-Year'].isin(selected_month_years)].copy()
    else:
        if 'All' in selected_month_years or not selected_month_years:
            filtered_pred_data = country_region_pred_data[country_region_pred_data['COUNTRY/REGION'] == selected_country].copy()
        else:
            filtered_pred_data = country_region_pred_data[(country_region_pred_data['COUNTRY/REGION'] == selected_country) &
                                                          (country_region_pred_data['Month-Year'].isin(selected_month_years))].copy()

    # Visualization
    if not filtered_pred_data.empty:
        if len(selected_month_years) == 1 and 'All' not in selected_month_years:
            # Single Month-Year selection: Speedometer visualization
            pred = filtered_pred_data['Forecast'].iloc[0]
            low_ci = filtered_pred_data['Lowest possible count'].iloc[0]
            high_ci = filtered_pred_data['Highest possible count'].iloc[0]

            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=pred,
                title={'text': f"Predicted Visitor Arrivals ({selected_month_years[0]})"},
                gauge={
                    'axis': {'range': [None, high_ci]},
                    'steps': [
                        {'range': [low_ci, high_ci], 'color': "lightgray"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': high_ci}}))

            st.plotly_chart(fig)

        else:
            # Multiple Month-Year selections or 'All': Line graph visualization
            fig = px.line(filtered_pred_data, x='DATE', y='Forecast', title=f"Predicted Visitor Arrivals for {selected_country if selected_country != 'All' else 'All Countries'}", line_shape='linear', width=1400, height=600)
            
            # Add lowest and highest possible count as lines to the graph
            fig.add_trace(go.Scatter(x=filtered_pred_data['DATE'], y=filtered_pred_data['Lowest possible count'],
                                     mode='lines', name='Lowest possible count', line=dict(color='blue')))
            fig.add_trace(go.Scatter(x=filtered_pred_data['DATE'], y=filtered_pred_data['Highest possible count'],
                                     mode='lines', name='Highest possible count', line=dict(color='green')))

            fig.update_layout(xaxis=dict(tickformat='%Y', dtick="M12"), xaxis_title="Date")
            
            st.plotly_chart(fig)

    else:
        st.write("No data available for the selected filters.")
# Page 2: Tourism Performance Metrics
elif selected_dashboard == "Tourism Performance Metrics":
    st.title("Tourism Performance Metrics")
    
    # Sidebar filters
    selected_year = st.sidebar.selectbox("Select Year", options=['All'] + sorted(metrics_data['year'].unique().tolist()))
    selected_month = st.sidebar.selectbox("Select Month", options=['All'] + list(metrics_data['month'].unique()))
    selected_country = st.sidebar.selectbox("Select Country/Region", options=['All'] + sorted(metrics_data['COUNTRY/REGION'].dropna().unique().tolist()))

    # Filter data based on selections
    filtered_data = metrics_data.copy()

    if selected_year != 'All':
        filtered_data = filtered_data[filtered_data['year'] == int(selected_year)]

    if selected_month != 'All':
        filtered_data = filtered_data[filtered_data['month'] == selected_month]

    if selected_country != 'All':
        filtered_data = filtered_data[filtered_data['COUNTRY/REGION'] == selected_country]

    # Display the filtered data if it's not empty
    if not filtered_data.empty:
        # Visualization: Port of Entry
        if not filtered_data['PORT OF ENTRY'].isna().all():
            ports_of_entry = filtered_data['PORT OF ENTRY'].value_counts().reset_index()
            ports_of_entry.columns = ['Port of Entry', 'Count']
            fig_ports = px.bar(ports_of_entry, x='Port of Entry', y='Count', title='Port of Entry')
            st.plotly_chart(fig_ports)

        # Visualization: Top Airlines
        if not filtered_data['Airline Name'].isna().all():
            top_airlines = filtered_data['Airline Name'].value_counts().reset_index()
            top_airlines.columns = ['Airline Name', 'Count']
            fig_airlines = px.bar(top_airlines, x='Airline Name', y='Count', title='Top Airlines')
            st.plotly_chart(fig_airlines)

        # Visualization: Purpose of Travel
        if not filtered_data['PURPOSE OF TRAVEL'].isna().all():
            purposes_of_travel = filtered_data['PURPOSE OF TRAVEL'].value_counts().reset_index()
            purposes_of_travel.columns = ['Purpose of Travel', 'Count']
            fig_purposes = px.bar(purposes_of_travel, x='Purpose of Travel', y='Count', title='Purpose of Travel')
            st.plotly_chart(fig_purposes)

        # Visualization: Travel Personas
        if not filtered_data['Travel Persona'].isna().all():
            travel_personas = filtered_data['Travel Persona'].value_counts().reset_index()
            travel_personas.columns = ['Travel Persona', 'Count']
            fig_personas = px.bar(travel_personas, x='Travel Persona', y='Count', title='Travel Personas')
            st.plotly_chart(fig_personas)

        # Stylized Card for Average Length of Stay
        if not filtered_data['LENGTH OF STAY'].isna().all():
            avg_length_of_stay = filtered_data['LENGTH OF STAY'].mean()
            st.markdown(
                f"""
                <div style="background-color: #2C3E50; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: white;">Average Length of Stay</h2>
                    <h1 style="color: #E74C3C;">{avg_length_of_stay:.2f} days</h1>
                </div>
                """, unsafe_allow_html=True
            )

        # Stylized Card for Total Visitor Arrivals
        total_visitor_arrivals = filtered_data['visitor'].sum()
        st.markdown(
            f"""
            <div style="background-color: #27AE60; padding: 20px; border-radius: 10px; text-align: center;">
                <h2 style="color: white;">Total Visitor Arrivals</h2>
                <h1 style="color: #F1C40F;">{total_visitor_arrivals:.0f}</h1>
            </div>
            """, unsafe_allow_html=True
        )

        # Stylized Card for YoY % Comparison
        if selected_year != 'All' and selected_year != '2015':
            previous_year = int(selected_year) - 1
            prev_year_data = metrics_data[(metrics_data['year'] == previous_year)]
            
            if selected_month != 'All':
                prev_year_data = prev_year_data[prev_year_data['month'] == selected_month]
            if selected_country != 'All':
                prev_year_data = prev_year_data[prev_year_data['COUNTRY/REGION'] == selected_country]

            prev_year_total = prev_year_data['visitor'].sum()
            yoy_change = ((total_visitor_arrivals - prev_year_total) / prev_year_total) * 100 if prev_year_total > 0 else 0
            st.markdown(
                f"""
                <div style="background-color: #8E44AD; padding: 20px; border-radius: 10px; text-align: center;">
                    <h2 style="color: white;">YoY % Comparison</h1>
                    <h1 style="color: #ECF0F1;">{yoy_change:.2f}%</h1>
                </div>
                """, unsafe_allow_html=True
            )
        elif selected_year == 'All' and selected_month == 'All' and selected_country == 'All':
            # Year on Year comparison for all years
            for year in sorted(metrics_data['year'].unique()):
                if year != 2015:
                    year_data = metrics_data[metrics_data['year'] == year]
                    year_total = year_data['visitor'].sum()
                    previous_year_data = metrics_data[metrics_data['year'] == year - 1]
                    previous_year_total = previous_year_data['visitor'].sum()
                    yoy_change = ((year_total - previous_year_total) / previous_year_total) * 100 if previous_year_total > 0 else 0
                    st.markdown(
                        f"""
                        <div style="background-color: #8E44AD; padding: 20px; border-radius: 10px; text-align: center;">
                            <h2 style="color: white;">YoY % Comparison for {year}</h2>
                            <h1 style="color: #ECF0F1;">{yoy_change:.2f}%</h1>
                        </div>
                        """, unsafe_allow_html=True
                    )
    else:
        st.write("No data available for the selected filters.")

# Page 3: Load Factor Predictions
elif selected_dashboard == "Load Factor Prediction":
    st.title("Load Factor Prediction")
    st.write("This section provides Load factor predictions for the next 24 months.")

    st.sidebar.title("Filters")
   
    # Ensure Date column is properly formatted in forecast_data
    if 'Date' in forecast_data.columns:
        forecast_data['Date'] = pd.to_datetime(forecast_data['Date'], errors='coerce')

    if forecast_data['Date'].isnull().all():
        st.error("The 'Date' column could not be converted to datetime format in forecast data. Please check the data.")
    else:
        selected_month_years = st.sidebar.multiselect("Select Month-Year(s)", options=['All'] + sorted(forecast_data['Date'].dt.to_period('M').unique().astype(str)), default='All')

        if 'All' in selected_month_years or not selected_month_years:
            filtered_forecast_data = forecast_data.copy()
        else:
            filtered_forecast_data = forecast_data[forecast_data['Date'].dt.to_period('M').astype(str).isin(selected_month_years)]

        if len(selected_month_years) == 1 and 'All' not in selected_month_years:
            selected_month_year = selected_month_years[0]
            forecast_value = filtered_forecast_data['Forecasted Load Factor'].iloc[0]
            lower_ci = filtered_forecast_data['Lower CI'].iloc[0]
            upper_ci = filtered_forecast_data['Upper CI'].iloc[0]

            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=forecast_value,
                title={'text': f"Predicted Load Factor for {selected_month_year}"},
                delta={'reference': lower_ci},
                gauge={
                    'axis': {'range': [None, upper_ci]},
                    'steps': [
                        {'range': [lower_ci, upper_ci], 'color': "lightgray"}],
                    'threshold': {'line': {'color': "red", 'width': 4}, 'thickness': 0.75, 'value': upper_ci}}))

            st.plotly_chart(fig_gauge)

        else:
            fig_forecast = px.line(filtered_forecast_data, x='Date', y='Forecasted Load Factor', title="Predicted Load Factor Over Time", width=1400, height=600)

            # Add lower and upper confidence intervals
            fig_forecast.add_traces([
                go.Scatter(x=filtered_forecast_data['Date'], y=filtered_forecast_data['Lower CI'], mode='lines', name='Lower CI', line=dict(dash='dash')),
                go.Scatter(x=filtered_forecast_data['Date'], y=filtered_forecast_data['Upper CI'], mode='lines', name='Upper CI', line=dict(dash='dash'))
            ])

            fig_forecast.update_layout(xaxis=dict(tickformat='%Y', dtick="M12"), xaxis_title="Date")
            
            st.plotly_chart(fig_forecast)

# Page 4: Aviation Metrics
elif selected_dashboard == "Aviation Metrics":
    st.title("Aviation Metrics")

    st.sidebar.title("Filters")
    year_month_filter = st.sidebar.selectbox("Select Year-Month", sorted(main_data['Date'].dt.to_period('M').unique().astype(str)))

    filtered_main_data = main_data[main_data['Date'].dt.to_period('M').astype(str) == year_month_filter]

    # Corrected calculation for average flights per week
    total_flights_in_month = len(filtered_main_data)  # Total flights in the selected month-year
    weeks_in_month = 4.345  # Approximate weeks in a month
    avg_flights_per_week = total_flights_in_month / weeks_in_month

    # Calculate average cities served per week in the selected month-year
    avg_cities_per_week = filtered_main_data['City'].nunique() / weeks_in_month
    
    # Calculate the average load factor
    average_load_factor = filtered_main_data['Load Factor'].mean()

    st.markdown(f"### Metrics for {year_month_filter}")

    # Display metrics as tiles
    flight_card_html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #ff6347; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
        <h2 style="color: #fff;">Average Flights/Week</h2>
        <p style="font-size: 48px; font-weight: bold; margin: 10px 0;">{avg_flights_per_week:.2f}</p>
    </div>
    """

    city_card_html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #4682b4; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
        <h2 style="color: #fff;">Average Cities/Week</h2>
        <p style="font-size: 48px; font-weight: bold; margin: 10px 0;">{avg_cities_per_week:.2f}</p>
    </div>
    """

    load_factor_card_html = f"""
    <div style="padding: 20px; border-radius: 10px; background-color: #32cd32; box-shadow: 2px 2px 8px rgba(0,0,0,0.1);">
        <h2 style="color: #fff;">Average Load Factor</h2>
        <p style="font-size: 48px; font-weight: bold; margin: 10px 0;">{average_load_factor:.2f}</p>
    </div>
    """

    # Display the cards
    st.markdown(flight_card_html, unsafe_allow_html=True)
    st.markdown(city_card_html, unsafe_allow_html=True)
    st.markdown(load_factor_card_html, unsafe_allow_html=True)

    # Display load factors by top 10 countries
    st.markdown("### Load Factor by Country")
    top_10_countries = filtered_main_data.groupby('Country')['Load Factor'].mean().nlargest(10).reset_index()
    fig_country = px.bar(top_10_countries, x='Country', y='Load Factor', title="Top Countries by Load Factor")
    fig_country.update_traces(hovertemplate='%{y:.2f}')
    st.plotly_chart(fig_country)
    
    # Display load factors by top 20 airlines
    st.markdown("### Load Factor by Airline")
    top_20_airlines = filtered_main_data.groupby('Airline')['Load Factor'].mean().nlargest(20).reset_index()
    fig_airline = px.bar(top_20_airlines, x='Airline', y='Load Factor', title="Top Airlines by Load Factor")
    fig_airline.update_traces(hovertemplate='%{y:.2f}')
    st.plotly_chart(fig_airline)
    
    # Display load factors by top 20 cities
    st.markdown("### Load Factor by City")
    top_20_cities = filtered_main_data.groupby('City')['Load Factor'].mean().nlargest(20).reset_index()
    fig_city = px.bar(top_20_cities, x='City', y='Load Factor', title="Top Cities by Load Factor")
    fig_city.update_traces(hovertemplate='%{y:.2f}')
    st.plotly_chart(fig_city)

# Page 5: NPS and Sentiment Analysis Dashboard
elif selected_dashboard == "NPS and Sentiment Analysis Dashboard":
    st.title("NPS and Sentiment Analysis Dashboard")
    st.write("This dashboard provides insights into the Net Promoter Score (NPS) data and sentiment analysis of comments and articles.")

    # NPS Over Time (Column Chart with Trend Line)
    st.subheader("NPS Over Time")
    nps_over_time = nps_data.groupby('Year')['NPS'].mean().reset_index()
    fig = px.bar(nps_over_time, x='Year', y='NPS', title="NPS Over Time")
    fig.update_traces(marker_color='blue')
    fig.add_scatter(x=nps_over_time['Year'], y=nps_over_time['NPS'], mode='lines+markers', line=dict(color='red'), name='Trend Line')
    st.plotly_chart(fig)

    # Year Filter
    st.sidebar.title("Filters")
    year = st.sidebar.selectbox("Select Year", options=sorted(nps_data['Year'].unique().tolist()))

    # NPS for Selected Year (Speedometer Gauge)
    st.subheader(f"NPS Score for {int(year)}")
    yearly_nps = nps_data[nps_data['Year'] == year]['NPS'].mean()
    delta_nps = yearly_nps - 60  # Difference from 60
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=yearly_nps,
        title={'text': f"NPS {int(year)}"},
        delta={'reference': 60, 'position': "top"},
        gauge={'axis': {'range': [-100, 100]},
               'bar': {'color': "darkblue"},
               'steps': [
                   {'range': [-100, 0], 'color': "red"},
                   {'range': [0, 100], 'color': "green"}],
               'threshold': {'line': {'color': "orange", 'width': 4}, 'thickness': 0.75, 'value': 60}}))
    st.plotly_chart(fig)

    # NPS Per Month for Selected Year (Line Chart)
    st.subheader(f"NPS Score Per Month for {int(year)}")
    monthly_nps = nps_data[nps_data['Year'] == year].groupby('Month')['NPS'].mean().reset_index()
    monthly_nps['Month'] = monthly_nps['Month'].apply(lambda x: pd.to_datetime(str(x), format='%m').strftime('%B'))  # Convert month numbers to names
    fig, ax = plt.subplots()
    sns.lineplot(data=monthly_nps, x='Month', y='NPS', marker='o', ax=ax)
    ax.set_ylabel("NPS")
    ax.set_xlabel("Month")
    ax.set_title(f"NPS Score Per Month for {int(year)}")
    st.pyplot(fig)

    # Sentiment Analysis Section
    st.subheader("Sentiment Analysis")

    # Year Filter for Sentiment Analysis
    sentiment_year = year  # Reusing the year selected for NPS

    # Sentiment Distribution Per Month for Selected Year
    st.subheader(f"Sentiment Distribution Per Month for {int(sentiment_year)}")
    monthly_sentiment = sentiment_data[sentiment_data['Year'] == sentiment_year].groupby('Month')['Sentiment'].mean().reset_index()
    monthly_sentiment['Month'] = monthly_sentiment['Month'].apply(lambda x: pd.to_datetime(str(int(x)), format='%m').strftime('%B'))
    fig = px.bar(monthly_sentiment, x='Month', y='Sentiment', title=f"Sentiment Distribution Per Month for {int(sentiment_year)}")
    fig.update_traces(texttemplate='%{y:.2f}', textposition='outside')
    st.plotly_chart(fig)

    # Word Cloud for Selected Year
    st.subheader(f"Word Cloud for {int(sentiment_year)}")
    year_comments = sentiment_data[sentiment_data['Year'] == sentiment_year]['Comments'].dropna().tolist()
    year_text = ' '.join(year_comments)
    wordcloud = WordCloud(width=800, height=400, background_color='white').generate(year_text)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    st.pyplot(plt)

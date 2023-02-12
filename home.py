from turtle import color
import streamlit as st
from streamlit_option_menu import option_menu
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as data
from keras.models import load_model
import seaborn as sns
import plotly.figure_factory as ff
from ta.volatility import BollingerBands
from ta.trend import MACD
from ta.momentum import RSIIndicator
import plost
import tensorflow as tf



st.set_page_config(
    layout="wide",
    page_title = "STOCKPRO"
)

page_bg_img = '''
<style>
body {
background-image: url("https://images.unsplash.com/photo-1542281286-9e0a16bb7366");
background-size: cover;
}
</style>
'''

st.markdown(page_bg_img, unsafe_allow_html=True)
#page_bg_img = """
#<style>
#[data-testid="stAppViewContainer"} {
#background-image: url('https://mdbootstrap.com/img/new/slides/003.jpg');
#background-size: cover;
#}
#</style>
#"""
#st.markdown(page_bg_img, unsafe_allow_html = True)



selected = option_menu(
    menu_title=None,
    options=["Home", "Data", "Visualization","Analogy","Prediction"],
    default_index=0,
    orientation="horizontal",
    styles={
        "container":{"padding":"0!important","background-color":"black"},
        "nav-link": {
            "font-size":"25px",
            "text-align":"center",
            "margin":"0px",
            "--hover-color":"#eee",
        },
        "nav-link-selected":{"background-color":"purple"},
    },
)







if selected == "Home":
    
    st.markdown("<h1 style = 'text-align:center;color:red;'>STOCKPRO</h1>",unsafe_allow_html = True)
    st.markdown("<h3 style = 'text-align:center;color:teal;'>Get perfect stock market solutions with StockPro</h3>",unsafe_allow_html = True)
    #from PIL import Image
    #image = Image.open('aboutt.jpg')
    #buffer, col5, col6 = st.columns([1,3,1])
    #with col5:
        #st.image(image)

    import time
    import requests
    import json



    from streamlit_lottie import st_lottie
    from streamlit_lottie import st_lottie_spinner

    
    cc1,cc2 = st.columns([1,1])
    with cc1:

        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)


        lottie_coding = load_lottiefile("lottie.json")
        st_lottie(
            lottie_coding,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            height="600px",
            width="600px",
            key=None,
        )

    with cc2:
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.markdown("<h3 style = 'text-align:center;color:white;'>Welcome to the world of insights about stock market data. We are a team of talented stock analysts.</h3>",unsafe_allow_html = True)
        
        
    cc8,cc9 = st.columns([1,1])

    with cc8:
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.write("#")
        st.markdown("<h3 style = 'text-align:center;color:white;'>STOCKPRO is all about getting to know every aspect of stock data. You will find the history and future of stock market in here.</h3>",unsafe_allow_html = True)

    with cc9:
        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)


        lottie_coding = load_lottiefile("lottief.json")
        st_lottie(
            lottie_coding,
            speed=1,
            reverse=False,
            loop=True,
            quality="high",
            height="600px",
            width="600px",
            key=None,
        )




       

if selected == "Data":
    
    st.markdown("<h1 style = 'text-align:center;color:white;'>Historical Data</h1>",unsafe_allow_html = True)
    
    user_input = st.sidebar.text_input('Enter Stock Ticker', 'GOOG')
    st.header('{}'.format(user_input))

    

    start = st.sidebar.date_input('Start Date', value = pd.to_datetime('2022-01-01'))
    
    end = st.sidebar.date_input('End Date', value = pd.to_datetime('today'))
    
    


    df = data.DataReader(user_input, 'yahoo', start, end)

    st.subheader('Data for chosen date')
    buffer, col5 = st.columns([1.2,3])
    with col5:
        st.write(df)

    st.subheader('Data for last day')
    buffer, col5 = st.columns([1.2,3])
    with col5:
        st.write(df.tail(1))

    st.subheader('Data for last week')
    buffer, col5 = st.columns([1.2,3])
    with col5:
        st.write(df.tail(7))


    st.subheader('Data for last month')
    buffer, col5 = st.columns([1.2,3])
    with col5:
        st.write(df.tail(31))

    
    cc1,cc2,cc3,cc4 = st.columns([1,1,1,1])
    with cc1:
        st.subheader('Average closing price')
    
        st.write(df['Close'].mean())


    with cc4:
        st.subheader('Average Volume')
        st.write(df['Volume'].mean())
    




if selected == "Visualization":
    st.markdown("<h1 style = 'text-align:center;color:white;'>Stock Analysis Visualization</h1>",unsafe_allow_html = True)
    user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    start = st.sidebar.date_input('Start Date', value = pd.to_datetime('2022-01-01'))
    end = st.sidebar.date_input('End Date', value = pd.to_datetime('today'))
    



    def Market(df):
        m = df['Close']*df['Volume']
        df['MarketCap'] = m
        df['MarketCap'] = df['MarketCap'].astype('int64')
        return df





    df = Market(data.DataReader(user_input,'yahoo',start,end))



    def retur(dl):
        n = (dl['Close']/dl['Close'].shift(1))-1
        dl['returns'] = n
        dl['returns'] = dl['returns']
        return dl
    dl = retur(data.DataReader(user_input,'yahoo',start,end))



    indicator_bb = BollingerBands(df['Close'])
    bb = df
    bb['bb_h'] = indicator_bb.bollinger_hband()
    bb['bb_l'] = indicator_bb.bollinger_lband()
    bb = bb[['Close','bb_h','bb_l']]

    # Moving Average Convergence Divergence
    macd = MACD(df['Close']).macd()

    # Resistence Strength Indicator
    rsi = RSIIndicator(df['Close']).rsi()



    plot_height = st.slider('Specify plot height', 200,600,300)

    cc1,cc2 = st.columns([1,1])
    with cc1:
        st.subheader('Open Price Chart')
        st.line_chart(df['Open'], height = plot_height)
        st.subheader('Close Price Chart')
        st.line_chart(df['Close'], height = plot_height)

    with cc2:
        st.subheader('High Price Chart')
        st.line_chart(df['High'], height = plot_height)
        st.subheader('Low Price Chart')
        st.line_chart(df['Low'], height = plot_height)


    cc3, cc4 = st.columns([1,1])
    with cc3:
        st.subheader('Stock Market Volume')
        st.line_chart(df['Volume'], height = plot_height)
        st.subheader('Adj Close Price Chart')
        st.line_chart(df['Adj Close'], height = plot_height)


    with cc4:
        st.subheader('Market Capitalization')
        st.line_chart(df['MarketCap'], height = plot_height)
        st.subheader('Percentage increase in stock value')
        st.line_chart(dl['returns'], height = plot_height)


    


    # Plot the prices and the bolinger bands
    st.subheader('Stock Bollinger Bands')
    st.line_chart(bb, height = plot_height)

    progress_bar = st.progress(0)

    # Plot MACD
    st.subheader('Stock Moving Average Convergence Divergence (MACD)')
    st.area_chart(macd, height = plot_height)

    # Plot RSI
    st.subheader('Stock RSI ')
    st.line_chart(rsi, height = plot_height)


    





    



if selected == "Analogy":
    st.markdown("<h1 style = 'text-align:center;color:white;'>Stock Comparison</h1>",unsafe_allow_html = True)
    tickers = ('META','AAPL','NFLX','GOOG','AMZN','AXP','BABA','ABT','AAL','MSFT','NVDA','ORCL','CSCO','IBM','INTC','PYPL','SONY','ABNB','UBER','SNOW','TWTR','HPQ','NOK','DELL','TSLA','XIACF','AMD','ADBE','ACN','TSM')
    dropdown = st.sidebar.multiselect('Pick', tickers)
    start = st.sidebar.date_input('Start Date', value = pd.to_datetime('2022-01-01'))
    end = st.sidebar.date_input('End Date', value = pd.to_datetime('today'))



    def relativeret(dt):
        rel = dt.pct_change()
        cumret = (1+rel).cumprod() - 1
        cumret = cumret.fillna(0)
        return cumret
    if len(dropdown)>0:
        dt = relativeret(data.DataReader(dropdown,'yahoo',start,end))


        
        cc1,cc2 = st.columns([1,1])
        with cc1:
            st.subheader('Volume Comparison')
            st.bar_chart(dt['Volume'])
            st.subheader('Open Price Comparison')
            st.bar_chart(dt['Open'])
        with cc2:
            st.subheader('Close Price Comparison')
            st.bar_chart(dt['Close'])
            st.subheader('High Price Comparison')
            st.bar_chart(dt['High'])

        cc3,cc4 = st.columns(2)
        with cc3:
            st.subheader('Low Price Comparison')
            st.bar_chart(dt['Low'])
        with cc4:
            st.subheader('Adj Close Comparison')
            st.bar_chart(dt['Adj Close'])


if selected == "Prediction":
    st.markdown("<h1 style = 'text-align:center;color:white;'>Stock Trend Prediction</h1>",unsafe_allow_html = True)
    
    


    user_input = st.sidebar.text_input('Enter Stock Ticker', 'AAPL')
    
    start = st.sidebar.date_input('Start Date', value = pd.to_datetime('1990-01-01'))
    end = st.sidebar.date_input('End Date', value = pd.to_datetime('today'))
    
    df = data.DataReader(user_input, 'yahoo', start, end)

    st.subheader('Closing Price vs Time Chart')
    fig = plt.figure(figsize = (10,4))
    plt.plot(df.Close)
    st.pyplot(fig)

    st.subheader('Closing Price vs Time Chart with 100MA')
    ma100 = df.Close.rolling(100).mean()
    fig = plt.figure(figsize = (10,4))
    plt.plot(ma100)
    plt.plot(df.Close)
    st.pyplot(fig)


    #splitting data into training and testing
    data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
    data_testing = pd.DataFrame(df['Close'][int(len(df)*0.70): int(len(df))])

    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0,1))

    data_training_array = scaler.fit_transform(data_training)



    #load my model
    model = load_model('keras_model.h5')


    #testing part
    past_100_days = data_training.tail(100)
    final_df = past_100_days.append(data_testing, ignore_index = True)
    input_data = scaler.fit_transform(final_df)

    x_test = []
    y_test = []

    for i in range(100, input_data.shape[0]):
        x_test.append(input_data[i-100: i])
        y_test.append(input_data[i, 0])

    x_test, y_test = np.array(x_test), np.array(y_test)
    y_predicted = model.predict(x_test)
    scaler = scaler.scale_

    scale_factor = 1/scaler[0]
    y_predicted = y_predicted * scale_factor
    y_test = y_test * scale_factor


    #final graph

    st.subheader('Predictions vs Original')
    fig2 = plt.figure(figsize=(10,4))
    plt.plot(y_test, 'b', label = 'Original Price')
    plt.plot(y_predicted, 'r', label = 'Predicted Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    st.pyplot(fig2)

        
        














    

    



    
    
        
    
        
         
        
        



    






    
    


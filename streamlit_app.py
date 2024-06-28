import streamlit as st
import pandas as pd
import yfinance as yf
import numpy as np
import requests
from textblob import TextBlob
from datetime import datetime, timedelta
from fredapi import Fred
import plotly.express as px
from scipy.optimize import minimize

# Alpha Vantage API Key for News Sentiment
alpha_vantage_api_key = '7ULSSVM1DNTM3E4C'

# Function to fetch news data using Alpha Vantage News API
def get_news_data(ticker):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_api_key}'
    response = requests.get(url)
    news_data = response.json()
    articles = news_data.get('feed', [])
    news_list = []
    for article in articles:
        published_at = article['time_published']
        title = article['title']
        description = article['summary']
        news_list.append([published_at, title, description])
    return news_list

# Function to analyze sentiment using TextBlob
def analyze_sentiment(news_data):
    sentiments = []
    for entry in news_data:
        title = entry[1]
        sentiment_score = TextBlob(title).sentiment.polarity
        sentiments.append(sentiment_score)
    return sentiments

# Function to fetch stock data from Yahoo Finance
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    return hist

# Function to analyze stock based on ticker symbol
def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info

    analysis = {
        'Ticker': ticker,
        'P/E Ratio': info.get('trailingPE'),
        'Forward P/E': info.get('forwardPE'),
        'Price to Sales Ratio': info.get('priceToSalesTrailing12Months'),
        'P/B Ratio': info.get('priceToBook'),
        'Dividend Yield': info.get('dividendYield'),
        'Trailing Eps': info.get('trailingEps'),
        'Target Price': info.get('targetMeanPrice'),
        'Sector': info.get('sector'),
        'Industry': info.get('industry'),
        'Full Time Employees': info.get('fullTimeEmployees'),
        'City': info.get('city'),
        'State': info.get('state'),
        'Country': info.get('country'),
        'Website': info.get('website'),
        'Market Cap (Billion $)': info.get('marketCap') / 1e9 if info.get('marketCap') else None,
        'Enterprise Value (Billion $)': info.get('enterpriseValue') / 1e9 if info.get('enterpriseValue') else None,
        'Enterprise to Revenue': info.get('enterpriseToRevenue'),
        'Enterprise to EBITDA': info.get('enterpriseToEbitda'),
        'Profit Margins': info.get('profitMargins'),
        'Gross Margins': info.get('grossMargins'),
        'EBITDA Margins': info.get('ebitdaMargins'),
        'Operating Margins': info.get('operatingMargins'),
        'Return on Assets (ROA)': info.get('returnOnAssets'),
        'Return on Equity (ROE)': info.get('returnOnEquity'),
        'Revenue Growth': info.get('revenueGrowth'),
        'Payout Ratio': info.get('payoutRatio'),
        'Total Cash (Million $)': info.get('totalCash') / 1e6 if info.get('totalCash') else None,
        'Total Debt (Million $)': info.get('totalDebt') / 1e6 if info.get('totalDebt') else None,
        'Total Revenue (Million $)': info.get('totalRevenue') / 1e6 if info.get('totalRevenue') else None,
        'Gross Profits': info.get('grossProfits'),
        'Total Revenue per Share': info.get('totalRevenuePerShare'),
        'Debt to Equity Ratio': info.get('debtToEquity'),
        'Current Ratio': info.get('currentRatio'),
        'Operating Cashflow (Million $)': info.get('operatingCashflow') / 1e6 if info.get('operatingCashflow') else None,
        'Levered Free Cashflow (Million $)': info.get('leveredFreeCashflow') / 1e6 if info.get('leveredFreeCashflow') else None,
        'Historical Prices': fetch_stock_data(ticker)
    }
    
    return analysis

# Function for portfolio optimization
def optimize_portfolio(tickers, min_weight, max_weight):
    # Datumsbereich festlegen
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    # Datenrahmen für bereinigte Schlusskurse erstellen
    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']

    # Log-Renditen berechnen
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()

    # Kovarianzmatrix berechnen
    cov_matrix = log_returns.cov() * 252

    # Funktionen für Standardabweichung, erwartete Rendite und Sharpe-Ratio
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    # FRED API verwenden, um den aktuellen 10-Jahres-Treasury-Rate zu erhalten
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]

    # Optimierung der Sharpe-Ratio mit einer Iterationsmethode
    num_assets = len(tickers)
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weight_array = np.zeros((num_portfolios, num_assets))

    def objective(weights):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Setze Grenzen für Gewichtungen
    bounds = [(min_weight / 100, max_weight / 100)] * num_assets

    for i in range(num_portfolios):
        weights = np.random.random(num_assets)
        weights /= np.sum(weights)
        weight_array[i, :] = weights

    optimized = minimize(objective, num_assets * [1. / num_assets,], method='SLSQP', bounds=bounds, constraints=constraints)

    optimal_weights = optimized['x']
    optimal_portfolio_return = expected_return(optimal_weights, log_returns)
    optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
    optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)

    return optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, adj_close_df

# Streamlit App
st.title('Stock and Portfolio Analysis')

# Sidebar für Stock Analysis Input
st.sidebar.header('Stock Analysis Input')
ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')

if st.sidebar.button("Analyze Stock"):
    # Analyze stock
    if ticker:
        result = analyze_stock(ticker)
        
        st.subheader(f'Stock Analysis Results for {ticker}')
        
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**P/E Ratio**: {result['P/E Ratio']}")
            st.write(f"**Forward P/E**: {result['Forward P/E']}")
            st.write(f"**Price to Sales Ratio**: {result['Price to Sales Ratio']}")
            st.write(f"**P/B Ratio**: {result['P/B Ratio']}")
            st.write(f"**Dividend Yield**: {result['Dividend Yield']}")
            st.write(f"**Trailing Eps**: {result['Trailing Eps']}")
            st.write(f"**Target Price**: {result['Target Price']}")
            st.write(f"**Sector**: {result['Sector']}")
            st.write(f"**Industry**: {result['Industry']}")
            st.write(f"**Full Time Employees**: {result['Full Time Employees']}")
            st.write(f"**City**: {result['City']}")
            st.write(f"**State**: {result['State']}")
            st.write(f"**Country**: {result['Country']}")
            st.write(f"**Website**: {result['Website']}")
        
        with col2:
            st.write(f"**Market Cap (Billion $)**: {result['Market Cap (Billion $)']}")
            st.write(f"**Enterprise Value (Billion $)**: {result['Enterprise Value (Billion $)']}")
            st.write(f"**Enterprise to Revenue**: {result['Enterprise to Revenue']}")
            st.write(f"**Enterprise to EBITDA**: {result['Enterprise to EBITDA']}")
            st.write(f"**Profit Margins**: {result['Profit Margins']}")
            st.write(f"**Gross Margins**: {result['Gross Margins']}")
            st.write(f"**EBITDA Margins**: {result['EBITDA Margins']}")
            st.write(f"**Operating Margins**: {result['Operating Margins']}")
            st.write(f"**Return on Assets (ROA)**: {result['Return on Assets (ROA)']}")
            st.write(f"**Return on Equity (ROE)**: {result['Return on Equity (ROE)']}")
            st.write(f"**Revenue Growth**: {result['Revenue Growth']}")
            st.write(f"**Payout Ratio**: {result['Payout Ratio']}")
            st.write(f"**Total Cash (Million $)**: {result['Total Cash (Million $)']}")
            st.write(f"**Total Debt (Million $)**: {result['Total Debt (Million $)']}")
            st.write(f"**Total Revenue (Million $)**: {result['Total Revenue (Million $)']}")
            st.write(f"**Gross Profits**: {result['Gross Profits']}")
            st.write(f"**Total Revenue per Share**: {result['Total Revenue per Share']}")
            st.write(f"**Debt to Equity Ratio**: {result['Debt to Equity Ratio']}")
            st.write(f"**Current Ratio**: {result['Current Ratio']}")
            st.write(f"**Operating Cashflow (Million $)**: {result['Operating Cashflow (Million $)']}")
            st.write(f"**Levered Free Cashflow (Million $)**: {result['Levered Free Cashflow (Million $)']}")

        # Display current and historical closing prices
        st.subheader(f'Current and Historical Closing Prices for {ticker}')
        st.write(f"**Current Price**: {result['Historical Prices']['Close'][-1]}")
        st.line_chart(result['Historical Prices']['Close'])

        # Calculate news sentiment
        try:
            news_data = get_news_data(ticker)
            # Analyze sentiment
            sentiments = analyze_sentiment(news_data)
            # Calculate average sentiment
            avg_sentiment = np.mean(sentiments)

            st.subheader('News Sentiment Analysis')
            st.write(f"Average Sentiment for {ticker}: {avg_sentiment:.2f}")

            # Display news articles
            st.subheader('Latest News Articles')
            for article in news_data[:5]:  # Displaying only the first 5 articles
                st.write(f"**Published At**: {article[0]}")
                st.write(f"**Title**: {article[1]}")
                st.write(f"**Summary**: {article[2]}")
                st.write('---')

        except Exception as e:
            st.error(f"Error fetching news data: {str(e)}")

# Sidebar für Portfolio Optimization Input
st.sidebar.header('Portfolio Optimization Input')
tickers_input = st.sidebar.text_input("Enter the stock tickers separated by commas (e.g., AAPL,GME,SAP,TSLA):", "AAPL,GME,SAP,TSLA")
tickers = [ticker.strip() for ticker in tickers_input.split(',')]

min_weight = st.sidebar.slider('Minimum Weight (%)', min_value=0, max_value=100, value=5)
max_weight = st.sidebar.slider('Maximum Weight (%)', min_value=0, max_value=100, value=30)

if st.sidebar.button("Optimize Portfolio"):
    optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, adj_close_df = optimize_portfolio(tickers, min_weight, max_weight)
    
    st.subheader("Optimal Portfolio Metrics:")
    st.write(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
    st.write(f"Expected Portfolio Volatility: {optimal_portfolio_volatility:.4f}")
    st.write(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

    st.subheader("Optimal Weights:")
    optimal_weights_df = pd.DataFrame(optimal_weights, index=tickers, columns=["Weight"])
    st.write(optimal_weights_df)

    # Plot Portfolio Allocation
    fig = px.pie(optimal_weights_df, values='Weight', names=optimal_weights_df.index, title='Portfolio Allocation')
    st.plotly_chart(fig)

    # Display current and historical closing prices for optimized portfolio
    st.subheader('Current and Historical Closing Prices for Optimized Portfolio')
    optimized_portfolio_prices = (adj_close_df * optimal_weights).sum(axis=1)
    st.line_chart(optimized_portfolio_prices)





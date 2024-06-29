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
alpha_vantage_api_key = 'YOUR_API_KEY_HERE'  # Replace with your API key

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
def analyze_stock(ticker, growth_rate):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = fetch_stock_data(ticker)
    
    # Calculate volatility
    hist['Log Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = hist['Log Return'].std() * np.sqrt(252)
    
    # Calculate Max Drawdown
    hist['Cumulative Return'] = (1 + hist['Log Return']).cumprod()
    hist['Cumulative Max'] = hist['Cumulative Return'].cummax()
    hist['Drawdown'] = hist['Cumulative Return'] / hist['Cumulative Max'] - 1
    max_drawdown = hist['Drawdown'].min()
    
    # Calculate Beta (using S&P 500 as a benchmark)
    sp500 = yf.Ticker('^GSPC').history(period='5y')
    sp500['Log Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
    covariance = np.cov(hist['Log Return'].dropna(), sp500['Log Return'].dropna())[0][1]
    beta = covariance / sp500['Log Return'].var()
    
    # Calculate correlation with market (S&P 500)
    correlation = hist['Log Return'].corr(sp500['Log Return'])
    
    # Function to calculate Cost of Equity
    def calculate_cost_of_equity(risk_free_rate, beta, average_market_return):
        cost_of_equity = risk_free_rate + beta * (average_market_return - risk_free_rate)
        return cost_of_equity

    # Use FRED API to get current 10-year Treasury rate
    fred = Fred(api_key='YOUR_API_KEY_HERE')  # Replace with your FRED API key
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]

    # Calculate average market return (you may need to adjust this calculation based on your data)
    # Example: Using S&P 500 index return as average market return
    average_market_return = sp500['Log Return'].mean() * 252

    # Calculate Cost of Equity
    cost_of_equity = calculate_cost_of_equity(risk_free_rate, beta, average_market_return)
    
    # Fair Value Metrics Calculation
    def calculate_graham_valuation(ticker, growth_rate):
        # EPS abrufen
        eps = info.get('trailingEps', None)
        
        if eps is None:
            return None  # Wenn EPS nicht verfügbar, kein Fair Value berechnen
        
        # Risk-Free Rate über FRED API abrufen
        fred = Fred(api_key='YOUR_API_KEY_HERE')  # Replace with your FRED API key
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        risk_free_rate = ten_year_treasury_rate.iloc[-1]
        
        # Fair Value nach Graham Formel berechnen
        graham_valuation = (eps * (8.5 + (2 * growth_rate) * 100) * 4.4) / (risk_free_rate * 100)
        
        return graham_valuation
    
    def calculate_formula_valuation(ticker, growth_rate):
        # Forward P/E Ratio abrufen
        forward_pe_ratio = info.get('forwardPE', None)
        
        if forward_pe_ratio is None:
            return None  # Wenn Forward P/E Ratio nicht verfügbar, kein Fair Value berechnen
        
        # EPS abrufen
        eps = info.get('trailingEps', None)
        
        if eps is None:
            return None  # Wenn EPS nicht verfügbar, kein Fair Value berechnen
        
        # Durchschnittlicher Markt Return
        sp500 = yf.Ticker('^GSPC').history(period='5y')
        average_market_return = sp500['Close'].pct_change().mean() * 252
        
        # Fair Value nach Formel berechnen
        formula_valuation = (forward_pe_ratio * eps * ((1 + growth_rate) ** 5)) / ((1 + average_market_return) ** 5)
        
        return formula_valuation
    
    def calculate_peter_lynch_score(ticker, growth_rate):
        # Dividendenrendite abrufen
        dividend_yield = info.get('dividendYield', None)
        
        if dividend_yield is None or dividend_yield <= 0:
            return None  # Wenn Dividendenrendite nicht verfügbar oder <= 0, kein Score berechnen
        
        # P/E Ratio abrufen
        pe_ratio = info.get('trailingPE', None)
        
        if pe_ratio is None:
            return None  # Wenn P/E Ratio nicht verfügbar, kein Score berechnen
        
        # Score gemäß der Formel berechnen
        peter_lynch_score = (growth_rate * 100) / (pe_ratio * dividend_yield * 100)
        
        return peter_lynch_score
    
    graham_valuation = calculate_graham_valuation(ticker, growth_rate)
    formula_valuation = calculate_formula_valuation(ticker, growth_rate)
    peter_lynch_score = calculate_peter_lynch_score(ticker, growth_rate)
    
    # Stock Analysis Results
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
        'Levered Free Cash Flow': info.get('leveredFreeCashFlow'),
        'Beta': beta,
        'Market Correlation': correlation,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Cost of Equity': cost_of_equity,
        'Graham Valuation': graham_valuation,
        'Formula Valuation': formula_valuation,
        'Peter Lynch Score': peter_lynch_score
    }
    
    return analysis, hist

# Streamlit App
st.title('Stock Analysis Dashboard')

# Sidebar
st.sidebar.title('User Input')
ticker = st.sidebar.text_input('Enter Ticker Symbol', 'AAPL')
growth_rate = st.sidebar.number_input('Enter Growth Rate (%)', value=5.0)

# Analyze Stock
if st.sidebar.button('Analyze Stock'):
    try:
        stock_analysis, stock_hist = analyze_stock(ticker, growth_rate / 100)
        st.subheader(f'Stock Analysis for {ticker}')
        
        # Display Fair Value Metrics
        st.subheader('Fair Value Metrics')
        st.write(f"Graham Valuation: {stock_analysis['Graham Valuation']}")
        st.write(f"Formula Valuation: {stock_analysis['Formula Valuation']}")
        st.write(f"Peter Lynch Score: {stock_analysis['Peter Lynch Score']}")
        
        # Display Return Metrics
        st.subheader('Return Metrics (Extrapolated for 5 years)')
        st.write(f"Expected Return based on Cost of Equity: {stock_analysis['Cost of Equity'] * 100:.2f}%")
        st.write(f"Expected Return based on Formula Valuation: {((stock_analysis['Formula Valuation'] / stock_hist['Close'].iloc[-1]) ** (1/5) - 1) * 100:.2f}%")
        
        # Display Market Metrics
        st.subheader('Market Metrics')
        st.write(f"Market Cap (Billion $): {stock_analysis['Market Cap (Billion $)']}")
        st.write(f"Profit Margins: {stock_analysis['Profit Margins']}")
        st.write(f"Beta: {stock_analysis['Beta']}")
        st.write(f"Volatility: {stock_analysis['Volatility']:.2f}")
        
        # Display Risk Management Metrics
        st.subheader('Risk Management Metrics')
        st.write(f"Max Drawdown: {stock_analysis['Max Drawdown'] * 100:.2f}%")
        st.write(f"Market Correlation: {stock_analysis['Market Correlation']:.2f}")
        st.write(f"Cost of Equity: {stock_analysis['Cost of Equity']:.4f}")
        
        # Display Current and Historical Closing Prices
        st.subheader('Current and Historical Closing Prices')
        st.line_chart(stock_hist['Close'])
        
        # Display News Sentiment Analysis
        st.subheader('News Sentiment Analysis')
        news_data = get_news_data(ticker)
        sentiments = analyze_sentiment(news_data)
        sentiment_df = pd.DataFrame({'Sentiment': sentiments}, index=[entry[0] for entry in news_data])
        st.line_chart(sentiment_df)
        
    except Exception as e:
        st.write(f"Error occurred: {e}")

# Portfolio Optimization
st.sidebar.title('Portfolio Optimization')
tickers = st.sidebar.text_input('Enter Ticker Symbols (comma separated)', 'AAPL,MSFT,GOOGL')
weights = st.sidebar.text_input('Enter Weights (comma separated)', '0.4,0.4,0.2')

if st.sidebar.button('Optimize Portfolio'):
    try:
        tickers = [ticker.strip() for ticker in tickers.split(',')]
        weights = [float(weight.strip()) for weight in weights.split(',')]
        
        # Fetch historical data for each ticker
        portfolio_data = pd.DataFrame()
        for ticker in tickers:
            hist = fetch_stock_data(ticker)
            if not hist.empty:
                portfolio_data[ticker] = hist['Close']
        
        # Normalize weights
        weights = np.array(weights)
        weights = weights / np.sum(weights)
        
        # Calculate portfolio returns and volatility
        log_returns = np.log(portfolio_data / portfolio_data.shift(1))
        portfolio_returns = np.sum(log_returns.mean() * weights) * 252
        portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(log_returns.cov() * 252, weights)))
        
        st.subheader('Portfolio Optimization Results')
        st.write(f'Expected Portfolio Return: {portfolio_returns * 100:.2f}%')
        st.write(f'Portfolio Volatility: {portfolio_volatility * 100:.2f}%')
        st.write('Optimal Weights:')
        st.write(pd.DataFrame(weights, index=tickers, columns=['Weight']))
        
    except Exception as e:
        st.write(f"Error occurred: {e}")











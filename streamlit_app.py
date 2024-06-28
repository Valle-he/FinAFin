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
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]

    # Calculate average market return (you may need to adjust this calculation based on your data)
    # Example: Using S&P 500 index return as average market return
    average_market_return = sp500['Log Return'].mean() * 252

    # Calculate Cost of Equity
    cost_of_equity = calculate_cost_of_equity(risk_free_rate, beta, average_market_return)
    
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
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Beta': beta,
        'Market Correlation': correlation,
        'Cost of Equity': cost_of_equity,
        'Historical Prices': hist
    }
    
    return analysis

# Function for portfolio optimization
def optimize_portfolio(tickers, min_weight, max_weight):
    # Define date range
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    # Create dataframe for adjusted closing prices
    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        data = yf.download(ticker, start=start_date, end=end_date)
        adj_close_df[ticker] = data['Adj Close']

    # Calculate log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))
    log_returns = log_returns.dropna()

    # Calculate covariance matrix
    cov_matrix = log_returns.cov() * 252

    # Functions for standard deviation, expected return, and Sharpe ratio
    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    # Use FRED API to get current 10-year Treasury rate
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]

    # Optimize Sharpe ratio using an iterative approach
    num_assets = len(tickers)
    num_portfolios = 10000
    results = np.zeros((3, num_portfolios))
    weight_array = np.zeros((num_portfolios, num_assets))

    def objective(weights):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Set boundaries for weights
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

# Sidebar for Stock Analysis Input
st.sidebar.header('Stock Analysis Input')
ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')

if st.sidebar.button("Analyze Stock"):
    # Analyze stock
    if ticker:
        result = analyze_stock(ticker)
        
        st.subheader(f'Stock Analysis Results for {ticker}')
        
        # Market Metrics section
        st.subheader('Market Metrics')
        st.write(f"**Market Cap (Billion $)**: {result['Market Cap (Billion $)']:.2f}")
        st.write(f"**Enterprise Value (Billion $)**: {result['Enterprise Value (Billion $)']:.2f}")
        st.write(f"**Enterprise to Revenue**: {result['Enterprise to Revenue']:.4f}")
        st.write(f"**Enterprise to EBITDA**: {result['Enterprise to EBITDA']:.4f}")
        st.write(f"**Cost of Equity**: {result['Cost of Equity']:.4f}")
        st.write("---")
        
        # Risk Management section
        st.subheader('Risk Management Metrics')
        st.write(f"**Volatility**: {result['Volatility']:.4f}")
        st.write(f"**Max Drawdown**: {result['Max Drawdown']:.4f}")
        st.write(f"**Beta**: {result['Beta']:.4f}")
        st.write(f"**Market Correlation**: {result['Market Correlation']:.4f}")
        st.write("---")
        
        # Sort and group ratios by type
        grouped_ratios = {
            'Valuation Ratios': ['P/E Ratio', 'Forward P/E', 'Price to Sales Ratio', 'P/B Ratio'],
            'Financial Ratios': ['Dividend Yield', 'Trailing Eps', 'Payout Ratio'],
            'Profitability Margins': ['Profit Margins', 'Gross Margins', 'EBITDA Margins', 'Operating Margins'],
            'Financial Metrics': ['Return on Assets (ROA)', 'Return on Equity (ROE)'],
            'Revenue Metrics': ['Revenue Growth', 'Total Revenue (Million $)', 'Total Revenue per Share'],
            'Financial Health': ['Debt to Equity Ratio', 'Current Ratio'],
            'Cashflow Metrics': ['Total Cash (Million $)', 'Operating Cashflow (Million $)', 'Levered Free Cashflow (Million $)'],
        }
        
        for group_name, ratios in grouped_ratios.items():
            st.subheader(group_name)
            for ratio in ratios:
                if result[ratio] is not None:
                    st.write(f"**{ratio}**: {result[ratio]}")
            st.write("---")
        
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

# Sidebar for Portfolio Optimization Input
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







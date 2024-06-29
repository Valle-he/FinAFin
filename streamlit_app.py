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
    
    # Function to calculate Fair Value according to Graham
    def calculate_graham_valuation(ticker, growth_rate):
        # EPS and Risk-Free Rate
        eps = info.get('trailingEps', None)
        if eps is None:
            return None
        graham_valuation = (eps * (8.5 + (2 * growth_rate) * 100) * 4.4) / (risk_free_rate * 100)
        return graham_valuation

    # Function to calculate Fair Value using a different formula (example)
    def calculate_formula_valuation(ticker, growth_rate):
        # EPS
        eps = info.get('trailingEps', None)
        if eps is None:
            return None
        # Calculate using your preferred formula
        formula_valuation = eps * (10 + growth_rate)
        return formula_valuation

    # Function to calculate Expected Return (Fundamental)
    def calculate_expected_return_fundamental(ticker, growth_rate):
        # Implement your method to calculate Expected Return based on fundamentals
        return growth_rate * 100

    # Function to calculate Peter Lynch Score
    def calculate_peter_lynch_score(ticker):
        # Implement your method to calculate Peter Lynch Score
        return 100  # Placeholder value

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
        'Graham Valuation': calculate_graham_valuation(ticker, 0.10),  # Example growth rate of 10%
        'Formula Valuation': calculate_formula_valuation(ticker, 0.10),  # Example growth rate of 10%
        'Expected Return (Fundamental)': calculate_expected_return_fundamental(ticker, 0.10),  # Example growth rate of 10%
        'Peter Lynch Score': calculate_peter_lynch_score(ticker)  # Placeholder value
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

    def expected_return(weights, returns):
        return np.sum(returns.mean() * weights) * 252

    def sharpe_ratio(weights, returns, cov_matrix, risk_free_rate):
        portfolio_return = expected_return(weights, returns)
        portfolio_volatility = standard_deviation(weights, cov_matrix)
        return (portfolio_return - risk_free_rate) / portfolio_volatility

    # Define optimization function
    def portfolio_optimizer(weights, returns, cov_matrix, risk_free_rate):
        return -sharpe_ratio(weights, returns, cov_matrix, risk_free_rate)

    # Set initial weights
    num_assets = len(tickers)
    initial_weights = np.random.uniform(min_weight, max_weight, num_assets)
    initial_weights /= np.sum(initial_weights)

    # Set constraints
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})

    # Set bounds for weights
    bounds = tuple((min_weight, max_weight) for _ in range(num_assets))

    # Run optimization
    optimization_result = minimize(portfolio_optimizer, initial_weights,
                                   args=(log_returns, cov_matrix, risk_free_rate),
                                   method='SLSQP', bounds=bounds, constraints=constraints)

    # Prepare results
    optimized_weights = optimization_result.x
    portfolio_return = expected_return(optimized_weights, log_returns)
    portfolio_volatility = standard_deviation(optimized_weights, cov_matrix)
    sharpe_ratio_value = sharpe_ratio(optimized_weights, log_returns, cov_matrix, risk_free_rate)

    return optimized_weights, portfolio_return, portfolio_volatility, sharpe_ratio_value

# Streamlit UI
st.title('Stock Analysis and Portfolio Optimization')

ticker = st.text_input('Enter Ticker Symbol:')
if st.button('Analyze'):
    analysis = analyze_stock(ticker)
    st.subheader('Stock Analysis')
    st.write(analysis)

    # Portfolio Optimization Section
    st.subheader('Portfolio Optimization')

    tickers = st.text_input('Enter Tickers for Portfolio Optimization (comma-separated):')
    min_weight = st.number_input('Minimum Weight:', min_value=0.0, max_value=1.0, step=0.01, value=0.0)
    max_weight = st.number_input('Maximum Weight:', min_value=0.0, max_value=1.0, step=0.01, value=1.0)

    if st.button('Optimize Portfolio'):
        tickers = tickers.split(',')
        tickers = [ticker.strip() for ticker in tickers]

        weights, portfolio_return, portfolio_volatility, sharpe_ratio_value = optimize_portfolio(tickers, min_weight, max_weight)

        st.write('**Optimized Portfolio Weights:**')
        st.write(pd.DataFrame(weights, index=tickers, columns=['Weight']))

        st.write('**Portfolio Return:**', portfolio_return)
        st.write('**Portfolio Volatility:**', portfolio_volatility)
        st.write('**Sharpe Ratio:**', sharpe_ratio_value)




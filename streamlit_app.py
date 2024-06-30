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

# Function to get current dividend yield
def get_dividend_yield(ticker):
    stock = yf.Ticker(ticker)
    dividend_yield = stock.info.get('dividendYield', None)
    return dividend_yield

# Function to calculate Peter Lynch Score
def calculate_peter_lynch_score(ticker, growth_rate):
    dividend_yield = get_dividend_yield(ticker)
    
    if dividend_yield is None or dividend_yield <= 0:
        return None
    
    stock = yf.Ticker(ticker)
    pe_ratio = stock.info.get('trailingPE', None)
    
    if pe_ratio is None:
        return None
    
    peter_lynch_score = (growth_rate * 100) / (pe_ratio * dividend_yield * 100)
    
    return peter_lynch_score

# Function to calculate Graham Valuation
def calculate_graham_valuation(ticker, growth_rate):
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None
    
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    
    graham_valuation = (eps * (8.5 + (2 * growth_rate) * 100) * 4.4) / (risk_free_rate * 100)
    
    return graham_valuation

# Function to calculate Formula Valuation
def calculate_formula_valuation(ticker, growth_rate):
    stock = yf.Ticker(ticker)
    forward_pe_ratio = stock.info.get('forwardPE', None)
    
    if forward_pe_ratio is None:
        return None
    
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None
    
    sp500 = yf.Ticker('^GSPC').history(period='5y')
    average_market_return = sp500['Close'].pct_change().mean() * 252
    
    formula_valuation = (forward_pe_ratio * eps * ((1 + growth_rate) ** 5)) / ((1 + average_market_return) ** 5)
    
    return formula_valuation

# Function to calculate Expected Return (fundamental)
def calculate_expected_return(ticker, growth_rate):
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None
    
    future_eps = eps * ((1 + growth_rate) ** 5)
    
    forward_pe_ratio = stock.info.get('forwardPE', None)
    
    if forward_pe_ratio is None:
        return None
    
    future_stock_price = forward_pe_ratio * future_eps
    current_stock_price = stock.history(period='1d')['Close'].iloc[-1]
    
    expected_return = ((future_stock_price / current_stock_price) ** (1 / 5) - 1)
    
    return expected_return

# Function to calculate Historical Expected Return
def calculate_expected_return_historical(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    
    data = yf.download(ticker, start=start_date, end=end_date)
    
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    historical_return = log_returns.mean() * 252
    
    return historical_return

# Function to fetch news data using Alpha Vantage API
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

# Function to fetch historical stock data from Yahoo Finance
def fetch_stock_data(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period='5y')
    return hist

# Function to analyze stock based on ticker symbol
def analyze_stock(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    hist = fetch_stock_data(ticker)
    
    hist['Log Return'] = np.log(hist['Close'] / hist['Close'].shift(1))
    volatility = hist['Log Return'].std() * np.sqrt(252)
    
    hist['Cumulative Return'] = (1 + hist['Log Return']).cumprod()
    hist['Cumulative Max'] = hist['Cumulative Return'].cummax()
    hist['Drawdown'] = hist['Cumulative Return'] / hist['Cumulative Max'] - 1
    max_drawdown = hist['Drawdown'].min()
    
    sp500 = yf.Ticker('^GSPC').history(period='5y')
    sp500['Log Return'] = np.log(sp500['Close'] / sp500['Close'].shift(1))
    common_index = hist['Log Return'].index.intersection(sp500['Log Return'].index)
    hist_aligned = hist.loc[common_index, 'Log Return'].dropna()
    sp500_aligned = sp500.loc[common_index, 'Log Return'].dropna()

    if len(hist_aligned) != len(sp500_aligned):
        return {"Error": "Data lengths do not match"}

    try:
        covariance = np.cov(hist_aligned, sp500_aligned)[0, 1]
        beta = covariance / sp500_aligned.var()
    except Exception as e:
        covariance = "N/A"
        beta = "N/A"
    
    correlation = hist['Log Return'].corr(sp500['Log Return'])
    
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    
    average_market_return = sp500['Log Return'].mean() * 252
    cost_of_equity = risk_free_rate + beta * (average_market_return - risk_free_rate)
    
    growth_rate = 0.10
    peter_lynch_score = calculate_peter_lynch_score(ticker, growth_rate)
    graham_valuation = calculate_graham_valuation(ticker, growth_rate)
    formula_valuation = calculate_formula_valuation(ticker, growth_rate)
    expected_return = calculate_expected_return(ticker, growth_rate)
    historical_expected_return = calculate_expected_return_historical(ticker)
    
    analysis = {
        'Ticker': ticker,
        'P/E Ratio': info.get('trailingPE'),
        'Forward P/E': info.get('forwardPE'),
        'P/S Ratio': info.get('priceToSalesTrailing12Months'),
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
        'EPS Growth': info.get('earningsGrowth'),
        'Volatility (1y)': volatility,
        'Max Drawdown (1y)': max_drawdown,
        'Beta': beta,
        'Correlation with S&P 500': correlation,
        'Cost of Equity': cost_of_equity,
        'Peter Lynch Score': peter_lynch_score,
        'Graham Valuation': graham_valuation,
        'Formula Valuation': formula_valuation,
        'Expected Return (Fundamental)': expected_return,
        'Expected Return (Historical)': historical_expected_return,
    }
    
    return analysis

# Function to optimize portfolio
def optimize_portfolio(returns):
    n_assets = len(returns.columns)
    args = (returns.mean(), np.diag(returns.cov()))
    bounds = tuple((0, 1) for asset in range(n_assets))
    constraints = ({'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1})
    initial_weights = n_assets * [1 / n_assets]
    result = minimize(minimize_volatility, initial_weights, args=args, bounds=bounds, constraints=constraints)
    return result.x

# Helper function for portfolio optimization
def minimize_volatility(weights, mean_returns, cov_matrix):
    portfolio_return = np.sum(mean_returns * weights)
    portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return portfolio_std_dev

# Main Streamlit app
def main():
    st.title('Stock Analysis and Portfolio Optimization')
    
    # Input sidebar
    st.sidebar.header('User Inputs')
    ticker_list = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
    selected_tickers = st.sidebar.multiselect('Select Tickers', ticker_list, default=['AAPL'])
    
    # Display selected tickers
    st.sidebar.header('Selected Tickers')
    for ticker in selected_tickers:
        st.sidebar.write(ticker)
    
    # Analyze each selected stock
    for ticker in selected_tickers:
        st.subheader(f'Analysis for {ticker}')
        try:
            analysis = analyze_stock(ticker)
            
            if 'Error' in analysis:
                st.error(analysis['Error'])
            else:
                st.write('**Company Information**')
                st.write(f"**Sector:** {analysis['Sector']}")
                st.write(f"**Industry:** {analysis['Industry']}")
                st.write(f"**Market Cap (Billion $):** {analysis['Market Cap (Billion $)']:.2f}")
                st.write(f"**Enterprise Value (Billion $):** {analysis['Enterprise Value (Billion $)']:.2f}")
                st.write(f"**Volatility (1y):** {analysis['Volatility (1y)']:.2%}")
                st.write(f"**Max Drawdown (1y):** {analysis['Max Drawdown (1y)']:.2%}")
                st.write(f"**Beta:** {analysis['Beta']:.2f}")
                st.write(f"**Correlation with S&P 500:** {analysis['Correlation with S&P 500']:.2f}")
                st.write(f"**Cost of Equity:** {analysis['Cost of Equity']:.2%}")
                st.write(f"**Peter Lynch Score:** {analysis['Peter Lynch Score']:.2f}")
                st.write(f"**Graham Valuation:** {analysis['Graham Valuation']:.2f}")
                st.write(f"**Formula Valuation:** {analysis['Formula Valuation']:.2f}")
                st.write(f"**Expected Return (Fundamental):** {analysis['Expected Return (Fundamental)']:.2%}")
                st.write(f"**Expected Return (Historical):** {analysis['Expected Return (Historical)']:.2%}")
                
                st.write('**Portfolio Optimization**')
                st.write('Optimizing portfolio based on selected tickers...')
                
                # Fetch historical returns for selected tickers
                returns = pd.DataFrame()
                for selected_ticker in selected_tickers:
                    hist_data = fetch_stock_data(selected_ticker)
                    hist_data['Log Return'] = np.log(hist_data['Close'] / hist_data['Close'].shift(1))
                    returns[selected_ticker] = hist_data['Log Return']
                
                # Perform portfolio optimization
                weights = optimize_portfolio(returns)
                st.write('**Optimal Portfolio Weights**')
                for i, selected_ticker in enumerate(selected_tickers):
                    st.write(f"{selected_ticker}: {weights[i]:.2%}")
                
        except Exception as e:
            st.error(f"Error analyzing {ticker}: {str(e)}")

if __name__ == '__main__':
    main()











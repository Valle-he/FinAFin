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
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]

    # Calculate average market return (you may need to adjust this calculation based on your data)
    # Example: Using S&P 500 index return as average market return
    average_market_return = sp500['Log Return'].mean() * 252

    # Calculate Cost of Equity
    cost_of_equity = calculate_cost_of_equity(risk_free_rate, beta, average_market_return)
    
    # Function to calculate Peter Lynch Valuation Score
    def calculate_peter_lynch_score(growth_rate, dividend_yield, pe_ratio):
        if dividend_yield is None or dividend_yield <= 0 or pe_ratio is None:
            return None
        return (growth_rate * 100) / (pe_ratio * dividend_yield * 100)
    
    # Function to calculate Graham Valuation
    def calculate_graham_valuation(growth_rate, eps):
        if eps is None:
            return None
        graham_valuation = (eps * (8.5 + (2 * growth_rate) * 100) * 4.4) / (risk_free_rate * 100)
        return graham_valuation
    
    # Function to calculate Formula Valuation
    def calculate_formula_valuation(growth_rate, forward_pe_ratio, eps):
        if forward_pe_ratio is None or eps is None:
            return None
        sp500 = yf.Ticker('^GSPC').history(period='5y')
        average_market_return = sp500['Close'].pct_change().mean() * 252
        formula_valuation = (forward_pe_ratio * eps * ((1 + growth_rate) ** 5)) / ((1 + average_market_return) ** 5)
        return formula_valuation
    
    # Calculate Peter Lynch Score
    dividend_yield = info.get('dividendYield', None)
    pe_ratio = info.get('trailingPE', None)
    peter_lynch_score = calculate_peter_lynch_score(growth_rate, dividend_yield, pe_ratio)
    
    # Calculate Graham Valuation
    eps = info.get('trailingEps', None)
    graham_valuation = calculate_graham_valuation(growth_rate, eps)
    
    # Calculate Formula Valuation
    forward_pe_ratio = info.get('forwardPE', None)
    formula_valuation = calculate_formula_valuation(growth_rate, forward_pe_ratio, eps)
    
    analysis = {
        'Ticker': ticker,
        'Volatility': volatility,
        'Max Drawdown': max_drawdown,
        'Beta': beta,
        'Market Correlation': correlation,
        'Cost of Equity': cost_of_equity,
        'P/E Ratio': info.get('trailingPE'),
        'Forward P/E': info.get('forwardPE'),
        'Price to Sales Ratio': info.get('priceToSalesTrailing12Months'),
        'P/B Ratio': info.get('priceToBook'),
        'Dividend Yield': dividend_yield,
        'Trailing Eps': eps,
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
        'Historical Prices': hist,
        'Peter Lynch Score': peter_lynch_score,
        'Graham Valuation': graham_valuation,
        'Formula Valuation': formula_valuation
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
        data = yf.download(ticker, start=start_date, end=end_date)['Adj Close']
        adj_close_df[ticker] = data

    # Calculate log returns
    log_returns = np.log(adj_close_df / adj_close_df.shift(1))

    # Mean daily return and covariance of daily returns
    mean_daily_returns = log_returns.mean()
    cov_matrix = log_returns.cov()

    # Function to minimize negative Sharpe ratio
    def neg_sharpe_ratio(weights):
        portfolio_return = np.sum(mean_daily_returns * weights) * 252
        portfolio_std_dev = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights))) * np.sqrt(252)
        sharpe_ratio = portfolio_return / portfolio_std_dev
        return -sharpe_ratio

    # Constraints
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})

    # Bounds
    bounds = tuple((min_weight, max_weight) for _ in range(len(tickers)))

    # Initial guess (equal weight)
    init_guess = [1.0 / len(tickers) for _ in range(len(tickers))]

    # Optimize portfolio
    optimized_results = minimize(neg_sharpe_ratio, init_guess, method='SLSQP', bounds=bounds, constraints=constraints)

    optimized_weights = optimized_results.x
    optimized_weights = [round(x, 4) for x in optimized_weights]

    return optimized_weights

# Streamlit UI
st.title('Stock Analysis Tool')

# Sidebar - Input parameters
st.sidebar.title('Input Parameters')
ticker = st.sidebar.text_input('Enter Ticker Symbol', 'AAPL')
growth_rate = st.sidebar.number_input('Enter Growth Rate (%)', value=10.0)
min_weight = st.sidebar.number_input('Minimum Weight for Optimization', value=0.0)
max_weight = st.sidebar.number_input('Maximum Weight for Optimization', value=1.0)

# Fetch stock data and analysis
if st.button('Analyze'):
    try:
        stock_analysis = analyze_stock(ticker, growth_rate / 100)
        st.subheader('Company Information')
        st.write(stock_analysis['Sector'], '-', stock_analysis['Industry'])
        st.write('Country:', stock_analysis['Country'])
        st.write('Website:', stock_analysis['Website'])

        st.subheader('Stock Performance Metrics')
        st.write('Volatility (Annualized):', round(stock_analysis['Volatility'], 4))
        st.write('Max Drawdown:', round(stock_analysis['Max Drawdown'] * 100, 2), '%')
        st.write('Beta (Market Correlation):', round(stock_analysis['Beta'], 4))
        st.write('Market Correlation:', round(stock_analysis['Market Correlation'], 4))
        st.write('Cost of Equity:', round(stock_analysis['Cost of Equity'] * 100, 2), '%')

        st.subheader('Stock Ratios')
        st.write('P/E Ratio (Trailing):', stock_analysis['P/E Ratio'])
        st.write('Forward P/E Ratio:', stock_analysis['Forward P/E'])
        st.write('Price to Sales Ratio (P/S):', stock_analysis['Price to Sales Ratio'])
        st.write('Price to Book Ratio (P/B):', stock_analysis['P/B Ratio'])
        st.write('Dividend Yield:', round(stock_analysis['Dividend Yield'] * 100, 2), '%')
        st.write('Trailing EPS:', stock_analysis['Trailing Eps'])
        
        st.subheader('Fair Value Metrics')
        st.write('Peter Lynch Score:', round(stock_analysis['Peter Lynch Score'], 2))
        st.write('Graham Valuation:', round(stock_analysis['Graham Valuation'], 2))
        st.write('Formula Valuation:', round(stock_analysis['Formula Valuation'], 2))

        st.subheader('Return Metrics')
        st.write('Return on Assets (ROA):', round(stock_analysis['Return on Assets (ROA)'] * 100, 2), '%')
        st.write('Return on Equity (ROE):', round(stock_analysis['Return on Equity (ROE)'] * 100, 2), '%')

        st.subheader('Historical Prices')
        st.line_chart(stock_analysis['Historical Prices']['Close'])

    except Exception as e:
        st.write('Error:', e)

# Portfolio Optimization
st.sidebar.title('Portfolio Optimization')
num_stocks = st.sidebar.number_input('Number of Stocks in Portfolio', min_value=2, max_value=10, value=5)
if st.sidebar.button('Optimize Portfolio'):
    try:
        tickers = st.sidebar.text_input('Enter Ticker Symbols (comma separated)', value='AAPL,MSFT,GOOGL,AMZN,TSLA')
        tickers = [ticker.strip().upper() for ticker in tickers.split(',')]
        optimized_weights = optimize_portfolio(tickers, min_weight, max_weight)
        
        st.subheader('Optimized Portfolio Weights')
        for i in range(len(tickers)):
            st.write(tickers[i], ':', optimized_weights[i])
        
        # Display pie chart of portfolio allocation
        fig = px.pie(names=tickers, values=optimized_weights, title='Portfolio Allocation')
        st.plotly_chart(fig)

    except Exception as e:
        st.write('Error:', e)

# News Sentiment Analysis
st.sidebar.title('News Sentiment Analysis')
if st.sidebar.button('Get News Sentiment'):
    try:
        news_data = get_news_data(ticker)
        sentiments = analyze_sentiment(news_data)
        
        st.subheader('Latest News Sentiment')
        for i in range(len(news_data)):
            st.write('Published At:', news_data[i][0])
            st.write('Title:', news_data[i][1])
            st.write('Summary:', news_data[i][2])
            st.write('Sentiment Score:', round(sentiments[i], 2))
            st.write('---')

    except Exception as e:
        st.write('Error:', e)












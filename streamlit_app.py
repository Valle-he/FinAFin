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

# Funktion zur Berechnung der aktuellen Dividendenrendite
def get_dividend_yield(ticker):
    try:
        stock = yf.Ticker(ticker)
        dividend_yield = stock.info.get('dividendYield', None)
        if dividend_yield is None:
            st.warning(f"No dividend yield available for {ticker}")
            return None
        return dividend_yield
    except Exception as e:
        st.error(f"Error fetching dividend yield for {ticker}: {str(e)}")
        return None


# Funktion zur Berechnung des Peter Lynch Valuation Scores
def calculate_peter_lynch_score(ticker, growth_rate):
    dividend_yield = get_dividend_yield(ticker)
    if dividend_yield is None or dividend_yield <= 0:
        st.warning(f"Invalid dividend yield for {ticker}")
        return None

    stock = yf.Ticker(ticker)
    pe_ratio = stock.info.get('trailingPE', None)
    if pe_ratio is None:
        st.warning(f"P/E ratio not available for {ticker}")
        return None

    try:
        peter_lynch_score = (growth_rate * 100) / (pe_ratio * dividend_yield * 100)
    except ZeroDivisionError:
        st.error("Division by zero encountered in Peter Lynch score calculation")
        return None

    return peter_lynch_score

# Funktion zur Berechnung des Fair Value nach Graham
def calculate_graham_valuation(ticker, growth_rate):
    # EPS abrufen
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None  # Wenn EPS nicht verfügbar, kein Fair Value berechnen
    
    # Risk-Free Rate über FRED API abrufen
    fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
    ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
    risk_free_rate = ten_year_treasury_rate.iloc[-1]
    
    # Fair Value nach Graham Formel berechnen
    graham_valuation = (eps * (8.5 + (2 * growth_rate) * 100) * 4.4) / (risk_free_rate * 100)
    
    return graham_valuation

# Funktion zur Berechnung des Fair Value nach Formel
def calculate_formula_valuation(ticker, growth_rate):
    # Forward P/E Ratio abrufen
    stock = yf.Ticker(ticker)
    forward_pe_ratio = stock.info.get('forwardPE', None)
    
    if forward_pe_ratio is None:
        return None  # Wenn Forward P/E Ratio nicht verfügbar, kein Fair Value berechnen
    
    # EPS abrufen
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None  # Wenn EPS nicht verfügbar, kein Fair Value berechnen
    
    # Durchschnittlicher Markt Return
    sp500 = yf.Ticker('^GSPC').history(period='5y')
    average_market_return = sp500['Close'].pct_change().mean() * 252
    
    # Fair Value nach Formel berechnen
    formula_valuation = (forward_pe_ratio * eps * ((1 + growth_rate) ** 5)) / ((1 + average_market_return) ** 5)
    
    return formula_valuation

# Funktion zur Berechnung des Expected Return (fundamental)
def calculate_expected_return(ticker, growth_rate):
    # EPS abrufen
    stock = yf.Ticker(ticker)
    eps = stock.info.get('trailingEps', None)
    
    if eps is None:
        return None  # Wenn EPS nicht verfügbar, kein Expected Return berechnen
    
    # Gewinn in 5 Jahren berechnen (Extrapolationszeitraum festgelegt auf 5 Jahre)
    future_eps = eps * ((1 + growth_rate) ** 5)
    
    # Programmierter Preis der Aktie in 5 Jahren (prog Kgv = Forward P/E Ratio)
    forward_pe_ratio = stock.info.get('forwardPE', None)
    
    if forward_pe_ratio is None:
        return None  # Wenn Forward P/E Ratio nicht verfügbar, kein Expected Return berechnen
    
    future_stock_price = forward_pe_ratio * future_eps
    
    # Aktueller Preis der Aktie
    current_stock_price = stock.history(period='1d')['Close'].iloc[-1]
    
    # Expected Return berechnen
    expected_return = ((future_stock_price / current_stock_price) ** (1 / 5) - 1) 
    
    return expected_return

# Funktion zur Berechnung des Expected Return (historical)
def calculate_expected_return_historical(ticker):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)
    
    # Daten von Yahoo Finance abrufen
    data = yf.download(ticker, start=start_date, end=end_date)
    
    # Log-Renditen berechnen
    log_returns = np.log(data['Adj Close'] / data['Adj Close'].shift(1))
    
    # Historischen Expected Return berechnen
    historical_return = log_returns.mean() * 252 # in Prozent umrechnen
    
    return historical_return

# Function to fetch news data using Alpha Vantage News API
def get_news_data(ticker):
    url = f'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers={ticker}&apikey={alpha_vantage_api_key}'
    response = requests.get(url)
    try:
        response.raise_for_status()
        news_data = response.json()
    except requests.exceptions.RequestException as e:
        st.error(f"Error fetching news data: {str(e)}")
        return []
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
    # Example alignment (adjust according to your data)
    common_index = hist['Log Return'].index.intersection(sp500['Log Return'].index)
    hist_aligned = hist.loc[common_index, 'Log Return'].dropna()
    sp500_aligned = sp500.loc[common_index, 'Log Return'].dropna()

# Calculate covariance
    if len(hist_aligned) != len(sp500_aligned):
        return {"Error": "Data lengths do not match"}

        # Calculate covariance and beta
    try:
        covariance = np.cov(hist_aligned, sp500_aligned)[0, 1]
        beta = covariance / sp500_aligned.var()
    except Exception as e:
        covariance = "N/A"
        beta = "N/A"
    
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
    
    # Calculate valuation metrics
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
        'Peter Lynch Score': peter_lynch_score,
        'Graham Valuation': graham_valuation,
        'Formula Valuation': formula_valuation,
        'Expected Return (Fundamental)': expected_return,
        'Historical Expected Return': historical_expected_return,
        'Historical Prices': hist
        'Audit Risk': info.get('auditRisk'),
'Board Risk': info.get('boardRisk'),
'Compensation Risk': info.get('compensationRisk'),
'Shareholder Rights Risk': info.get('shareHolderRightsRisk'),
'Overall Risk': info.get('overallRisk'),

'Price Hint': info.get('priceHint'),
'Previous Close': info.get('previousClose'),
'Open': info.get('open'),
'Day Low': info.get('dayLow'),
'Day High': info.get('dayHigh'),
'Regular Market Previous Close': info.get('regularMarketPreviousClose'),
'Regular Market Open': info.get('regularMarketOpen'),
'Regular Market Day Low': info.get('regularMarketDayLow'),
'Regular Market Day High': info.get('regularMarketDayHigh'),

'Ex-Dividend Date': info.get('exDividendDate'),
'Five Year Average Dividend Yield': info.get('fiveYearAvgDividendYield'),

'Volume': info.get('volume'),
'Regular Market Volume': info.get('regularMarketVolume'),
'Average Volume': info.get('averageVolume'),
'Average Volume 10 Days': info.get('averageVolume10days'),
'Average Daily Volume 10 Day': info.get('averageDailyVolume10Day'),
'Bid': info.get('bid'),
'Ask': info.get('ask'),
'Bid Size': info.get('bidSize'),
'Ask Size': info.get('askSize'),

'Fifty-Two Week Low': info.get('fiftyTwoWeekLow'),
'Fifty-Two Week High': info.get('fiftyTwoWeekHigh'),
'Price to Sales Trailing 12 Months': info.get('priceToSalesTrailing12Months'),
'Fifty Day Average': info.get('fiftyDayAverage'),
'Two Hundred Day Average': info.get('twoHundredDayAverage'),
'Currency': info.get('currency'),

'Float Shares': info.get('floatShares'),
'Shares Outstanding': info.get('sharesOutstanding'),
'Shares Short': info.get('sharesShort'),
'Shares Short Prior Month': info.get('sharesShortPriorMonth'),
'Shares Short Previous Month Date': info.get('sharesShortPreviousMonthDate'),
'Date Short Interest': info.get('dateShortInterest'),
'Shares Percent Shares Out': info.get('sharesPercentSharesOut'),
'Held Percent Insiders': info.get('heldPercentInsiders'),
'Held Percent Institutions': info.get('heldPercentInstitutions'),
'Short Ratio': info.get('shortRatio'),
'Short Percent of Float': info.get('shortPercentOfFloat'),
'Implied Shares Outstanding': info.get('impliedSharesOutstanding'),
'Book Value': info.get('bookValue'),
'Price to Book': info.get('priceToBook'),

'Net Income to Common': info.get('netIncomeToCommon'),
'Trailing EPS': info.get('trailingEps'),
'Forward EPS': info.get('forwardEps'),
'PEG Ratio': info.get('pegRatio'),
'Last Split Factor': info.get('lastSplitFactor'),

'Enterprise to Revenue': info.get('enterpriseToRevenue'),
'Enterprise to EBITDA': info.get('enterpriseToEbitda'),
'52 Week Change': info.get('52WeekChange'),
'S&P 52 Week Change': info.get('SandP52WeekChange'),
'Last Dividend Value': info.get('lastDividendValue'),
'Last Dividend Date': info.get('lastDividendDate'),
'Exchange': info.get('exchange'),

'Symbol': info.get('symbol'),
'Underlying Symbol': info.get('underlyingSymbol'),
'Short Name': info.get('shortName'),
'Long Name': info.get('longName'),
'First Trade Date Epoch UTC': info.get('firstTradeDateEpochUtc'),
'Time Zone Full Name': info.get('timeZoneFullName'),
'Time Zone Short Name': info.get('timeZoneShortName'),
'UUID': info.get('uuid'),
'Message Board ID': info.get('messageBoardId'),
'GMT Offset Milliseconds': info.get('gmtOffSetMilliseconds'),
'Current Price': info.get('currentPrice'),
'Target High Price': info.get('targetHighPrice'),
'Target Low Price': info.get('targetLowPrice'),
'Target Mean Price': info.get('targetMeanPrice'),
'Target Median Price': info.get('targetMedianPrice'),
'Recommendation Mean': info.get('recommendationMean'),
'Recommendation Key': info.get('recommendationKey'),
'Number of Analyst Opinions': info.get('numberOfAnalystOpinions'),

'Total Cash Per Share': info.get('totalCashPerShare'),
'EBITDA': info.get('ebitda'),
'Total Debt': info.get('totalDebt'),
'Quick Ratio': info.get('quickRatio'),

'Total Revenue': info.get('totalRevenue'),
'Free Cashflow': info.get('freeCashflow'),

'EBITDA Margins': info.get('ebitdaMargins'),
'Operating Margins': info.get('operatingMargins'),
'Financial Currency': info.get('financialCurrency')
  
    }

    
    return analysis

# Function for portfolio optimization
def optimize_portfolio(tickers, min_weight, max_weight):
    end_date = datetime.today()
    start_date = end_date - timedelta(days=5*365)

    adj_close_df = pd.DataFrame()

    for ticker in tickers:
        try:
            data = yf.download(ticker, start=start_date, end=end_date)
            adj_close_df[ticker] = data['Adj Close']
        except Exception as e:
            st.error(f"Error downloading data for {ticker}: {str(e)}")
            return None, None, None, None, None

    log_returns = np.log(adj_close_df / adj_close_df.shift(1)).dropna()
    cov_matrix = log_returns.cov() * 252

    def standard_deviation(weights, cov_matrix):
        variance = weights.T @ cov_matrix @ weights
        return np.sqrt(variance)

    def expected_return(weights, log_returns):
        return np.sum(log_returns.mean() * weights) * 252

    def sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate):
        return (expected_return(weights, log_returns) - risk_free_rate) / standard_deviation(weights, cov_matrix)

    try:
        fred = Fred(api_key='2bbf1ed4d0b03ad1f325efaa03312596')
        ten_year_treasury_rate = fred.get_series_latest_release('GS10') / 100
        risk_free_rate = ten_year_treasury_rate.iloc[-1]
    except Exception as e:
        st.error(f"Error fetching risk-free rate: {str(e)}")
        return None, None, None, None, None

    num_assets = len(tickers)
    results = np.zeros((3, 10000))
    weight_array = np.zeros((10000, num_assets))

    def objective(weights):
        return -sharpe_ratio(weights, log_returns, cov_matrix, risk_free_rate)

    constraints = {'type': 'eq', 'fun': lambda weights: np.sum(weights) - 1}
    bounds = [(min_weight / 100, max_weight / 100)] * num_assets

    try:
        optimized = minimize(objective, num_assets * [1. / num_assets], method='SLSQP', bounds=bounds, constraints=constraints)
        optimal_weights = optimized['x']
        optimal_portfolio_return = expected_return(optimal_weights, log_returns)
        optimal_portfolio_volatility = standard_deviation(optimal_weights, cov_matrix)
        optimal_sharpe_ratio = sharpe_ratio(optimal_weights, log_returns, cov_matrix, risk_free_rate)
    except Exception as e:
        st.error(f"Error optimizing portfolio: {str(e)}")
        return None, None, None, None, None

    return optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, adj_close_df


# Streamlit App
st.title('Stock and Portfolio Analysis')

# Sidebar for Stock Analysis Input
st.sidebar.title('Stock and Portfolio Analysis')

st.sidebar.header('Stock Analysis Input')
ticker = st.sidebar.text_input('Enter the stock ticker:', 'AAPL')
if not ticker.isalpha():
    st.error("Invalid ticker symbol. Please enter a valid stock ticker.")


if st.sidebar.button("Analyze Stock"):
    if ticker:
        try:
            result = analyze_stock(ticker)
            if 'Error' in result:
                st.error(result['Error'])
            else:
                st.subheader(f'Stock Analysis Results for {ticker}')
                # Sort and group ratios by type, display analysis...
        except Exception as e:
            st.error(f"Error analyzing stock: {str(e)}")
        
        # Sort and group ratios by type
        grouped_ratios = {
           
            'Company Information':['Sector','Idustry','Full Time Employees','City','State','Country','Website'],
            'Valuation Ratios': ['P/E Ratio', 'Forward P/E', 'P/S Ratio', 'P/B Ratio'],
            'Financial Ratios': ['Dividend Yield', 'Trailing Eps', 'Payout Ratio'],
            'Profitability Margins': ['Profit Margins', 'Gross Margins', 'EBITDA Margins', 'Operating Margins'],
            'Financial Metrics': ['Return on Assets (ROA)', 'Return on Equity (ROE)'],
            'Revenue Metrics': ['Revenue Growth', 'Total Revenue (Million $)', 'Total Revenue per Share','Gross Profits'],
            'Financial Health': ['Total Debt (Million $)', 'Debt to Equity Ratio', 'Current Ratio'],
            'Cashflow Metrics': ['Total Cash (Million $)', 'Operating Cashflow (Million $)', 'Levered Free Cashflow (Million $)'],
        }
        
        for group_name, ratios in grouped_ratios.items():
            st.subheader(group_name)
            for ratio in ratios:
                if ratio in result and result[ratio] is not None:
                    st.write(f"**{ratio}**: {result[ratio]}")
                else:
                    st.write(f"**{ratio}**: N/A")
            st.write("---")
        
        # Risk Management section
        # Risk Management section
        st.subheader('Risk Management Metrics')

        if 'Volatility' in result and result['Volatility'] is not None:
            st.write(f"**Volatility**: {result['Volatility']:.4f}")
        else:
            st.write("**Volatility**: N/A")

        if 'Max Drawdown' in result and result['Max Drawdown'] is not None:
            st.write(f"**Max Drawdown**: {result['Max Drawdown']:.4f}")
        else:
            st.write("**Max Drawdown**: N/A")

        if 'Beta' in result and result['Beta'] is not None:
            st.write(f"**Beta**: {result['Beta']:.4f}")
        else:
           st.write("**Beta**: N/A")

        if 'Market Correlation' in result and result['Market Correlation'] is not None:
           st.write(f"**Market Correlation**: {result['Market Correlation']:.4f}")
        else:
           st.write("**Market Correlation**: N/A")
        st.write("---")

        
        # Market Metrics section
        # Market Metrics section
       

        # Market Metrics section
        st.subheader('Market Metrics')

        if 'Market Cap (Billion $)' in result and result['Market Cap (Billion $)'] is not None:
            st.write(f"**Market Cap (Billion $)**: {result['Market Cap (Billion $)']:.2f}")
        else:
            st.write("**Market Cap (Billion $)**: N/A")

        if 'Enterprise Value (Billion $)' in result and result['Enterprise Value (Billion $)'] is not None:
            st.write(f"**Enterprise Value (Billion $)**: {result['Enterprise Value (Billion $)']:.2f}")
        else:
            st.write("**Enterprise Value (Billion $)**: N/A")

        if 'Enterprise to Revenue' in result and result['Enterprise to Revenue'] is not None:
            st.write(f"**Enterprise to Revenue**: {result['Enterprise to Revenue']:.4f}")
        else:
            st.write("**Enterprise to Revenue**: N/A")

        if 'Enterprise to EBITDA' in result and result['Enterprise to EBITDA'] is not None:
            st.write(f"**Enterprise to EBITDA**: {result['Enterprise to EBITDA']:.4f}")
        else:
            st.write("**Enterprise to EBITDA**: N/A")

        if 'Cost of Equity' in result and result['Cost of Equity'] is not None:
            st.write(f"**Cost of Equity**: {result['Cost of Equity']:.4f}")
        else:
            st.write("**Cost of Equity**: N/A")
        st.write("---")


        
        # Valuation Metrics section
        st.subheader('Valuation Metrics')
        if 'Peter Lynch Score' in result and result['Peter Lynch Score'] is not None:
            st.write(f"**Peter Lynch Score**: {result['Peter Lynch Score']:.2f}")
        else:
            st.write("**Peter Lynch Score**: N/A")

        if 'Graham Valuation' in result and result['Graham Valuation'] is not None:
            st.write(f"**Graham Valuation**: {result['Graham Valuation']:.2f}")
        else:
            st.write("**Graham Valuation**: N/A")


        if 'Formula Valuation' in result and result['Formula Valuation'] is not None:
            st.write(f"**Formula Valuation**: {result['Formula Valuation']:.2f}")
        else:
            st.write("**Formula Valuation**: N/A")

        if 'Target Price' is not None:
            st.write(f"**Target Price**: {result['Target Price']:.4f}")
        else:
            st.write("**Target Price**: N/A")

        if 'Expected Return (Fundamental)' in result and result['Expected Return (Fundamental)'] is not None:
            st.write(f"**Expected Return (Fundamental)**: {result['Expected Return (Fundamental)']:.4f}")
        else:
            st.write("**Expected Return (Fundamental)**: N/A")


        if result.get('Historical Expected Return') is not None:
            st.write(f"**Historical Return (5 Years Average)**: {result['Historical Expected Return']:.4f}")
        else:
            st.write("**Historical Return (5 Years Average)**: N/A")

        
        st.write("---")
        
        # Display current and historical closing prices
        st.subheader(f'Current and Historical Closing Prices for {ticker}')
        if not result['Historical Prices']['Close'].empty:
            st.write(f"**Current Price**: {result['Historical Prices']['Close'][-1]}")
        else:
           st.write("**Current Price**: Data not available")
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
# Sidebar for Portfolio Optimization Input
st.sidebar.header('Portfolio Optimization Input')
tickers_input = st.sidebar.text_input("Enter the stock tickers separated by commas (e.g., AAPL,GME,SAP,TSLA):", "AAPL,GME,SAP,TSLA")
tickers = [ticker.strip() for ticker in tickers_input.split(',')]

min_weight = st.sidebar.slider('Minimum Weight (%)', min_value=0, max_value=100, value=5)
max_weight = st.sidebar.slider('Maximum Weight (%)', min_value=0, max_value=100, value=30)

if st.sidebar.button("Optimize Portfolio"):
    if not tickers:
        st.error("Please enter at least one valid stock ticker.")
    elif min_weight > max_weight:
        st.error("Minimum weight should be less than or equal to maximum weight.")
    elif min_weight > (100 / len(tickers)):
        st.error(f"Minimum weight should be less than or equal to {100 / len(tickers):.2f}% (1/number of stocks).")
    else:
        try:
            optimal_weights, optimal_portfolio_return, optimal_portfolio_volatility, optimal_sharpe_ratio, adj_close_df = optimize_portfolio(tickers, min_weight, max_weight)
            if optimal_weights is None:
                st.error("Error optimizing portfolio.")
            else:
                st.subheader("Optimal Portfolio Metrics:")
                st.write(f"Expected Annual Return: {optimal_portfolio_return:.4f}")
                st.write(f"Expected Portfolio Volatility: {optimal_portfolio_volatility:.4f}")
                st.write(f"Sharpe Ratio: {optimal_sharpe_ratio:.4f}")

                st.subheader("Optimal Weights:")
                optimal_weights_df = pd.DataFrame(optimal_weights, index=tickers, columns=["Weight"])
                st.write(optimal_weights_df)

                fig = px.pie(optimal_weights_df, values='Weight', names=optimal_weights_df.index, title='Portfolio Allocation')
                st.plotly_chart(fig)

                st.subheader('Current and Historical Closing Prices for Optimized Portfolio')
                optimized_portfolio_prices = (adj_close_df * optimal_weights).sum(axis=1)
                st.line_chart(optimized_portfolio_prices)
        except Exception as e:
            st.error(f"Error optimizing portfolio: {str(e)}")


















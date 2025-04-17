import os
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from dotenv import load_dotenv
import yfinance as yf
import finnhub
from alpha_vantage.timeseries import TimeSeries
import google.generativeai as genai
from groq import Groq

class FinancialAgent:
    def __init__(self):
        """Initialize the FinancialAgent with API configurations"""
        load_dotenv()  # Load environment variables
        
        # Initialize API keys
        self.alpha_vantage_key = os.getenv('ALPHA_VANTAGE_API_KEY')
        self.finnhub_key = os.getenv('FINNHUB_API_KEY')
        self.google_api_key = os.getenv('GOOGLE_API_KEY')
        self.groq_api_key = os.getenv('GROQ_API_KEY')
        
        # Initialize API clients
        self.finnhub_client = finnhub.Client(api_key=self.finnhub_key) if self.finnhub_key else None
        self.alpha_vantage = TimeSeries(key=self.alpha_vantage_key) if self.alpha_vantage_key else None
        
        # Initialize AI models
        if self.google_api_key:
            genai.configure(api_key=self.google_api_key)
            self.gemini_model = genai.GenerativeModel('gemini-pro')
        
        if self.groq_api_key:
            self.groq_client = Groq(api_key=self.groq_api_key)
    
    def get_stock_data(self, symbol: str, start_date: Optional[datetime] = None, end_date: Optional[datetime] = None) -> pd.DataFrame:
        """
        Fetch stock data from multiple sources and combine them
        
        Args:
            symbol: Stock symbol
            start_date: Start date for historical data
            end_date: End date for historical data
        
        Returns:
            DataFrame containing stock data
        """
        try:
            # Default to last 30 days if no dates provided
            if not end_date:
                end_date = datetime.now()
            if not start_date:
                start_date = end_date - timedelta(days=30)
            
            # Get data from yfinance
            yf_data = yf.download(symbol, start=start_date, end=end_date)
            
            # Get additional data from Finnhub if available
            if self.finnhub_client:
                try:
                    company_profile = self.finnhub_client.company_profile2(symbol=symbol)
                    if company_profile:
                        yf_data['Company_Name'] = company_profile.get('name', '')
                        yf_data['Industry'] = company_profile.get('finnhubIndustry', '')
                except Exception as e:
                    print(f"Finnhub API error: {e}")
            
            # Get additional data from Alpha Vantage if available
            if self.alpha_vantage:
                try:
                    av_data, _ = self.alpha_vantage.get_daily(symbol=symbol)
                    # Process and merge relevant Alpha Vantage data
                    av_df = pd.DataFrame.from_dict(av_data, orient='index')
                    av_df.index = pd.to_datetime(av_df.index)
                    av_df = av_df[av_df.index >= start_date]
                    # Add any unique indicators from Alpha Vantage
                except Exception as e:
                    print(f"Alpha Vantage API error: {e}")
            
            return yf_data
        
        except Exception as e:
            print(f"Error fetching stock data: {e}")
            return pd.DataFrame()
    
    def analyze_portfolio(self, portfolio_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Analyze portfolio performance and generate insights
        
        Args:
            portfolio_data: DataFrame containing portfolio information
        
        Returns:
            Dictionary containing analysis results
        """
        try:
            analysis = {
                'total_value': 0,
                'daily_returns': [],
                'risk_metrics': {},
                'recommendations': []
            }
            
            if not portfolio_data.empty:
                # Calculate total portfolio value
                if 'Close' in portfolio_data.columns and 'Shares' in portfolio_data.columns:
                    analysis['total_value'] = (portfolio_data['Close'] * portfolio_data['Shares']).sum()
                
                # Calculate daily returns
                if 'Close' in portfolio_data.columns:
                    daily_returns = portfolio_data['Close'].pct_change()
                    analysis['daily_returns'] = daily_returns.dropna().tolist()
                    
                    # Calculate risk metrics
                    analysis['risk_metrics'] = {
                        'volatility': daily_returns.std() * np.sqrt(252),  # Annualized volatility
                        'sharpe_ratio': (daily_returns.mean() * 252) / (daily_returns.std() * np.sqrt(252)),
                        'max_drawdown': (portfolio_data['Close'] / portfolio_data['Close'].expanding(min_periods=1).max() - 1).min()
                    }
            
            return analysis
        
        except Exception as e:
            print(f"Error analyzing portfolio: {e}")
            return {}
    
    def get_ai_insights(self, data: Dict[str, Any], query: str) -> str:
        """
        Generate AI-powered insights using available models
        
        Args:
            data: Dictionary containing financial data and analysis
            query: User's specific question or analysis request
        
        Returns:
            String containing AI-generated insights
        """
        try:
            # Prepare a summarized version of the data
            data_summary = self._prepare_data_summary(data)
            
            # Try Gemini first
            if hasattr(self, 'gemini_model'):
                try:
                    response = self.gemini_model.generate_content(
                        f"Financial Analysis Request: {query}\nData Summary: {data_summary}"
                    )
                    return response.text
                except Exception as e:
                    print(f"Gemini API error: {e}")
            
            # Fallback to Groq if available
            if hasattr(self, 'groq_client'):
                try:
                    completion = self.groq_client.chat.completions.create(
                        model="mixtral-8x7b-32768",
                        messages=[
                            {"role": "system", "content": "You are a financial analysis expert. Provide concise, actionable insights based on the data provided."},
                            {"role": "user", "content": f"Based on this data summary: {data_summary}\n\nAnswer this question: {query}"}
                        ],
                        max_tokens=1000,
                        temperature=0.7
                    )
                    return completion.choices[0].message.content
                except Exception as e:
                    print(f"Groq API error: {e}")
            
            return "AI analysis currently unavailable. Please check your API configurations."
            
        except Exception as e:
            print(f"Error generating AI insights: {e}")
            return "Error generating insights. Please try again later."
    
    def _prepare_data_summary(self, data: Dict[str, Any]) -> str:
        """
        Prepare a concise summary of the data for AI analysis
        
        Args:
            data: Dictionary containing financial data and analysis
        
        Returns:
            String containing a summarized version of the data
        """
        try:
            summary = []
            
            # If data is a DataFrame converted to dict
            if isinstance(data, dict) and 'index' in data and 'columns' in data:
                df = pd.DataFrame(data)
                
                # Basic dataset info
                summary.append(f"Dataset contains {len(df)} rows and {len(df.columns)} columns")
                
                # Column types
                num_cols = df.select_dtypes(include=['number']).columns
                cat_cols = df.select_dtypes(exclude=['number']).columns
                
                if len(num_cols) > 0:
                    # Numerical columns summary
                    summary.append("\nNumerical Columns Summary:")
                    for col in num_cols[:5]:  # Limit to first 5 columns
                        stats = df[col].describe()
                        summary.append(f"{col}: mean={stats['mean']:.2f}, min={stats['min']:.2f}, max={stats['max']:.2f}")
                
                if len(cat_cols) > 0:
                    # Categorical columns summary
                    summary.append("\nCategorical Columns Summary:")
                    for col in cat_cols[:5]:  # Limit to first 5 columns
                        unique_vals = df[col].nunique()
                        summary.append(f"{col}: {unique_vals} unique values")
            
            # If data is analysis results
            else:
                if 'total_value' in data:
                    summary.append(f"Total Portfolio Value: ${data['total_value']:,.2f}")
                
                if 'risk_metrics' in data:
                    metrics = data['risk_metrics']
                    summary.append("\nRisk Metrics:")
                    summary.append(f"- Volatility: {metrics.get('volatility', 0):.2%}")
                    summary.append(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                    summary.append(f"- Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            return "\n".join(summary)
        
        except Exception as e:
            print(f"Error preparing data summary: {e}")
            return "Error preparing data summary"
    
    def process_financial_documents(self, file_path: str) -> pd.DataFrame:
        """
        Process uploaded financial documents (PDF, Excel, etc.)
        
        Args:
            file_path: Path to the financial document
        
        Returns:
            DataFrame containing extracted financial data
        """
        try:
            if file_path.endswith('.xlsx') or file_path.endswith('.xls'):
                return pd.read_excel(file_path)
            elif file_path.endswith('.csv'):
                return pd.read_csv(file_path)
            elif file_path.endswith('.pdf'):
                import pdfplumber
                
                data = []
                with pdfplumber.open(file_path) as pdf:
                    for page in pdf.pages:
                        text = page.extract_text()
                        # Add basic parsing logic here
                        # This is a simplified version - you might want to add more sophisticated parsing
                        data.append({'page': page.page_number, 'content': text})
                
                return pd.DataFrame(data)
            else:
                raise ValueError("Unsupported file format")
        
        except Exception as e:
            print(f"Error processing document: {e}")
            return pd.DataFrame()
    
    def generate_report(self, analysis_data: Dict[str, Any]) -> str:
        """
        Generate a comprehensive financial report
        
        Args:
            analysis_data: Dictionary containing analysis results
        
        Returns:
            String containing the formatted report
        """
        try:
            report = []
            report.append("Financial Analysis Report")
            report.append("=" * 50)
            
            if 'total_value' in analysis_data:
                report.append(f"\nTotal Portfolio Value: ${analysis_data['total_value']:,.2f}")
            
            if 'risk_metrics' in analysis_data:
                report.append("\nRisk Metrics:")
                metrics = analysis_data['risk_metrics']
                report.append(f"- Volatility: {metrics.get('volatility', 0):.2%}")
                report.append(f"- Sharpe Ratio: {metrics.get('sharpe_ratio', 0):.2f}")
                report.append(f"- Maximum Drawdown: {metrics.get('max_drawdown', 0):.2%}")
            
            if 'recommendations' in analysis_data:
                report.append("\nRecommendations:")
                for rec in analysis_data['recommendations']:
                    report.append(f"- {rec}")
            
            return "\n".join(report)
        
        except Exception as e:
            print(f"Error generating report: {e}")
            return "Error generating report. Please try again later."

class FinBot:
    def __init__(self):
        """Initialize FinBot with basic financial knowledge"""
        self.knowledge_base = {
            "greeting": ["hi", "hello", "hey", "greetings"],
            "farewell": ["bye", "goodbye", "see you", "cya"],
            "thanks": ["thank", "thanks", "appreciate"],
            "help": ["help", "assist", "support", "guide"],
            "stock": ["stock", "price", "market", "share"],
            "portfolio": ["portfolio", "investment", "holdings"],
            "analysis": ["analyze", "analysis", "study", "examine"],
            "risk": ["risk", "safety", "secure", "protect"]
        }

    def get_response(self, user_input: str) -> str:
        """
        Generate appropriate response based on user input
        
        Args:
            user_input: User's message
            
        Returns:
            str: Bot's response
        """
        user_input = user_input.lower().strip()

        # Check for greetings
        if any(word in user_input for word in self.knowledge_base["greeting"]):
            return "Hello! I'm here to help you with your financial questions. What would you like to know?"

        # Check for farewells
        if any(word in user_input for word in self.knowledge_base["farewell"]):
            return "Goodbye! Feel free to come back if you have more financial questions!"

        # Check for thanks
        if any(word in user_input for word in self.knowledge_base["thanks"]):
            return "You're welcome! Is there anything else you'd like to know?"

        # Check for help request
        if any(word in user_input for word in self.knowledge_base["help"]):
            return """I can help you with:
            - Stock market information
            - Basic financial advice
            - Portfolio analysis
            - Risk assessment
            - Market trends
            What specific area would you like to explore?"""

        # Check for stock-related queries
        if any(word in user_input for word in self.knowledge_base["stock"]):
            try:
                # Try to extract stock symbol if present
                words = user_input.split()
                for word in words:
                    if word.isupper() and len(word) <= 5:
                        stock = yf.Ticker(word)
                        info = stock.info
                        if 'regularMarketPrice' in info:
                            return f"Current price of {word}: ${info['regularMarketPrice']:.2f}\nToday's change: {info.get('regularMarketChangePercent', 0):.2%}"
                return "I can help you with stock information! Just provide a stock symbol like 'AAPL' or 'MSFT'."
            except:
                return "I can look up stock prices for you. Just provide a stock symbol like 'AAPL' or 'MSFT'."

        # Check for portfolio-related queries
        if any(word in user_input for word in self.knowledge_base["portfolio"]):
            return """Here are some portfolio management tips:
            1. Diversify your investments
            2. Regular portfolio rebalancing
            3. Monitor market trends
            4. Set clear investment goals
            Would you like to know more about any of these aspects?"""

        # Check for analysis-related queries
        if any(word in user_input for word in self.knowledge_base["analysis"]):
            return """I can help analyze:
            - Market trends
            - Stock performance
            - Portfolio diversity
            - Risk factors
            What specific analysis would you like?"""

        # Check for risk-related queries
        if any(word in user_input for word in self.knowledge_base["risk"]):
            return """Important risk management strategies:
            1. Diversification
            2. Asset allocation
            3. Regular monitoring
            4. Stop-loss orders
            Would you like to learn more about any of these?"""

        # Default response for unrecognized queries
        return """I'm not sure about that, but I can help you with:
        - Stock market information
        - Basic financial advice
        - Portfolio analysis
        - Risk assessment
        What would you like to know about these topics?"""

    def get_stock_price(self, symbol: str) -> str:
        """
        Get current stock price for a given symbol
        
        Args:
            symbol: Stock symbol (e.g., AAPL, MSFT)
            
        Returns:
            str: Stock price information
        """
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            if 'regularMarketPrice' in info:
                return f"Current price of {symbol}: ${info['regularMarketPrice']:.2f}\nToday's change: {info.get('regularMarketChangePercent', 0):.2%}"
            return f"Could not find price information for {symbol}"
        except:
            return f"Sorry, I couldn't fetch the price for {symbol}. Please check the symbol and try again."

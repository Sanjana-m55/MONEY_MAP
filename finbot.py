import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import google.generativeai as genai
from typing import Dict, Any
import os
from dotenv import load_dotenv

class FinBot:
    def __init__(self):
        """Initialize FinBot with AI capabilities"""
        # Load environment variables
        load_dotenv()
        
        # Configure Gemini API
        self.api_key = os.getenv('GEMINI_API_KEY')
        if self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp-01-21')
            
            # Set the context for the AI
            self.context = """You are Fin Bot, a friendly and knowledgeable financial assistant. 
            You specialize in providing financial advice, stock market information, and casual conversation.
            Keep your responses friendly, use emojis occasionally, and maintain a conversational tone while being informative.
            You can access real-time stock data and provide financial insights."""
        else:
            print("Warning: GEMINI_API_KEY not found in environment variables. AI features will be limited.")

    def get_stock_data(self, symbol: str) -> Dict[str, Any]:
        """Get real-time stock data"""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            return {
                'price': info.get('regularMarketPrice'),
                'change': info.get('regularMarketChangePercent'),
                'volume': info.get('regularMarketVolume'),
                'market_cap': info.get('marketCap'),
                'pe_ratio': info.get('forwardPE'),
                'dividend_yield': info.get('dividendYield')
            }
        except:
            return {}

    def get_response(self, user_input: str) -> str:
        """Generate AI-powered response based on user input"""
        try:
            if not self.api_key:
                return "Sorry, I'm currently operating with limited capabilities. Please set up the GEMINI_API_KEY to access my full potential!"

            # Check for stock symbol in user input
            words = user_input.split()
            stock_data = None
            for word in words:
                if word.isupper() and len(word) <= 5:
                    stock_data = self.get_stock_data(word)
                    break

            # Prepare the prompt with context and any stock data
            prompt = f"{self.context}\n\nUser: {user_input}"
            if stock_data:
                prompt += f"\n\nReal-time stock data: {stock_data}"

            # Generate response using Gemini
            response = self.model.generate_content(prompt)
            
            # Format and return the response
            return response.text.strip()

        except Exception as e:
            return f"I encountered an error while processing your request. Please try again! Error: {str(e)}"

    def get_stock_price(self, symbol: str) -> str:
        """Get current stock price with AI-enhanced insights"""
        try:
            stock_data = self.get_stock_data(symbol)
            if stock_data.get('price'):
                prompt = f"""Given this stock data for {symbol}: {stock_data},
                provide a brief, friendly analysis of the stock's current status.
                Include the price, daily change, and one interesting insight."""
                
                response = self.model.generate_content(prompt)
                return response.text.strip()
            return f"Hmm, couldn't find {symbol}'s price right now. Mind double-checking that symbol?"
        except:
            return f"Oops! Had trouble getting {symbol}'s price. Make sure it's a valid symbol and try again!" 
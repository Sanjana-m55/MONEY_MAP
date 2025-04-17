# Money Map - AI-Powered Financial Insights Platform

![Money Map Logo](https://img.pikbest.com/wp/202408/stock-market-trend-3d-render-of-an-upward-trending-graph-with-growth-coins-and-investing-icon_9747276.jpg!w700wp)

## 🌟 Overview

Money Map is a comprehensive financial insights platform that helps users navigate their financial journey with AI-powered analytics, interactive visualizations, and personalized recommendations. The platform allows users to upload their financial data in various formats and gain valuable insights instantly.

## 📹 YouTube Video

[![YouTube Demo](https://img.shields.io/badge/YouTube-Watch%20Demo-red?style=for-the-badge&logo=youtube)](https://www.youtube.com/shorts/hAp24Rc1tAE)

## ✨ Features

### 🔍 Data Insights
- Upload financial data in CSV, PDF, or Excel format
- Get AI-powered insights instantly
- Advanced algorithms detect patterns in your financial data
- Real-time market data integration

### 📊 Visual Analytics
- Transform your data into interactive visualizations
- Spot trends and patterns at a glance
- Compare different financial metrics easily
- Share insights with others effectively

### 💬 Financial Assistant (Fin Bot)
- Chat with your friendly financial assistant
- Get real-time financial recommendations
- Receive market updates and insights
- Ask questions about your financial data

### 🔒 Security & Privacy
- Your data remains private and secure
- Local processing for sensitive information
- No data sharing with third parties
- Optional cloud backup with encryption

## 🚀 How It Works

1. **Upload your financial data** in various formats (CSV, Excel, PDF)
2. **Explore visual representations** with interactive charts and graphs
3. **Chat with our AI assistant** to get insights and real-time financial information

## 🛠️ Tech Stack

### Frontend
- Streamlit (`streamlit==1.32.0`)
- Plotly (`plotly==5.19.0`)

### Backend
- Python 3.9+

### Data Processing
- Pandas (`pandas==2.2.1`)
- NumPy (`numpy==1.26.4`)
- scikit-learn (`scikit-learn==1.4.1.post1`)
- OpenPyXL (`openpyxl==3.1.2`)
- PDFPlumber (`pdfplumber==0.10.3`)

### AI & Machine Learning
- Google Generative AI (`google-generativeai==0.3.2`)
- Groq (`groq==0.4.2`)

### Financial Data
- Yahoo Finance (`yfinance==0.2.37`)
- Finnhub (`finnhub-python==2.4.19`)
- Alpha Vantage (`alpha_vantage==2.3.1`)

### Utilities
- Python-dotenv (`python-dotenv==1.0.1`)
- Requests (`requests==2.31.0`)
- Pydantic (`pydantic==2.6.3`, `pydantic-core==2.16.3`)
- Python-multipart (`python-multipart==0.0.9`)

## 📦 Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/money-map.git
cd money-map

# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env file with your API keys

# Run the application
streamlit run main.py 
```

## 🔧 Configuration

Create a `.env` file in the root directory with the following variables:

```
FINNHUB_API_KEY=your_finnhub_api_key
ALPHA_VANTAGE_API_KEY=your_alpha_vantage_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
MISTRAL_API_KEY=your_mistral_api_key
```

## 📝 Usage

1. Start the application using `streamlit run main.py`
2. Navigate to `http://localhost:8501` in your web browser
3. Upload your financial data through the "Data Upload & Insights" section
4. Explore visualizations in the "Visualization Studio"
5. Chat with Fin Bot in the "Financial Assistant" section for personalized advice

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 👥 Contributors

- [Sanjana M](https://github.com/Sanjana-m55)


<p align="center">Made with ❤️ for financial freedom</p>

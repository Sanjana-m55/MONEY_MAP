import streamlit as st
from dashboard import create_dashboard
from data_processor import process_data, perform_clustering
import io
import pandas as pd
from financial_agent import FinancialAgent
from finbot import FinBot

def main():
    # Set page config and title
    st.set_page_config(
        page_title="Money Map - Financial Insights Platform",
        page_icon="💰",
        layout="wide"
    )

    # Initialize FinancialAgent
    if 'financial_agent' not in st.session_state:
        st.session_state.financial_agent = FinancialAgent()

    # Custom CSS for styling
    st.markdown("""
    <style>
        .main-header {
            font-family: 'Helvetica Neue', sans-serif;
            font-size: 42px;
            font-weight: bold;
            color: #1E3A8A;
            text-align: center;
            margin-bottom: 20px;
            background: linear-gradient(90deg, #1E3A8A 0%, #3B82F6 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
        }
        .sub-header {
            font-size: 20px;
            color: #64748B;
            text-align: center;
            margin-bottom: 30px;
        }
        .feature-card {
            background-color: #F8FAFC;
            border-radius: 10px;
            padding: 20px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            margin-bottom: 20px;
        }
        .feature-title {
            font-size: 24px;
            font-weight: bold;
            color: #1E3A8A;
            margin-bottom: 10px;
        }
        .feature-desc {
            font-size: 16px;
            color: #475569;
        }
    </style>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.image("https://img.pikbest.com/wp/202408/stock-market-trend-3d-render-of-an-upward-trending-graph-with-growth-coins-and-investing-icon_9747276.jpg!w700wp", width=180)
        st.markdown("<h2 style='text-align: left;'>Money Map</h2>", unsafe_allow_html=True)
        st.markdown("<p style='text-align: left;'>Your Personal Finance Navigator</p>", unsafe_allow_html=True)
        st.divider()
        
        menu_options = ["Home", "Data Upload & Insights", "Visualization Studio", "Financial Assistant"]
        selected_menu = st.radio("Navigation", menu_options)
        
        st.divider()
        st.markdown("### About")
        st.markdown("Money Map helps you navigate your financial journey with AI-powered insights and real-time market data.")
        
        st.divider()
        st.markdown("### Contact")
        st.markdown("Need help? [Contact Support](mailto:support@moneymap.io)")
    
    # Main content area
    if selected_menu == "Home":
        show_home()
    elif selected_menu == "Data Upload & Insights":
        show_data_upload()
    elif selected_menu == "Visualization Studio":
        show_visualization()
    elif selected_menu == "Financial Assistant":
        show_financial_assistant()

def show_home():
    st.markdown("<h1 class='main-header'>Welcome to Money Map</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Navigate your financial journey with AI-powered insights</p>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>🔍 Data Insights</h3>
            <p class='feature-desc'>Upload your financial data in CSV, PDF, or Excel format and get AI-powered insights instantly.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col2:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>📊 Visual Analytics</h3>
            <p class='feature-desc'>Transform your data into interactive visualizations to spot trends and patterns.</p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class='feature-card'>
            <h3 class='feature-title'>💬 Financial Assistant</h3>
            <p class='feature-desc'>Chat with your data and get real-time financial recommendations and market updates.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("## How It Works")
    st.markdown("""
    1. **Upload your financial data** in various formats (CSV, Excel, PDF)
    2. **Explore visual representations** with interactive charts and graphs
    3. **Chat with our AI assistant** to get insights and real-time financial information
    """)
    
    st.markdown("## Why Choose Money Map?")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        - **AI-Powered Analysis**: Advanced algorithms to find patterns in your data
        - **Real-Time Market Data**: Stay updated with the latest financial information
        - **Smart Recommendations**: Get personalized financial advice
        """)
    
    with col2:
        st.markdown("""
        - **Interactive Visualizations**: Explore your data through intuitive charts
        - **Multi-Format Support**: Upload data in CSV, Excel, or PDF formats
        - **Secure Processing**: Your data remains private and secure
        """)

def show_data_upload():
    st.markdown("<h1 class='main-header'>Data Upload & Insights</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Upload your financial data and discover valuable insights</p>", unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload your financial data", type=["csv", "xlsx", "pdf"])
    
    if uploaded_file is not None:
        file_details = {"Filename": uploaded_file.name, "FileType": uploaded_file.type, "FileSize": f"{uploaded_file.size / 1024:.2f} KB"}
        st.write(file_details)
        
        try:
            # Process data based on file type
            if uploaded_file.type == "text/csv":
                df = pd.read_csv(uploaded_file)
            elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet":
                df = pd.read_excel(uploaded_file)
            elif uploaded_file.type == "application/pdf":
                df = st.session_state.financial_agent.process_financial_documents(uploaded_file)
            
            # Process the data
            insights, processed_df = process_data(df)
            
            # Show data preview
            st.subheader("Data Preview")
            st.dataframe(processed_df.head())
            
            # Show basic statistics
            st.subheader("Data Statistics")
            st.write(processed_df.describe())
            
            # Show insights
            st.subheader("AI-Generated Insights")
            for insight in insights:
                st.markdown(f"- {insight}")
            
            # Perform clustering if possible
            if len(processed_df.select_dtypes(include=['number']).columns) >= 2:
                st.subheader("Data Clustering")
                clustering_results = perform_clustering(processed_df)
                st.write("K-means clustering has identified patterns in your data. Navigate to the Visualization Studio to see these clusters.")
                st.session_state['clustered_data'] = clustering_results
                st.session_state['uploaded_data'] = processed_df
            else:
                st.warning("Not enough numerical columns for clustering. Add more numerical data for advanced analysis.")
                st.session_state['uploaded_data'] = processed_df
            
        except Exception as e:
            st.error(f"Error processing file: {e}")
    else:
        st.info("👆 Upload a file to get started with the analysis")
        
        # Sample data showcase
        st.markdown("### Sample Data Format")
        sample_data = pd.DataFrame({
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03', '2023-01-04'],
            'Income': [5000, 0, 1000, 0],
            'Expense': [1200, 500, 200, 300],
            'Category': ['Salary', 'Groceries', 'Freelance', 'Utilities']
        })
        st.dataframe(sample_data)

def show_visualization():
    st.markdown("<h1 class='main-header'>Visualization Studio</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Transform your data into insightful visualizations</p>", unsafe_allow_html=True)
    
    if 'uploaded_data' not in st.session_state:
        st.info("Please upload your data in the 'Data Upload & Insights' section first")
        st.markdown("""
        ### Why visualize your data?
        
        Data visualization helps you:
        - Identify trends and patterns at a glance
        - Compare different metrics easily
        - Make better financial decisions
        - Share insights with others effectively
        
        Upload your data to get started!
        """)
    else:
        create_dashboard(st.session_state['uploaded_data'])

def show_financial_assistant():
    st.markdown("<h1 class='main-header'>Fin Bot</h1>", unsafe_allow_html=True)
    st.markdown("<p class='sub-header'>Your friendly financial chat companion</p>", unsafe_allow_html=True)
    
    # Initialize FinBot
    if 'finbot' not in st.session_state:
        st.session_state.finbot = FinBot()
    
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = [
            {"role": "assistant", "content": "👋 Hi! I'm Fin Bot, your friendly financial assistant. How can I help you today?"}
        ]

    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # User input
    if prompt := st.chat_input("Chat with Fin Bot..."):
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Display user message
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate response
        with st.chat_message("assistant"):
            response = st.session_state.finbot.get_response(prompt)
            st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
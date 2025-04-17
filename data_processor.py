import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
import re

def process_data(df):
    """
    Process uploaded data and extract insights
    
    Args:
        df: Pandas DataFrame with uploaded data
        
    Returns:
        tuple: (insights list, processed dataframe)
    """
    # Make a copy to avoid modifying the original
    processed_df = df.copy()
    insights = []
    
    # Basic data cleaning
    # Convert potential date columns to datetime
    for col in processed_df.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            try:
                processed_df[col] = pd.to_datetime(processed_df[col])
                insights.append(f"Converted '{col}' to datetime format for better analysis.")
            except:
                pass
    
    # Handle missing values
    missing_counts = processed_df.isnull().sum()
    if missing_counts.sum() > 0:
        missing_cols = missing_counts[missing_counts > 0].index.tolist()
        
        # For numerical columns, fill with median
        num_cols = processed_df.select_dtypes(include=['number']).columns
        for col in num_cols:
            if col in missing_cols:
                median_val = processed_df[col].median()
                processed_df[col].fillna(median_val, inplace=True)
                insights.append(f"Filled missing values in '{col}' with median value ({median_val:.2f}).")
        
        # For categorical columns, fill with mode
        cat_cols = processed_df.select_dtypes(exclude=['number']).columns
        for col in cat_cols:
            if col in missing_cols and col not in processed_df.select_dtypes(include=['datetime64']).columns:
                mode_val = processed_df[col].mode()[0] if not processed_df[col].mode().empty else "Unknown"
                processed_df[col].fillna(mode_val, inplace=True)
                insights.append(f"Filled missing values in '{col}' with most common value ('{mode_val}').")
    
    # Generate basic insights
    # Count of records
    insights.append(f"Your dataset contains {len(processed_df)} records with {len(processed_df.columns)} variables.")
    
    # Numerical columns analysis
    num_cols = processed_df.select_dtypes(include=['number']).columns
    if len(num_cols) > 0:
        for col in num_cols:
            mean_val = processed_df[col].mean()
            max_val = processed_df[col].max()
            min_val = processed_df[col].min()
            
            # Check if column might be financial
            if any(term in col.lower() for term in ['price', 'cost', 'amount', 'income', 'expense', 'salary', 'revenue', 'profit', 'loss', 'budget']):
                insights.append(f"The average {col} is ${mean_val:.2f}, ranging from ${min_val:.2f} to ${max_val:.2f}.")
            else:
                insights.append(f"The average {col} is {mean_val:.2f}, ranging from {min_val:.2f} to {max_val:.2f}.")
            
            # Check for outliers using IQR method
            Q1 = processed_df[col].quantile(0.25)
            Q3 = processed_df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = processed_df[(processed_df[col] < lower_bound) | (processed_df[col] > upper_bound)]
            if len(outliers) > 0:
                insights.append(f"Detected {len(outliers)} potential outliers in '{col}' that may require attention.")
    
    # Categorical columns analysis
    cat_cols = processed_df.select_dtypes(exclude=['number', 'datetime64']).columns
    if len(cat_cols) > 0:
        for col in cat_cols:
            value_counts = processed_df[col].value_counts()
            if len(value_counts) <= 10:  # Only for columns with reasonable number of categories
                most_common = value_counts.index[0]
                most_common_pct = (value_counts.iloc[0] / len(processed_df)) * 100
                insights.append(f"The most common {col} is '{most_common}' ({most_common_pct:.1f}% of records).")
    
    # Time-based analysis for datetime columns
    date_cols = processed_df.select_dtypes(include=['datetime64']).columns
    if len(date_cols) > 0:
        for col in date_cols:
            date_range = (processed_df[col].max() - processed_df[col].min()).days
            insights.append(f"Your data spans {date_range} days from {processed_df[col].min().date()} to {processed_df[col].max().date()}.")
            
            # Check if we have time series financial data
            if len(num_cols) > 0 and len(processed_df) > 5:
                for num_col in num_cols:
                    if any(term in num_col.lower() for term in ['price', 'cost', 'amount', 'income', 'expense', 'salary', 'revenue', 'profit', 'loss', 'budget']):
                        # Check for trend
                        try:
                            first_vals = processed_df.sort_values(by=col).head(5)[num_col].mean()
                            last_vals = processed_df.sort_values(by=col).tail(5)[num_col].mean()
                            pct_change = ((last_vals - first_vals) / first_vals) * 100 if first_vals != 0 else 0
                            
                            if pct_change > 10:
                                insights.append(f"There's an upward trend of {pct_change:.1f}% in {num_col} over the time period.")
                            elif pct_change < -10:
                                insights.append(f"There's a downward trend of {abs(pct_change):.1f}% in {num_col} over the time period.")
                        except:
                            pass
    
    return insights, processed_df

def perform_clustering(df, n_clusters=3):
    """
    Perform K-means clustering on numerical data
    
    Args:
        df: Pandas DataFrame with processed data
        n_clusters: Number of clusters for K-means
        
    Returns:
        dict: Clustering results including labeled data and cluster info
    """
    # Select only numerical columns for clustering
    num_cols = df.select_dtypes(include=['number']).columns
    
    if len(num_cols) < 2:
        return {"error": "Not enough numerical columns for clustering"}
    
    # Select numerical data
    X = df[num_cols].copy()
    
    # Handle missing values if any
    imputer = SimpleImputer(strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    # Normalize data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_imputed)
    
    # Apply K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(X_scaled)
    
    # Add cluster labels to the original data
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    # Calculate cluster statistics
    cluster_stats = {}
    for i in range(n_clusters):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == i]
        stats = {}
        
        for col in num_cols:
            stats[col] = {
                'mean': cluster_data[col].mean(),
                'median': cluster_data[col].median(),
                'std': cluster_data[col].std(),
                'min': cluster_data[col].min(),
                'max': cluster_data[col].max()
            }
        
        cluster_stats[i] = {
            'size': len(cluster_data),
            'percentage': (len(cluster_data) / len(df)) * 100,
            'stats': stats
        }
    
    # Perform PCA for visualization
    if len(num_cols) > 2:
        pca = PCA(n_components=2)
        pca_result = pca.fit_transform(X_scaled)
        
        # Add PCA components to dataframe
        df_with_clusters['pca_x'] = pca_result[:, 0]
        df_with_clusters['pca_y'] = pca_result[:, 1]
    else:
        # If only 2 columns, use them directly
        cols = list(num_cols)
        df_with_clusters['pca_x'] = X_scaled[:, 0]
        df_with_clusters['pca_y'] = X_scaled[:, 1]
    
    # Determine cluster characteristics
    cluster_characteristics = {}
    for i in range(n_clusters):
        characteristics = []
        cluster_df = df_with_clusters[df_with_clusters['cluster'] == i]
        
        for col in num_cols:
            # Compare cluster mean to overall mean
            cluster_mean = cluster_df[col].mean()
            overall_mean = df[col].mean()
            
            if cluster_mean > overall_mean * 1.25:
                characteristics.append(f"High {col}")
            elif cluster_mean < overall_mean * 0.75:
                characteristics.append(f"Low {col}")
            
        cluster_characteristics[i] = characteristics
    
    # Return clustering results
    return {
        'data': df_with_clusters,
        'cluster_stats': cluster_stats,
        'cluster_characteristics': cluster_characteristics,
        'feature_names': list(num_cols),
        'centroids': kmeans.cluster_centers_,
        'pca_explained_variance': pca.explained_variance_ratio_ if len(num_cols) > 2 else None
    }

def detect_financial_data_types(df):
    """
    Detect and categorize columns in financial data
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        dict: Categorized columns by financial type
    """
    financial_categories = {
        'date': [],
        'amount': [],
        'price': [],
        'category': [],
        'ticker': [],
        'name': [],
        'other': []
    }
    
    # Check each column
    for col in df.columns:
        col_lower = col.lower()
        sample_values = df[col].dropna().astype(str).tolist()[:5]
        
        # Date detection
        if ('date' in col_lower or 'time' in col_lower or 'day' in col_lower or 'month' in col_lower or 'year' in col_lower):
            financial_categories['date'].append(col)
        
        # Amount detection
        elif any(term in col_lower for term in ['amount', 'sum', 'total', 'price', 'cost', 'fee', 'income', 'expense', 'balance', 'budget']):
            if df[col].dtype in ['int64', 'float64'] or (df[col].dtype == 'object' and all(re.match(r'^[\$£€]?\s*-?\d+\.?\d*$', str(v)) for v in sample_values if str(v).strip())):
                financial_categories['amount'].append(col)
        
        # Price detection
        elif any(term in col_lower for term in ['price', 'rate', 'value']):
            if df[col].dtype in ['int64', 'float64'] or (df[col].dtype == 'object' and all(re.match(r'^[\$£€]?\s*-?\d+\.?\d*$', str(v)) for v in sample_values if str(v).strip())):
                financial_categories['price'].append(col)
        
        # Category detection
        elif any(term in col_lower for term in ['category', 'type', 'group', 'class']):
            financial_categories['category'].append(col)
        
        # Ticker/Symbol detection
        elif any(term in col_lower for term in ['ticker', 'symbol', 'stock']):
            financial_categories['ticker'].append(col)
        
        # Name detection
        elif any(term in col_lower for term in ['name', 'description', 'item', 'product']):
            financial_categories['name'].append(col)
        
        # Default to other
        else:
            financial_categories['other'].append(col)
    
    return financial_categories
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

def create_dashboard(df):
    """
    Create interactive dashboard for the uploaded data
    
    Args:
        df: Pandas DataFrame with processed data
    """
    if df is None or df.empty:
        st.warning("No data available for visualization. Please upload data first.")
        return
    
    # Display data summary
    st.subheader("Data Overview")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", f"{len(df):,}")
    with col2:
        st.metric("Total Features", f"{len(df.columns):,}")
    
    # Get column types for visualization options
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    cat_cols = df.select_dtypes(exclude=['number']).columns.tolist()
    date_cols = [col for col in df.columns if df[col].dtype == 'datetime64[ns]' or 'date' in col.lower()]

    # Create tabs for different visualization categories
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Trend Analysis", 
        "Distribution Analysis", 
        "Correlation Analysis", 
        "Categorical Analysis",
        "Cluster Analysis"
    ])
    
    # Tab 1: Trend Analysis
    with tab1:
        st.markdown("### Trend Analysis")
        st.markdown("Analyze how your data changes over time")
        
        if len(date_cols) > 0 and len(num_cols) > 0:
            # Select date column and numeric column for trend analysis
            trend_date_col = st.selectbox("Select Date Column", date_cols, key="trend_date")
            trend_measure_cols = st.multiselect("Select Measures to Plot", num_cols, default=[num_cols[0]] if num_cols else [], key="trend_measure")
            
            if trend_date_col and trend_measure_cols:
                # Convert to datetime if not already
                df_trend = df.copy()
                if df_trend[trend_date_col].dtype != 'datetime64[ns]':
                    try:
                        df_trend[trend_date_col] = pd.to_datetime(df_trend[trend_date_col])
                    except:
                        st.error(f"Could not convert {trend_date_col} to datetime. Please select another column.")
                        return
                
                # Create line chart
                fig = go.Figure()
                
                for col in trend_measure_cols:
                    # Calculate moving average if enough data points
                    if len(df_trend) > 5:
                        df_trend[f'{col}_MA5'] = df_trend[col].rolling(window=5).mean()
                        # Add both raw data and moving average
                        fig.add_trace(go.Scatter(
                            x=df_trend[trend_date_col],
                            y=df_trend[col],
                            mode='lines+markers',
                            name=col,
                            line=dict(width=1)
                        ))
                        fig.add_trace(go.Scatter(
                            x=df_trend[trend_date_col],
                            y=df_trend[f'{col}_MA5'],
                            mode='lines',
                            name=f'{col} (5-point MA)',
                            line=dict(width=3)
                        ))
                    else:
                        # Just add raw data if not enough for moving average
                        fig.add_trace(go.Scatter(
                            x=df_trend[trend_date_col],
                            y=df_trend[col],
                            mode='lines+markers',
                            name=col
                        ))
                
                fig.update_layout(
                    title=f"Trend Analysis over Time",
                    xaxis_title=trend_date_col,
                    yaxis_title="Value",
                    legend_title="Measures",
                    height=500,
                    hovermode="x unified"
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Add trend statistics
                if len(df_trend) > 1:
                    st.subheader("Trend Statistics")
                    stat_cols = st.columns(len(trend_measure_cols))
                    
                    for i, col in enumerate(trend_measure_cols):
                        first_val = df_trend.sort_values(trend_date_col).iloc[0][col]
                        last_val = df_trend.sort_values(trend_date_col).iloc[-1][col]
                        change = last_val - first_val
                        pct_change = (change / first_val) * 100 if first_val != 0 else 0
                        
                        with stat_cols[i]:
                            st.metric(
                                label=f"{col} Change",
                                value=f"{change:,.2f}",
                                delta=f"{pct_change:.1f}%"
                            )
            else:
                st.info("Please select date and measure columns to analyze trends")
        else:
            st.warning("Trend analysis requires at least one date column and one numeric column. Your data doesn't seem to have the required columns.")
    
    # Tab 2: Distribution Analysis
    with tab2:
        st.markdown("### Distribution Analysis")
        st.markdown("Examine the distribution of your numerical data")
        
        if len(num_cols) > 0:
            dist_col = st.selectbox("Select Numeric Column", num_cols, key="dist_col")
            dist_type = st.radio("Chart Type", ["Histogram", "Box Plot", "Violin Plot"], horizontal=True)
            
            if dist_col:
                if dist_type == "Histogram":
                    fig = px.histogram(
                        df, 
                        x=dist_col,
                        nbins=20,
                        marginal="box",
                        title=f"Distribution of {dist_col}",
                        color_discrete_sequence=['#3B82F6']
                    )
                elif dist_type == "Box Plot":
                    fig = px.box(
                        df,
                        y=dist_col,
                        title=f"Box Plot of {dist_col}",
                        points="all",
                        color_discrete_sequence=['#3B82F6']
                    )
                else:  # Violin Plot
                    fig = px.violin(
                        df,
                        y=dist_col,
                        box=True,
                        points="all",
                        title=f"Violin Plot of {dist_col}",
                        color_discrete_sequence=['#3B82F6']
                    )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display distribution statistics
                st.subheader("Distribution Statistics")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Mean", f"{df[dist_col].mean():.2f}")
                with col2:
                    st.metric("Median", f"{df[dist_col].median():.2f}")
                with col3:
                    st.metric("Std Dev", f"{df[dist_col].std():.2f}")
                with col4:
                    st.metric("IQR", f"{df[dist_col].quantile(0.75) - df[dist_col].quantile(0.25):.2f}")

                # Add percentile information
                st.markdown("#### Percentiles")
                pct_cols = st.columns(5)
                percentiles = [10, 25, 50, 75, 90]
                for i, p in enumerate(percentiles):
                    with pct_cols[i]:
                        st.metric(f"{p}th", f"{df[dist_col].quantile(p/100):.2f}")
                
                # Distribution by category if categorical columns exist
                if len(cat_cols) > 0:
                    st.markdown("#### Distribution by Category")
                    cat_col = st.selectbox("Select Category", cat_cols, key="dist_cat_col")
                    if cat_col:
                        fig = px.box(
                            df,
                            x=cat_col,
                            y=dist_col,
                            title=f"Distribution of {dist_col} by {cat_col}",
                            color=cat_col
                        )
                        fig.update_layout(height=500)
                        st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("No categorical columns available for grouped distribution analysis")
        else:
            st.warning("Distribution analysis requires at least one numerical column. Your data doesn't seem to have numerical columns.")
    
    # Tab 3: Correlation Analysis
    with tab3:
        st.markdown("### Correlation Analysis")
        st.markdown("Explore relationships between numerical variables")
        
        if len(num_cols) >= 2:
            corr_type = st.radio("Chart Type", ["Scatter Plot", "Correlation Matrix", "Pair Plot"], horizontal=True)
            
            if corr_type == "Scatter Plot":
                col1, col2 = st.columns(2)
                with col1:
                    x_col = st.selectbox("X-Axis", num_cols, key="scatter_x")
                with col2:
                    y_col = st.selectbox("Y-Axis", num_cols, index=1 if len(num_cols) > 1 else 0, key="scatter_y")
                
                color_col = st.selectbox("Color By (Optional)", ["None"] + cat_cols, key="scatter_color")
                color = None if color_col == "None" else color_col
                
                fig = px.scatter(
                    df,
                    x=x_col,
                    y=y_col,
                    color=color,
                    title=f"Relationship between {x_col} and {y_col}",
                    trendline="ols" if color is None else None,
                    opacity=0.7
                )
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Calculate correlation
                correlation = df[x_col].corr(df[y_col])
                st.metric("Pearson Correlation", f"{correlation:.3f}")
                
                # Interpret correlation
                if abs(correlation) < 0.3:
                    st.info("This indicates a weak correlation between the variables.")
                elif abs(correlation) < 0.7:
                    st.info("This indicates a moderate correlation between the variables.")
                else:
                    st.info("This indicates a strong correlation between the variables.")
                
            elif corr_type == "Correlation Matrix":
                if len(num_cols) > 10:
                    selected_cols = st.multiselect("Select Columns (max 10 recommended)", num_cols, default=num_cols[:5])
                else:
                    selected_cols = num_cols
                
                if selected_cols:
                    corr_matrix = df[selected_cols].corr()
                    
                    fig = px.imshow(
                        corr_matrix,
                        text_auto=True,
                        aspect="auto",
                        color_continuous_scale="RdBu_r",
                        title="Correlation Matrix"
                    )
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Highlight strongest correlations
                    if len(selected_cols) > 1:
                        st.markdown("#### Strongest Correlations")
                        # Get the upper triangle of the correlation matrix
                        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
                        corr_df = corr_matrix.mask(mask).stack().reset_index()
                        corr_df.columns = ['Variable 1', 'Variable 2', 'Correlation']
                        corr_df = corr_df.sort_values(by='Correlation', key=abs, ascending=False)
                        
                        st.dataframe(corr_df.head(5), hide_index=True)
            else:  # Pair Plot
                if len(num_cols) > 6:
                    selected_cols = st.multiselect("Select Columns (max 6 recommended)", num_cols, default=num_cols[:3])
                else:
                    selected_cols = st.multiselect("Select Columns", num_cols, default=num_cols[:min(3, len(num_cols))])
                
                if len(selected_cols) >= 2:
                    color_col = st.selectbox("Color By (Optional)", ["None"] + cat_cols, key="pairplot_color")
                    color = None if color_col == "None" else color_col
                    
                    fig = px.scatter_matrix(
                        df,
                        dimensions=selected_cols,
                        color=color,
                        title="Pair Plot",
                        opacity=0.5
                    )
                    fig.update_layout(height=700)
                    st.plotly_chart(fig, use_container_width=True)
                else:
                    st.info("Please select at least 2 columns for the pair plot")
        else:
            st.warning("Correlation analysis requires at least two numerical columns. Your data doesn't seem to have enough numerical columns.")
    
    # Tab 4: Categorical Analysis
    with tab4:
        st.markdown("### Categorical Analysis")
        st.markdown("Analyze categorical variables and their relationships")
        
        if len(cat_cols) > 0:
            chart_type = st.radio(
                "Chart Type", 
                ["Bar Chart", "Pie Chart", "Grouped Bar Chart", "Stacked Bar Chart", "Treemap"],
                horizontal=True
            )
            
            col1, col2 = st.columns(2)
            with col1:
                main_cat = st.selectbox("Main Category", cat_cols, key="cat_main")
            
            if chart_type in ["Grouped Bar Chart", "Stacked Bar Chart", "Treemap"]:
                with col2:
                    second_cat = st.selectbox("Secondary Category", [c for c in cat_cols if c != main_cat], key="cat_second")
            
            # Optional value column for aggregations
            agg_col = None
            if len(num_cols) > 0:
                agg_col = st.selectbox("Aggregate Value (Optional)", ["Count"] + num_cols, key="cat_agg")
                agg_func = "count" if agg_col == "Count" else "sum"
            else:
                agg_func = "count"
            
            if main_cat:
                if chart_type == "Bar Chart":
                    if agg_col == "Count" or agg_col is None:
                        chart_data = df[main_cat].value_counts().reset_index()
                        chart_data.columns = [main_cat, 'Count']
                        fig = px.bar(
                            chart_data,
                            x=main_cat,
                            y='Count',
                            title=f"Counts of {main_cat}",
                            color=main_cat
                        )
                    else:
                        chart_data = df.groupby(main_cat)[agg_col].sum().reset_index()
                        fig = px.bar(
                            chart_data,
                            x=main_cat,
                            y=agg_col,
                            title=f"Sum of {agg_col} by {main_cat}",
                            color=main_cat
                        )
                
                elif chart_type == "Pie Chart":
                    if agg_col == "Count" or agg_col is None:
                        chart_data = df[main_cat].value_counts().reset_index()
                        chart_data.columns = [main_cat, 'Count']
                        fig = px.pie(
                            chart_data,
                            names=main_cat,
                            values='Count',
                            title=f"Distribution of {main_cat}"
                        )
                    else:
                        chart_data = df.groupby(main_cat)[agg_col].sum().reset_index()
                        fig = px.pie(
                            chart_data,
                            names=main_cat,
                            values=agg_col,
                            title=f"Distribution of {agg_col} by {main_cat}"
                        )
                
                elif chart_type == "Grouped Bar Chart":
                    if 'second_cat' in locals():
                        if agg_col == "Count" or agg_col is None:
                            chart_data = df.groupby([main_cat, second_cat]).size().reset_index(name='Count')
                            fig = px.bar(
                                chart_data,
                                x=main_cat,
                                y='Count',
                                color=second_cat,
                                barmode='group',
                                title=f"Counts of {main_cat} by {second_cat}"
                            )
                        else:
                            chart_data = df.groupby([main_cat, second_cat])[agg_col].sum().reset_index()
                            fig = px.bar(
                                chart_data,
                                x=main_cat,
                                y=agg_col,
                                color=second_cat,
                                barmode='group',
                                title=f"Sum of {agg_col} by {main_cat} and {second_cat}"
                            )
                
                elif chart_type == "Stacked Bar Chart":
                    if 'second_cat' in locals():
                        if agg_col == "Count" or agg_col is None:
                            chart_data = df.groupby([main_cat, second_cat]).size().reset_index(name='Count')
                            fig = px.bar(
                                chart_data,
                                x=main_cat,
                                y='Count',
                                color=second_cat,
                                barmode='stack',
                                title=f"Counts of {main_cat} by {second_cat}"
                            )
                        else:
                            chart_data = df.groupby([main_cat, second_cat])[agg_col].sum().reset_index()
                            fig = px.bar(
                                chart_data,
                                x=main_cat,
                                y=agg_col,
                                color=second_cat,
                                barmode='stack',
                                title=f"Sum of {agg_col} by {main_cat} and {second_cat}"
                            )
                
                elif chart_type == "Treemap":
                    if 'second_cat' in locals():
                        if agg_col == "Count" or agg_col is None:
                            df_temp = df.copy()
                            df_temp['Count'] = 1
                            fig = px.treemap(
                                df_temp,
                                path=[main_cat, second_cat],
                                values='Count',
                                title=f"Hierarchical View of {main_cat} and {second_cat}"
                            )
                        else:
                            fig = px.treemap(
                                df,
                                path=[main_cat, second_cat],
                                values=agg_col,
                                title=f"Hierarchical View of {agg_col} by {main_cat} and {second_cat}"
                            )
                
                fig.update_layout(height=500)
                st.plotly_chart(fig, use_container_width=True)
                
                # Show top categories table
                st.subheader("Top Categories")
                if agg_col == "Count" or agg_col is None:
                    top_cats = df[main_cat].value_counts().reset_index()
                    top_cats.columns = [main_cat, 'Count']
                    top_cats['Percentage'] = (top_cats['Count'] / top_cats['Count'].sum() * 100).round(2)
                    top_cats['Percentage'] = top_cats['Percentage'].apply(lambda x: f"{x:.2f}%")
                else:
                    top_cats = df.groupby(main_cat)[agg_col].sum().reset_index()
                    top_cats = top_cats.sort_values(by=agg_col, ascending=False)
                    top_cats['Percentage'] = (top_cats[agg_col] / top_cats[agg_col].sum() * 100).round(2)
                    top_cats['Percentage'] = top_cats['Percentage'].apply(lambda x: f"{x:.2f}%")
                
                st.dataframe(top_cats.head(10), hide_index=True)
        else:
            st.warning("Categorical analysis requires at least one categorical column. Your data doesn't seem to have categorical columns.")
    
    # Tab 5: Cluster Analysis
    with tab5:
        st.markdown("### Cluster Analysis")
        st.markdown("Explore patterns and groups in your data using K-means clustering")
        
        if 'clustered_data' in st.session_state:
            clustering = st.session_state['clustered_data']
            if 'error' in clustering:
                st.warning(clustering['error'])
            else:
                cluster_data = clustering['data']
                
                # Plot clusters
                st.subheader("Cluster Visualization")
                
                # Create a scatter plot of the clusters
                fig = px.scatter(
                    cluster_data,
                    x='pca_x' if 'pca_x' in cluster_data.columns else clustering['feature_names'][0],
                    y='pca_y' if 'pca_y' in cluster_data.columns else clustering['feature_names'][1],
                    color='cluster',
                    title="Cluster Visualization" + (" (PCA Projection)" if 'pca_x' in cluster_data.columns else ""),
                    labels={
                        'pca_x': 'PCA Component 1',
                        'pca_y': 'PCA Component 2'
                    } if 'pca_x' in cluster_data.columns else {},
                    color_continuous_scale=px.colors.qualitative.G10
                )
                
                # Add centroids to the plot
                if 'centroids' in clustering:
                    centroids = clustering['centroids']
                    if 'pca_x' in cluster_data.columns and 'pca_explained_variance' in clustering and clustering['pca_explained_variance'] is not None:
                        # Need to project centroids to PCA space
                        # Simplified approach: just show the existing points
                        pass
                    else:
                        # Plot centroids directly if no PCA was applied
                        fig.add_trace(go.Scatter(
                            x=centroids[:, 0],
                            y=centroids[:, 1],
                            mode='markers',
                            marker=dict(
                                color='black',
                                size=15,
                                symbol='x'
                            ),
                            name='Centroids'
                        ))
                
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)
                
                # Display cluster characteristics
                st.subheader("Cluster Characteristics")
                
                cluster_char = clustering['cluster_characteristics']
                cluster_stats = clustering['cluster_stats']
                
                for i in range(len(cluster_char)):
                    with st.expander(f"Cluster {i} ({cluster_stats[i]['percentage']:.1f}% of data)"):
                        # Show characteristics
                        st.markdown(f"**Key Characteristics:**")
                        if cluster_char[i]:
                            for char in cluster_char[i]:
                                st.markdown(f"- {char}")
                        else:
                            st.markdown("- No distinctive characteristics identified")
                        
                        # Show stats for this cluster
                        st.markdown(f"**Statistics:**")
                        stats_df = pd.DataFrame()
                        
                        for feature in clustering['feature_names']:
                            stats_df[feature] = [
                                f"Mean: {cluster_stats[i]['stats'][feature]['mean']:.2f}",
                                f"Median: {cluster_stats[i]['stats'][feature]['median']:.2f}",
                                f"Std Dev: {cluster_stats[i]['stats'][feature]['std']:.2f}",
                                f"Range: {cluster_stats[i]['stats'][feature]['min']:.2f} - {cluster_stats[i]['stats'][feature]['max']:.2f}"
                            ]
                        
                        stats_df.index = ['Mean', 'Median', 'Std Dev', 'Range']
                        st.dataframe(stats_df.T)
        else:
            st.info("Cluster analysis will be available after you upload and process data with numerical columns.")
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import json
import io
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Set page config
st.set_page_config(page_title="Video Game Sales Dashboard", page_icon="🎮", layout="wide")

# Initialize session state for favorites
if 'favorites' not in st.session_state:
    st.session_state.favorites = []

# Title
st.title("🎮 Video Game Sales Dashboard")

# Load data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("vgsales.csv")
        return df
    except FileNotFoundError:
        st.error("vgsales.csv not found. Please ensure the file is in the same directory as this app.")
        return pd.DataFrame()

df = load_data()

if df.empty:
    st.stop()

# Sidebar
st.sidebar.header("Filters")

# Year range
min_year = int(df['Year'].min())
max_year = int(df['Year'].max())
year_range = st.sidebar.slider("Select Year Range", min_year, max_year, (min_year, max_year))

# Genre filter
genres = df['Genre'].unique()
selected_genres = st.sidebar.multiselect("Select Genres", genres, default=genres)

# Publisher filter
publishers = df['Publisher'].unique()
selected_publishers = st.sidebar.multiselect("Select Publishers", publishers, default=publishers[:10])  # Default to top 10

# Search by game name
search_term = st.sidebar.text_input("Search by Game Name", "")

# Filter data
filtered_df = df[
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1]) &
    (df['Genre'].isin(selected_genres)) &
    (df['Publisher'].isin(selected_publishers))
]

# Apply search filter
if search_term:
    filtered_df = filtered_df[filtered_df['Name'].str.contains(search_term, case=False, na=False)]

# Tabs
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["Overview", "Charts", "Data", "Clustering", "Analytics", "Comparison", "Favorites"])

with tab1:
    st.header("Overview")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Games", len(filtered_df))
    with col2:
        st.metric("Total Global Sales (M)", f"{filtered_df['Global_Sales'].sum():.2f}")
    with col3:
        st.metric("Average Global Sales (M)", f"{filtered_df['Global_Sales'].mean():.2f}")
    
    st.subheader("Top 10 Games by Global Sales")
    top_games = filtered_df.nlargest(10, 'Global_Sales')[['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']]
    st.table(top_games)

with tab2:
    st.header("Charts")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Global Sales by Genre")
        genre_sales = filtered_df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False)
        fig, ax = plt.subplots()
        genre_sales.plot(kind='bar', ax=ax, color='skyblue')
        plt.xticks(rotation=45, ha='right')
        plt.ylabel('Global Sales (M)')
        st.pyplot(fig)
    
    with col2:
        st.subheader("Sales Distribution")
        fig, ax = plt.subplots()
        sns.histplot(filtered_df['Global_Sales'], bins=50, ax=ax, color='green')
        plt.xlabel('Global Sales (M)')
        st.pyplot(fig)
    
    st.subheader("Global Sales Over Time")
    yearly_sales = filtered_df.groupby('Year')['Global_Sales'].sum()
    fig, ax = plt.subplots()
    yearly_sales.plot(ax=ax, color='red')
    plt.ylabel('Global Sales (M)')
    st.pyplot(fig)
    
    col3, col4 = st.columns(2)
    
    with col3:
        st.subheader("Top Platforms by Sales")
        platform_sales = filtered_df.groupby('Platform')['Global_Sales'].sum().nlargest(10)
        fig, ax = plt.subplots()
        platform_sales.plot(kind='barh', ax=ax, color='orange')
        plt.xlabel('Global Sales (M)')
        st.pyplot(fig)
    
    with col4:
        st.subheader("Regional Sales Comparison")
        regions = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales']
        regional_sales = filtered_df[regions].sum()
        fig, ax = plt.subplots()
        regional_sales.plot(kind='pie', autopct='%1.1f%%', ax=ax, startangle=90)
        ax.set_ylabel('')
        st.pyplot(fig)

with tab3:
    st.header("Data")
    
    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df)
    
    # Export
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="Download Filtered Data as CSV",
        data=csv,
        file_name='filtered_vgsales.csv',
        mime='text/csv',
        key='download-csv'
    )

with tab4:
    st.header("Clustering Analysis")
    
    st.write("Apply K-Means clustering to group games based on selected features.")
    
    # Select features for clustering
    numeric_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
    if 'Year' in filtered_df.columns:
        numeric_cols.append('Year')
    
    selected_features = st.multiselect(
        "Select features for clustering:",
        numeric_cols,
        default=['Global_Sales', 'NA_Sales', 'EU_Sales']
    )
    
    if len(selected_features) < 2:
        st.warning("Please select at least 2 features for clustering.")
    else:
        # Number of clusters
        n_clusters = st.slider("Number of clusters:", 2, 10, 3)
        
        # Prepare data
        cluster_data = filtered_df[selected_features].dropna()
        
        if len(cluster_data) == 0:
            st.error("No data available for clustering with selected features.")
        else:
            # Scale data
            scaler = StandardScaler()
            scaled_data = scaler.fit_transform(cluster_data)
            
            # Apply K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_data)
            
            # Add cluster labels to data
            cluster_data['Cluster'] = clusters
            
            # Display cluster centers
            st.subheader("Cluster Centers")
            centers = pd.DataFrame(
                scaler.inverse_transform(kmeans.cluster_centers_),
                columns=selected_features
            )
            st.dataframe(centers)
            
            # Visualize clusters (if 2D or use PCA)
            if len(selected_features) == 2:
                st.subheader("Cluster Visualization")
                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    cluster_data[selected_features[0]], 
                    cluster_data[selected_features[1]], 
                    c=cluster_data['Cluster'], 
                    cmap='viridis'
                )
                ax.set_xlabel(selected_features[0])
                ax.set_ylabel(selected_features[1])
                plt.colorbar(scatter)
                st.pyplot(fig)
            else:
                # Use PCA for visualization
                pca = PCA(n_components=2)
                pca_data = pca.fit_transform(scaled_data)
                pca_df = pd.DataFrame(pca_data, columns=['PC1', 'PC2'])
                pca_df['Cluster'] = clusters
                
                st.subheader("Cluster Visualization (PCA)")
                fig, ax = plt.subplots()
                scatter = ax.scatter(
                    pca_df['PC1'], 
                    pca_df['PC2'], 
                    c=pca_df['Cluster'], 
                    cmap='viridis'
                )
                ax.set_xlabel('Principal Component 1')
                ax.set_ylabel('Principal Component 2')
                plt.colorbar(scatter)
                st.pyplot(fig)
            
            # Cluster summary
            st.subheader("Cluster Summary")
            summary = cluster_data.groupby('Cluster').agg({
                **{col: 'mean' for col in selected_features},
                'Cluster': 'count'
            }).rename(columns={'Cluster': 'Count'})
            st.dataframe(summary)

with tab5:
    st.header("Advanced Analytics")

    st.subheader("Correlation Analysis")
    numeric_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Year']
    corr_data = filtered_df[numeric_cols].corr()

    fig = px.imshow(corr_data,
                    text_auto=True,
                    aspect="auto",
                    color_continuous_scale='RdBu_r',
                    title="Correlation Matrix")
    st.plotly_chart(fig)

    st.subheader("Sales Trends & Forecasting")

    # Prepare time series data
    yearly_data = filtered_df.groupby('Year')['Global_Sales'].sum().reset_index()

    # Simple linear regression for forecasting
    if len(yearly_data) > 5:
        X = yearly_data['Year'].values.reshape(-1, 1)
        y = yearly_data['Global_Sales'].values

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Predict next 5 years
        future_years = np.array([yearly_data['Year'].max() + i for i in range(1, 6)]).reshape(-1, 1)
        future_predictions = model.predict(future_years)

        # Create forecast dataframe
        forecast_df = pd.DataFrame({
            'Year': future_years.flatten(),
            'Predicted_Global_Sales': future_predictions,
            'Type': 'Forecast'
        })

        # Combine historical and forecast data
        historical_df = yearly_data.copy()
        historical_df['Type'] = 'Historical'

        combined_df = pd.concat([historical_df, forecast_df.rename(columns={'Predicted_Global_Sales': 'Global_Sales'})])

        fig = px.line(combined_df, x='Year', y='Global_Sales', color='Type',
                     title='Global Sales Trends with 5-Year Forecast',
                     markers=True)
        st.plotly_chart(fig)

        # Model performance
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Model R² Score", f"{r2:.3f}")
        with col2:
            st.metric("Mean Squared Error", f"{mse:.3f}")

    st.subheader("Genre Performance Analysis")
    genre_yearly = filtered_df.pivot_table(values='Global_Sales', index='Year', columns='Genre', aggfunc='sum')

    # Select top genres for cleaner visualization
    top_genres = genre_yearly.sum().nlargest(5).index
    genre_yearly_top = genre_yearly[top_genres]

    fig = px.area(genre_yearly_top, x=genre_yearly_top.index, y=genre_yearly_top.columns,
                  title='Top 5 Genres Sales Trends Over Time')
    st.plotly_chart(fig)

with tab6:
    st.header("Game & Platform Comparison")

    st.subheader("Compare Multiple Games")

    # Game selection
    game_names = filtered_df['Name'].unique()
    selected_games = st.multiselect("Select games to compare (max 5)", game_names,
                                   max_selections=5, key='game_comparison')

    if selected_games:
        comparison_data = filtered_df[filtered_df['Name'].isin(selected_games)]

        # Sales comparison chart
        sales_cols = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales']
        comparison_melted = comparison_data.melt(id_vars=['Name'],
                                                value_vars=sales_cols,
                                                var_name='Region',
                                                value_name='Sales')

        fig = px.bar(comparison_melted, x='Name', y='Sales', color='Region',
                    title='Regional Sales Comparison',
                    barmode='stack')
        st.plotly_chart(fig)

        # Detailed comparison table
        st.subheader("Detailed Comparison")
        comparison_table = comparison_data[['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']]
        st.dataframe(comparison_table)

    st.subheader("Platform Comparison")

    # Platform selection
    platforms = filtered_df['Platform'].unique()
    selected_platforms = st.multiselect("Select platforms to compare", platforms,
                                       default=platforms[:3], key='platform_comparison')

    if selected_platforms:
        platform_data = filtered_df[filtered_df['Platform'].isin(selected_platforms)]

        # Platform performance over time
        platform_yearly = platform_data.groupby(['Year', 'Platform'])['Global_Sales'].sum().reset_index()

        fig = px.line(platform_yearly, x='Year', y='Global_Sales', color='Platform',
                     title='Platform Sales Trends Over Time', markers=True)
        st.plotly_chart(fig)

        # Platform market share
        platform_totals = platform_data.groupby('Platform')['Global_Sales'].sum()

        fig = px.pie(values=platform_totals.values, names=platform_totals.index,
                    title='Platform Market Share')
        st.plotly_chart(fig)

with tab7:
    st.header("Favorites & Bookmarks")

    col1, col2 = st.columns([3, 1])

    with col1:
        st.subheader("Add to Favorites")

        # Search and select games to favorite
        favorite_search = st.text_input("Search for games to add to favorites", key='favorite_search')

        if favorite_search:
            search_results = filtered_df[
                filtered_df['Name'].str.contains(favorite_search, case=False, na=False)
            ][['Name', 'Platform', 'Year', 'Genre', 'Global_Sales']].head(10)

            if not search_results.empty:
                selected_favorite = st.selectbox(
                    "Select a game to add to favorites",
                    search_results['Name'] + " (" + search_results['Platform'] + ")",
                    key='favorite_select'
                )

                if st.button("Add to Favorites", key='add_favorite'):
                    game_name = selected_favorite.split(" (")[0]
                    if game_name not in st.session_state.favorites:
                        st.session_state.favorites.append(game_name)
                        st.success(f"Added {game_name} to favorites!")
                    else:
                        st.warning(f"{game_name} is already in favorites!")

    with col2:
        st.subheader("Quick Stats")
        st.metric("Total Favorites", len(st.session_state.favorites))

        if st.session_state.favorites:
            favorite_games = filtered_df[filtered_df['Name'].isin(st.session_state.favorites)]
            if not favorite_games.empty:
                st.metric("Total Favorite Sales", f"{favorite_games['Global_Sales'].sum():.2f}M")

    if st.session_state.favorites:
        st.subheader("Your Favorite Games")

        favorite_games = filtered_df[filtered_df['Name'].isin(st.session_state.favorites)].copy()

        if not favorite_games.empty:
            # Display favorites
            st.dataframe(favorite_games[['Name', 'Platform', 'Year', 'Genre', 'Publisher', 'Global_Sales']])

            # Remove from favorites
            remove_options = st.multiselect("Select games to remove from favorites",
                                          st.session_state.favorites,
                                          key='remove_favorites')

            if st.button("Remove Selected", key='remove_button'):
                for game in remove_options:
                    if game in st.session_state.favorites:
                        st.session_state.favorites.remove(game)
                st.success("Removed selected games from favorites!")
                st.rerun()

            # Export favorites
            if st.button("Export Favorites to CSV", key='export_favorites'):
                csv_data = favorite_games.to_csv(index=False)
                st.download_button(
                    label="Download Favorites CSV",
                    data=csv_data,
                    file_name='favorite_games.csv',
                    mime='text/csv',
                    key='download_favorites'
                )
        else:
            st.info("None of your favorite games match the current filters.")
    else:
        st.info("No favorite games yet. Search and add some games above!")

# Enhanced export functionality in Data tab
with tab3:
    st.header("Data")

    st.subheader("Filtered Data Preview")
    st.dataframe(filtered_df)

    # Multiple export formats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="Download CSV",
            data=csv,
            file_name='filtered_vgsales.csv',
            mime='text/csv',
            key='download-csv'
        )

    with col2:
        json_data = filtered_df.to_json(orient='records', indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name='filtered_vgsales.json',
            mime='application/json',
            key='download-json'
        )

    with col3:
        buffer = io.BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            filtered_df.to_excel(writer, sheet_name='Video Games', index=False)
        buffer.seek(0)
        st.download_button(
            label="Download Excel",
            data=buffer,
            file_name='filtered_vgsales.xlsx',
            mime='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            key='download-excel'
        )

    with col4:
        # Summary statistics export
        summary_stats = filtered_df.describe()
        summary_csv = summary_stats.to_csv()
        st.download_button(
            label="Download Summary Stats",
            data=summary_csv,
            file_name='summary_statistics.csv',
            mime='text/csv',
            key='download-summary'
        )

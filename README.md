# Video Game Sales Dashboard

A comprehensive web application for exploring and visualizing video game sales data with multiple frontend and backend implementations.

## Features

### Streamlit Dashboard
- Interactive dashboard with filters for year, genre, and publisher
- Key metrics: total games, total sales, average sales
- Data preview table with multiple export formats (CSV, JSON, Excel)
- Visualizations:
  - Global sales by genre (bar chart)
  - Sales distribution (histogram)
  - Regional sales comparison (pie chart)
  - Top platforms by sales
  - Sales trends over time
- K-Means clustering analysis
- Advanced analytics with correlation analysis
- Sales forecasting with linear regression
- Genre performance trends
- Game and platform comparison tools
- Favorites/bookmarks system
- Multiple data export options

### REST API (FastAPI)
- Complete REST API for video game sales data
- Endpoints for games, filtering, analytics, and clustering
- CORS enabled for frontend integration
- Pydantic models for data validation
- Comprehensive error handling

### JavaScript Frontend
- Interactive web interface consuming the REST API
- Real-time filtering and search
- Chart.js visualizations
- Pagination for large datasets
- Responsive design

## Setup

1. Ensure you have Python installed (3.7+ recommended).
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Place `vgsales.csv` in the same directory as the application files.

## Running the Applications

### Unified Dashboard (Recommended)
```bash
python open_dashboard.py
```
This opens a single browser window with all three services embedded side-by-side for easy comparison and testing.

### Individual Services
#### Streamlit Dashboard
```bash
streamlit run app.py
```
The dashboard will open in your default web browser at `http://localhost:8501`.

#### REST API Server
```bash
python api.py
```
The API will be available at `http://localhost:8000`.

#### JavaScript Frontend
1. Start the API server first (see above)
2. Open `frontend.html` in your web browser

### Multi-Service Runner
```bash
python run_all.py
```
Starts both Streamlit dashboard and FastAPI server simultaneously.

## API Endpoints

### Games
- `GET /` - API information
- `GET /games` - Get games with filtering and pagination
- `GET /games/{game_id}` - Get specific game by rank

### Analytics
- `GET /summary` - Summary statistics
- `GET /analytics/sales-by-genre` - Sales grouped by genre
- `GET /analytics/sales-by-year` - Sales grouped by year
- `GET /analytics/top-games` - Top games by sales
- `POST /analytics/cluster` - K-means clustering

### Metadata
- `GET /genres` - List all genres
- `GET /platforms` - List all platforms
- `GET /publishers` - List all publishers

## Technologies Used

### Backend
- **Streamlit** - Interactive web dashboard
- **FastAPI** - REST API framework
- **Pandas** - Data manipulation
- **Scikit-learn** - Machine learning (clustering, regression)

### Frontend
- **HTML/CSS/JavaScript** - Traditional web frontend
- **Chart.js** - Data visualization
- **Plotly** - Advanced charts in Streamlit

### Data Processing
- **Matplotlib** - Static plotting
- **Seaborn** - Statistical visualization
- **NumPy** - Numerical computing

## Data Source

The dataset `vgsales.csv` contains video game sales data with columns: Rank, Name, Platform, Year, Genre, Publisher, NA_Sales, EU_Sales, JP_Sales, Other_Sales, Global_Sales.

## Project Structure

```
├── app.py              # Streamlit dashboard
├── api.py              # FastAPI REST API
├── frontend.html       # JavaScript frontend
├── unified_dashboard.html # Combined dashboard (iframes)
├── open_dashboard.py   # Unified dashboard launcher
├── data.css           # Styles for HTML frontend
├── data.html          # Alternative HTML interface
├── index.html         # Static HTML table
├── run_all.py         # Multi-service runner
├── requirements.txt   # Python dependencies
├── vgsales.csv        # Dataset
└── README.md          # This file
```

## New Features Added

1. **Advanced Analytics Tab**: Correlation analysis, sales forecasting, and genre trends
2. **Comparison Tools**: Compare multiple games and platforms side-by-side
3. **Favorites System**: Bookmark favorite games with export functionality
4. **Multiple Export Formats**: CSV, JSON, Excel, and summary statistics
5. **REST API**: Complete backend API with FastAPI
6. **JavaScript Frontend**: Interactive web interface using the API
7. **Enhanced Visualizations**: Interactive Plotly charts
8. **Machine Learning**: K-means clustering and linear regression forecasting

## API Usage Examples

```python
import requests

# Get summary statistics
response = requests.get("http://localhost:8000/summary")
data = response.json()

# Get games with filters
params = {
    "genre": "Action",
    "year_min": 2010,
    "limit": 50
}
response = requests.get("http://localhost:8000/games", params=params)
games = response.json()

# Perform clustering
cluster_data = {
    "n_clusters": 3,
    "features": ["Global_Sales", "NA_Sales", "EU_Sales"]
}
response = requests.post("http://localhost:8000/analytics/cluster", json=cluster_data)
clusters = response.json()
```

## Contributing

Feel free to contribute by:
- Adding new features to the dashboard
- Improving the API endpoints
- Enhancing the frontend interface
- Adding more advanced analytics
- Improving documentation

## Pushing to GitHub

### Quick Setup (Recommended)
1. **Authenticate with GitHub:**
   ```bash
   gh auth login
   ```
   Follow the prompts to authenticate with your GitHub account.

2. **Push to GitHub:**
   ```bash
   # Run the automated script
   push_to_github.bat    # Windows Batch
   # OR
   .\push_to_github.ps1  # PowerShell
   ```

### Manual Steps
If you prefer to do it manually:

1. **Authenticate:**
   ```bash
   gh auth login
   ```

2. **Create and push repository:**
   ```bash
   gh repo create video-game-sales-dashboard --public --source=. --remote=origin --push
   ```

3. **View your repository:**
   Visit: `https://github.com/YOUR_USERNAME/video-game-sales-dashboard`

## Project Structure

```
├── app.py              # Streamlit dashboard
├── api.py              # FastAPI REST API
├── frontend.html       # JavaScript frontend
├── unified_dashboard.html # Combined dashboard (iframes)
├── open_dashboard.py   # Unified dashboard launcher
├── push_to_github.bat  # Windows batch script for GitHub push
├── push_to_github.ps1  # PowerShell script for GitHub push
├── data.css           # Styles for HTML frontend
├── data.html          # Alternative HTML interface
├── index.html         # Static HTML table
├── run_all.py         # Multi-service runner
├── requirements.txt   # Python dependencies
├── vgsales.csv        # Dataset
└── README.md          # This file
```
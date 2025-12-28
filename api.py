from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import pandas as pd
import uvicorn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

app = FastAPI(title="Video Game Sales API", description="REST API for Video Game Sales data")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load data
try:
    df = pd.read_csv("vgsales.csv")
except FileNotFoundError:
    df = pd.DataFrame()

# Pydantic models
class Game(BaseModel):
    Rank: int
    Name: str
    Platform: str
    Year: Optional[float]
    Genre: str
    Publisher: str
    NA_Sales: float
    EU_Sales: float
    JP_Sales: float
    Other_Sales: float
    Global_Sales: float

class GameSummary(BaseModel):
    total_games: int
    total_sales: float
    avg_sales: float
    top_genre: str

class FilterParams(BaseModel):
    genre: Optional[str] = None
    platform: Optional[str] = None
    year_min: Optional[int] = None
    year_max: Optional[int] = None
    publisher: Optional[str] = None

@app.get("/")
def read_root():
    return {"message": "Video Game Sales API", "version": "1.0"}

@app.get("/games", response_model=List[Game])
def get_games(
    skip: int = Query(0, ge=0),
    limit: int = Query(100, ge=1, le=1000),
    genre: Optional[str] = None,
    platform: Optional[str] = None,
    year_min: Optional[int] = None,
    year_max: Optional[int] = None,
    publisher: Optional[str] = None,
    search: Optional[str] = None
):
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    filtered_df = df.copy()

    if genre:
        filtered_df = filtered_df[filtered_df['Genre'].str.lower() == genre.lower()]
    if platform:
        filtered_df = filtered_df[filtered_df['Platform'].str.lower() == platform.lower()]
    if year_min:
        filtered_df = filtered_df[filtered_df['Year'] >= year_min]
    if year_max:
        filtered_df = filtered_df[filtered_df['Year'] <= year_max]
    if publisher:
        filtered_df = filtered_df[filtered_df['Publisher'].str.lower() == publisher.lower()]
    if search:
        filtered_df = filtered_df[filtered_df['Name'].str.contains(search, case=False, na=False)]

    result = filtered_df.iloc[skip:skip + limit].to_dict('records')
    return result

@app.get("/games/{game_id}", response_model=Game)
def get_game(game_id: int):
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    game = df[df['Rank'] == game_id]
    if game.empty:
        raise HTTPException(status_code=404, detail="Game not found")

    return game.iloc[0].to_dict()

@app.get("/summary", response_model=GameSummary)
def get_summary():
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    total_games = len(df)
    total_sales = df['Global_Sales'].sum()
    avg_sales = df['Global_Sales'].mean()
    top_genre = df.groupby('Genre')['Global_Sales'].sum().idxmax()

    return {
        "total_games": total_games,
        "total_sales": round(total_sales, 2),
        "avg_sales": round(avg_sales, 2),
        "top_genre": top_genre
    }

@app.get("/genres")
def get_genres():
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    genres = df['Genre'].unique().tolist()
    return {"genres": genres}

@app.get("/platforms")
def get_platforms():
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    platforms = df['Platform'].unique().tolist()
    return {"platforms": platforms}

@app.get("/publishers")
def get_publishers():
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    publishers = df['Publisher'].dropna().unique().tolist()
    return {"publishers": publishers}

@app.get("/analytics/sales-by-genre")
def get_sales_by_genre():
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    sales_by_genre = df.groupby('Genre')['Global_Sales'].sum().sort_values(ascending=False).to_dict()
    return {"sales_by_genre": sales_by_genre}

@app.get("/analytics/sales-by-year")
def get_sales_by_year():
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    sales_by_year = df.groupby('Year')['Global_Sales'].sum().to_dict()
    return {"sales_by_year": sales_by_year}

@app.get("/analytics/top-games")
def get_top_games(limit: int = Query(10, ge=1, le=100)):
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    top_games = df.nlargest(limit, 'Global_Sales')[['Rank', 'Name', 'Platform', 'Year', 'Genre', 'Global_Sales']].to_dict('records')
    return {"top_games": top_games}

@app.post("/analytics/cluster")
def cluster_games(
    n_clusters: int = Query(3, ge=2, le=10),
    features: List[str] = Query(['Global_Sales', 'NA_Sales', 'EU_Sales'], description="Features to use for clustering")
):
    if df.empty:
        raise HTTPException(status_code=404, detail="Data not found")

    # Validate features
    available_features = ['NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales', 'Global_Sales', 'Year']
    invalid_features = [f for f in features if f not in available_features]
    if invalid_features:
        raise HTTPException(status_code=400, detail=f"Invalid features: {invalid_features}")

    # Prepare data
    cluster_data = df[features].dropna()

    if len(cluster_data) < n_clusters:
        raise HTTPException(status_code=400, detail="Not enough data points for the requested number of clusters")

    # Scale and cluster
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(scaled_data)

    # Add cluster labels
    result_df = cluster_data.copy()
    result_df['Cluster'] = clusters

    # Get cluster centers
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    return {
        "n_clusters": n_clusters,
        "features": features,
        "cluster_centers": centers.tolist(),
        "data_points": result_df.to_dict('records')
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
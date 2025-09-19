import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import re
import random
import time

# Page configuration
st.set_page_config(
    page_title="ARGO Ocean Data Explorer",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        color: white;
        text-align: center;
    }
    
    .stat-card {
        background: linear-gradient(45deg, #3498db, #2980b9);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        margin: 0.5rem;
    }
    
    .example-query {
        background: #e8f4fd;
        color: #2980b9;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        margin: 0.25rem;
        display: inline-block;
        border: 1px solid #3498db;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    .user-message {
        background: #3498db;
        color: white;
        margin-left: 20%;
    }
    
    .assistant-message {
        background: #ecf0f1;
        color: #2c3e50;
        margin-right: 20%;
    }
</style>
""", unsafe_allow_html=True)

# Generate synthetic ARGO data
@st.cache_data
def generate_argo_data():
    """Generate realistic ARGO float data for demo"""
    np.random.seed(42)  # For consistent demo data
    
    # Indian Ocean bounds
    lat_range = (-30, 25)
    lon_range = (30, 100)
    
    n_profiles = 500
    profiles = []
    
    base_date = datetime(2023, 1, 1)
    
    for i in range(n_profiles):
        # Generate realistic location
        lat = np.random.uniform(*lat_range)
        lon = np.random.uniform(*lon_range)
        
        # Date progression
        date = base_date + timedelta(days=np.random.randint(0, 365))
        
        # Realistic temperature based on latitude
        base_temp = 28 - (abs(lat) * 0.3)
        temp_surface = max(15, base_temp + np.random.normal(0, 2))
        temp_100m = max(10, temp_surface - np.random.uniform(2, 5))
        temp_200m = max(8, temp_100m - np.random.uniform(1, 3))
        
        # Realistic salinity with regional variations
        if 60 < lon < 100 and lat > 0:  # Bay of Bengal
            sal_surface = 33.5 + np.random.normal(0, 0.5)
        elif 50 < lon < 70 and lat > 10:  # Arabian Sea
            sal_surface = 36.0 + np.random.normal(0, 0.3)
        else:
            sal_surface = 35.0 + np.random.normal(0, 0.5)
        
        # Determine ocean basin
        if lon < 50:
            basin = "Western Indian Ocean"
        elif 50 <= lon < 80:
            basin = "Arabian Sea"
        elif 80 <= lon < 95 and lat > 5:
            basin = "Bay of Bengal"
        else:
            basin = "Central Indian Ocean"
        
        profiles.append({
            'float_id': f"ARGO_{2900000 + i:04d}",
            'cycle_number': np.random.randint(1, 200),
            'date': date,
            'latitude': round(lat, 3),
            'longitude': round(lon, 3),
            'temp_surface': round(temp_surface, 2),
            'temp_100m': round(temp_100m, 2),
            'temp_200m': round(temp_200m, 2),
            'salinity_surface': round(sal_surface, 2),
            'ocean_basin': basin
        })
    
    return pd.DataFrame(profiles)

# Initialize database
@st.cache_resource
def init_database():
    """Initialize SQLite database with ARGO data"""
    df = generate_argo_data()
    
    conn = sqlite3.connect(':memory:')
    df.to_sql('profiles', conn, if_exists='replace', index=False)
    
    # Create indexes for better performance
    conn.execute("CREATE INDEX idx_lat_lon ON profiles(latitude, longitude)")
    conn.execute("CREATE INDEX idx_date ON profiles(date)")
    conn.execute("CREATE INDEX idx_temp ON profiles(temp_surface)")
    conn.execute("CREATE INDEX idx_basin ON profiles(ocean_basin)")
    
    return conn

# Query processing
def process_query(query, conn):
    """Process natural language query and return results"""
    query_lower = query.lower()
    
    # Hardcoded patterns for reliable demo
    if any(word in query_lower for word in ['temperature', 'temp', 'warm']) and any(word in query_lower for word in ['equator', 'tropical']):
        sql = "SELECT * FROM profiles WHERE latitude BETWEEN -5 AND 5 AND temp_surface IS NOT NULL ORDER BY temp_surface DESC LIMIT 50"
        explanation = "Found temperature profiles near the equatorial region (-5° to 5° latitude)"
        
    elif any(word in query_lower for word in ['arabian sea', 'arabian']):
        sql = "SELECT * FROM profiles WHERE ocean_basin = 'Arabian Sea' ORDER BY temp_surface DESC LIMIT 50"
        explanation = "Retrieved data from Arabian Sea region"
        
    elif any(word in query_lower for word in ['bay of bengal', 'bengal']):
        sql = "SELECT * FROM profiles WHERE ocean_basin = 'Bay of Bengal' ORDER BY date DESC LIMIT 50"
        explanation = "Found profiles from Bay of Bengal region"
        
    elif any(word in query_lower for word in ['salinity', 'salt']):
        sql = "SELECT * FROM profiles WHERE salinity_surface IS NOT NULL ORDER BY salinity_surface DESC LIMIT 50"
        explanation = "Retrieved salinity measurements from all regions"
        
    elif any(word in query_lower for word in ['warm', 'hot', 'highest temperature']):
        sql = "SELECT * FROM profiles WHERE temp_surface > 28 ORDER BY temp_surface DESC LIMIT 30"
        explanation = "Found the warmest water locations (>28°C)"
        
    elif any(word in query_lower for word in ['cold', 'cool', 'lowest temperature']):
        sql = "SELECT * FROM profiles WHERE temp_surface < 22 ORDER BY temp_surface ASC LIMIT 30"
        explanation = "Located cooler water regions (<22°C)"
        
    elif any(word in query_lower for word in ['recent', 'latest', 'new']):
        sql = "SELECT * FROM profiles ORDER BY date DESC LIMIT 40"
        explanation = "Retrieved most recent ARGO float measurements"
        
    elif any(word in query_lower for word in ['deep', 'depth', '200m']):
        sql = "SELECT * FROM profiles WHERE temp_200m IS NOT NULL ORDER BY temp_200m DESC LIMIT 40"
        explanation = "Found temperature profiles at 200m depth"
        
    else:
        # Default fallback query
        sql = "SELECT * FROM profiles ORDER BY date DESC LIMIT 30"
        explanation = "Showing recent ARGO float profiles"
    
    # Execute query
    try:
        df = pd.read_sql_query(sql, conn)
        return df, explanation
    except Exception as e:
        st.error(f"Query error: {e}")
        # Return sample data as fallback
        fallback_sql = "SELECT * FROM profiles LIMIT 20"
        df = pd.read_sql_query(fallback_sql, conn)
        return df, "Showing sample ARGO data"

# Visualization functions
def create_map_visualization(df):
    """Create interactive map of ARGO float locations"""
    if df.empty:
        st.warning("No data to display on map")
        return
    
    fig = px.scatter_mapbox(
        df,
        lat="latitude",
        lon="longitude",
        color="temp_surface",
        size="temp_surface",
        hover_data={
            'float_id': True,
            'date': True,
            'temp_surface': True,
            'salinity_surface': True,
            'ocean_basin': True,
            'latitude': ':.2f',
            'longitude': ':.2f'
        },
        color_continuous_scale="Viridis",
        size_max=15,
        zoom=3,
        title="ARGO Float Locations Colored by Surface Temperature",
        labels={'temp_surface': 'Temperature (°C)'}
    )
    
    fig.update_layout(
        mapbox_style="open-street-map",
        height=500,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

def create_temperature_chart(df):
    """Create temperature distribution chart"""
    if df.empty or 'temp_surface' not in df.columns:
        return None
    
    fig = px.histogram(
        df,
        x="temp_surface",
        title="Surface Temperature Distribution",
        labels={'temp_surface': 'Temperature (°C)', 'count': 'Number of Profiles'},
        color_discrete_sequence=['#3498db']
    )
    
    fig.update_layout(
        height=300,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

def create_salinity_chart(df):
    """Create salinity by region chart"""
    if df.empty or 'salinity_surface' not in df.columns:
        return None
    
    # Group by ocean basin
    basin_salinity = df.groupby('ocean_basin')['salinity_surface'].mean().reset_index()
    
    fig = px.bar(
        basin_salinity,
        x="ocean_basin",
        y="salinity_surface",
        title="Average Salinity by Ocean Basin",
        labels={'salinity_surface': 'Salinity (PSU)', 'ocean_basin': 'Ocean Basin'},
        color='salinity_surface',
        color_continuous_scale='Blues'
    )
    
    fig.update_layout(
        height=300,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

def create_depth_profile(df):
    """Create temperature-depth profile"""
    if df.empty:
        return None
    
    # Sample some profiles for depth visualization
    sample_profiles = df.head(10)
    
    fig = go.Figure()
    
    for idx, row in sample_profiles.iterrows():
        temps = [row['temp_surface'], row['temp_100m'], row['temp_200m']]
        depths = [0, 100, 200]
        
        fig.add_trace(go.Scatter(
            x=temps,
            y=depths,
            mode='lines+markers',
            name=f"Float {row['float_id'][-4:]}",
            line=dict(width=2),
            marker=dict(size=6)
        ))
    
    fig.update_layout(
        title="Temperature vs Depth Profiles",
        xaxis_title="Temperature (°C)",
        yaxis_title="Depth (m)",
        yaxis=dict(autorange='reversed'),
        height=350,
        margin={"r":0,"t":50,"l":0,"b":0}
    )
    
    return fig

# Initialize database
conn = init_database()

# Main UI
def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌊 ARGO Ocean Data Explorer</h1>
        <p>AI-Powered Natural Language Interface for Oceanographic Data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Statistics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="stat-card">
            <h3>500</h3>
            <p>Sample Profiles</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="stat-card">
            <h3>Indian Ocean</h3>
            <p>Primary Region</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="stat-card">
            <h3>2023</h3>
            <p>Sample Data Year</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="stat-card">
            <h3>AI-Powered</h3>
            <p>Query Processing</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Main content layout
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("🗺️ Interactive Ocean Data Map")
        
        # Initialize session state for query results
        if 'query_results' not in st.session_state:
            # Load initial data
            initial_df = pd.read_sql_query("SELECT * FROM profiles ORDER BY date DESC LIMIT 50", conn)
            st.session_state.query_results = initial_df
        
        # Display map
        if not st.session_state.query_results.empty:
            map_fig = create_map_visualization(st.session_state.query_results)
            if map_fig:
                st.plotly_chart(map_fig, use_container_width=True)
        
    with col2:
        st.subheader("💬 Natural Language Queries")
        
        # Example queries
        st.markdown("**Try these examples:**")
        example_queries = [
            "Show me temperature near the equator",
            "Find warm water in Arabian Sea",
            "Salinity in Bay of Bengal",
            "Recent measurements",
            "Coldest water locations",
            "Deep temperature profiles"
        ]
        
        for query in example_queries:
            if st.button(query, key=f"example_{query}", help="Click to use this example"):
                # Process the query
                with st.spinner("🔄 Processing query..."):
                    time.sleep(1)  # Simulate processing
                    results_df, explanation = process_query(query, conn)
                    st.session_state.query_results = results_df
                    st.session_state.last_explanation = explanation
                    st.rerun()
        
        st.markdown("---")
        
        # Custom query input
        st.markdown("**Or ask your own question:**")
        custom_query = st.text_input(
            "Enter your query:",
            placeholder="e.g., Show me salinity data from recent measurements",
            help="Ask about temperature, salinity, locations, or time periods"
        )
        
        if st.button("🔍 Search", type="primary"):
            if custom_query.strip():
                with st.spinner("🔄 Processing your query..."):
                    time.sleep(1.5)  # Simulate AI processing
                    results_df, explanation = process_query(custom_query, conn)
                    st.session_state.query_results = results_df
                    st.session_state.last_explanation = explanation
                    st.rerun()
            else:
                st.warning("Please enter a query first!")
    
    # Display query explanation
    if 'last_explanation' in st.session_state:
        st.info(f"📊 **Query Results:** {st.session_state.last_explanation}")
    
    # Visualization section
    st.markdown("---")
    st.subheader("📈 Data Visualizations")
    
    if not st.session_state.query_results.empty:
        # Create visualization tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📊 Temperature Distribution", "🧂 Salinity by Region", "📉 Depth Profiles", "📋 Data Table"])
        
        with tab1:
            temp_chart = create_temperature_chart(st.session_state.query_results)
            if temp_chart:
                st.plotly_chart(temp_chart, use_container_width=True)
            else:
                st.info("No temperature data available for visualization")
        
        with tab2:
            sal_chart = create_salinity_chart(st.session_state.query_results)
            if sal_chart:
                st.plotly_chart(sal_chart, use_container_width=True)
            else:
                st.info("No salinity data available for visualization")
        
        with tab3:
            depth_chart = create_depth_profile(st.session_state.query_results)
            if depth_chart:
                st.plotly_chart(depth_chart, use_container_width=True)
            else:
                st.info("No depth profile data available")
        
        with tab4:
            st.dataframe(
                st.session_state.query_results,
                use_container_width=True,
                height=400
            )
            
            # Download option
            csv = st.session_state.query_results.to_csv(index=False)
            st.download_button(
                label="📥 Download Data as CSV",
                data=csv,
                file_name=f"argo_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    else:
        st.info("No data to display. Try running a query first!")
    
    # Sidebar with additional information
    with st.sidebar:
        st.header("ℹ️ About ARGO Data")
        st.info("""
        **ARGO floats** are autonomous oceanographic instruments that drift through the ocean, collecting temperature and salinity data.
        
        This demo uses synthetic data modeled after real ARGO measurements from the Indian Ocean region.
        """)
        
        st.header("🎯 Query Examples")
        st.markdown("""
        **Temperature Queries:**
        - "Show warmest water"
        - "Temperature near equator"
        
        **Regional Queries:**
        - "Arabian Sea data"
        - "Bay of Bengal salinity"
        
        **Time-based:**
        - "Recent measurements"
        - "Latest profiles"
        
        **Depth-based:**
        - "Deep temperature"
        - "200m depth data"
        """)
        
        st.header("📈 Data Statistics")
        if not st.session_state.query_results.empty:
            df = st.session_state.query_results
            st.metric("Profiles Found", len(df))
            if 'temp_surface' in df.columns:
                st.metric("Avg Temperature", f"{df['temp_surface'].mean():.1f}°C")
            if 'salinity_surface' in df.columns:
                st.metric("Avg Salinity", f"{df['salinity_surface'].mean():.1f} PSU")
        
        st.header("🔧 Technical Info")
        st.markdown("""
        **Technologies Used:**
        - Python & Streamlit
        - Plotly for visualizations
        - SQLite for data storage
        - Pattern matching for query processing
        """)

if __name__ == "__main__":
    main()
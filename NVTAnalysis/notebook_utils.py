import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def plot_experiments_across_australia(df: DataFrame):
    gdf = gpd.GeoDataFrame(
        df, 
        geometry=gpd.points_from_xy(
        df['Trial GPS Long'],
        df['Trial GPS Lat']),
        crs="EPSG:4326"
    )
    aus_states = gpd.read_file("/g/data/xe2/ya6227/NVTAnalysis/data/gadm41_AUS_1.shp")
    gdf_filtered = gdf.dropna(subset=["Single Site Yield", "RegionName"])

    gdf_filtered = gdf.dropna(subset=["Single Site Yield", "RegionName"])

    # Group by GPS + Region to get average yield per site
    gdf_grouped = (
        gdf_filtered.groupby(["Trial GPS Lat", "Trial GPS Long", "RegionName"])
        .agg({
            "Single Site Yield": "mean",
            "geometry": "first"
        })
        .reset_index()
    )
    
    # Convert back to GeoDataFrame
    gdf_grouped = gpd.GeoDataFrame(gdf_grouped, geometry="geometry", crs="EPSG:4326")
    
    # Normalize marker size
    min_size, max_size = 20, 200
    yield_min = gdf_grouped["Single Site Yield"].min()
    yield_max = gdf_grouped["Single Site Yield"].max()
    gdf_grouped["marker_size"] = gdf_grouped["Single Site Yield"].apply(
        lambda y: min_size + (y - yield_min) / (yield_max - yield_min) * (max_size - min_size)
    )
    
    
    unique_regions = gdf_grouped["RegionName"].unique()
    num_regions = len(unique_regions)
    colormap = cm.get_cmap("tab20", num_regions)  # You can also try "hsv" or "tab20b"
    
    # Create region → color map using hex codes
    region_colors = {
        region: mcolors.to_hex(colormap(i)) for i, region in enumerate(unique_regions)
    }
    
    # Plot
    fig, ax = plt.subplots(figsize=(16, 16))
    aus_states.plot(ax=ax, color='white', edgecolor='black')
    
    for region, group in gdf_grouped.groupby("RegionName"):
        group.plot(
            ax=ax,
            color=region_colors[region],
            markersize=group["marker_size"],
            edgecolor="black",
            alpha=0.7,
            label=region
        )
    
    ax.set_title("NVT Trial Sites by Region with Yield-based Marker Size (2017–2024)", fontsize=14)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.legend(title="Region", bbox_to_anchor=(1.05, 1), loc='upper left')
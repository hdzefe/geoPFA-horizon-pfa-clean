"""
Filter GeoTIS temperature data to North German Basin extent using shapefile
Removes all points outside basin boundary to ensure representative data only
"""

import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
from shapely.geometry import Point
import re

# Load basin boundary
basin_shp = 'data/inputs/extent/north_german_basin_.shp'
basin_gdf = gpd.read_file(basin_shp)

print(f"Basin shapefile loaded:")
print(f"  CRS: {basin_gdf.crs}")
print(f"  Geometry type: {basin_gdf.geometry.type.values}")
print(f"  Bounds: {basin_gdf.total_bounds}")

# Create unified polygon from all geometries
basin_polygon = basin_gdf.geometry.unary_union
print(f"\n✓ Basin polygon created (area: {basin_polygon.area/1e6:.0f} km²)")

# Process all GeoTIS files
geoTIS_dir = Path('data/inputs/geotIS/temperature')
output_dir = Path('data/inputs/geotIS/temperature_basin_filtered')
output_dir.mkdir(exist_ok=True)

data_files = sorted(geoTIS_dir.glob("*.DATA")) + sorted(geoTIS_dir.glob("*.data"))
print(f"\nProcessing {len(data_files)} GeoTIS files...")

for idx, data_file in enumerate(data_files, 1):
    filename = data_file.stem
    match = re.search(r'([+-]\d+)', filename)
    
    if not match:
        continue
    
    depth_str = match.group(1)
    
    # Read file
    df = pd.read_csv(data_file, delimiter=';', skiprows=3,
                      names=['X','Y','T','STDV'], low_memory=False,
                      skipinitialspace=True, decimal=',')
    
    # Convert to numeric
    df['X'] = pd.to_numeric(df['X'], errors='coerce')
    df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
    df['T'] = pd.to_numeric(df['T'], errors='coerce')
    df['STDV'] = pd.to_numeric(df['STDV'], errors='coerce')
    df = df.dropna()
    
    # Filter out missing value marker (-99999)
    df = df[(df['T'] != -99999.0) & (df['STDV'] != -99999.0)]
    
    if len(df) == 0:
        print(f"  [{idx:3d}] {filename:35s} - NO VALID DATA")
        continue
    
    # Create geometries
    geometry = [Point(xy) for xy in zip(df['X'], df['Y'])]
    gdf = gpd.GeoDataFrame(df, geometry=geometry, crs=basin_gdf.crs)
    
    # Spatial filter: only points within basin
    gdf_filtered = gdf[gdf.geometry.within(basin_polygon)]
    
    if len(gdf_filtered) == 0:
        print(f"  [{idx:3d}] {filename:35s} - NO POINTS IN BASIN (all {len(df)} outside)")
        continue
    
    # Save filtered file
    output_file = output_dir / data_file.name
    gdf_filtered[['X', 'Y', 'T', 'STDV']].to_csv(
        output_file, sep=';', index=False, decimal='.',
        float_format=lambda x: f'{x:.1f}' if pd.notna(x) else '-99999'
    )
    
    print(f"  [{idx:3d}] {filename:35s} - {len(df):6d} → {len(gdf_filtered):6d} points | T: {gdf_filtered['T'].min():6.1f}-{gdf_filtered['T'].max():6.1f}°C")

print(f"\n✓ Filtered GeoTIS data saved to: {output_dir}")
print(f"\nNext: Update thermal_data_loader.py to use filtered data from:")
print(f"  {output_dir}")

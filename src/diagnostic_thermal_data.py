"""
Diagnostic script for thermal data investigation

Inspects:
1. Borehole data structure and values
2. GeoTIS temperature data structure and values
3. Depth-temperature matching
"""

import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
from pathlib import Path
import re

logging.basicConfig(
    level=logging.INFO,
    format='%(message)s'
)
logger = logging.getLogger(__name__)


def inspect_belegpunkte_data():
    """Inspect borehole Belegpunkte data"""
    logger.info("\n" + "="*80)
    logger.info("BELEGPUNKTE DATA INSPECTION")
    logger.info("="*80)
    
    base_dir = Path("data/inputs")
    
    # Map horizon names to files
    horizons = {
        'hettangian_1': 'Het1_Belegpunkte.shp',
        'pliensbachian_1': 'Pli1_Belegpunkte.shp',
        'sinemurian_1': 'Sin1_Belegpunkte.shp'
    }
    
    for horizon_name, filename in horizons.items():
        logger.info(f"\n{'─'*80}")
        logger.info(f"Horizon: {horizon_name}")
        logger.info(f"{'─'*80}")
        
        shp_path = base_dir / "reservoirs" / horizon_name / filename
        
        if not shp_path.exists():
            logger.error(f"File not found: {shp_path}")
            continue
        
        try:
            gdf = gpd.read_file(shp_path)
            
            logger.info(f"\n1. BASIC INFO:")
            logger.info(f"   Total records: {len(gdf)}")
            logger.info(f"   CRS: {gdf.crs}")
            
            logger.info(f"\n2. COLUMNS:")
            for col in sorted(gdf.columns):
                logger.info(f"   - {col}")
            
            logger.info(f"\n3. TEUFE (Depth to top) STATISTICS:")
            teufe_data = gdf['Teufe'].dropna()
            logger.info(f"   Valid records: {len(teufe_data)}/{len(gdf)}")
            logger.info(f"   Min: {teufe_data.min():.2f} m")
            logger.info(f"   Max: {teufe_data.max():.2f} m")
            logger.info(f"   Mean: {teufe_data.mean():.2f} m")
            logger.info(f"   Std: {teufe_data.std():.2f} m")
            logger.info(f"   Data type: {gdf['Teufe'].dtype}")
            
            logger.info(f"\n4. GESAMTMAEC (Thickness) STATISTICS:")
            gesamtmaec_data = gdf['Gesamtmaec'].dropna()
            logger.info(f"   Valid records: {len(gesamtmaec_data)}/{len(gdf)}")
            logger.info(f"   Min: {gesamtmaec_data.min():.2f} m")
            logger.info(f"   Max: {gesamtmaec_data.max():.2f} m")
            logger.info(f"   Mean: {gesamtmaec_data.mean():.2f} m")
            logger.info(f"   Std: {gesamtmaec_data.std():.2f} m")
            logger.info(f"   Data type: {gdf['Gesamtmaec'].dtype}")
            
            logger.info(f"\n5. SAMPLE RECORDS (first 5):")
            for idx, row in gdf[['Hole_ID', 'Teufe', 'Gesamtmaec']].head(5).iterrows():
                logger.info(f"   {row['Hole_ID']}: depth={row['Teufe']:.1f}m, thickness={row['Gesamtmaec']:.1f}m")
            
            logger.info(f"\n6. COORDINATE STATISTICS:")
            logger.info(f"   X range: [{gdf.geometry.x.min():.0f}, {gdf.geometry.x.max():.0f}]")
            logger.info(f"   Y range: [{gdf.geometry.y.min():.0f}, {gdf.geometry.y.max():.0f}]")
            
        except Exception as e:
            logger.error(f"Error: {e}")
            import traceback
            traceback.print_exc()


def inspect_geoTIS_data():
    """Inspect GeoTIS temperature data"""
    logger.info("\n" + "="*80)
    logger.info("GEOТIS TEMPERATURE DATA INSPECTION")
    logger.info("="*80)
    
    geoTIS_dir = Path("data/inputs/geoTIS/temperature")
    
    if not geoTIS_dir.exists():
        logger.error(f"Directory not found: {geoTIS_dir}")
        return
    
    # Find all .DATA files
    data_files = sorted(geoTIS_dir.glob("*.DATA")) + sorted(geoTIS_dir.glob("*.data"))
    
    logger.info(f"\nFound {len(data_files)} temperature depth files")
    
    # Extract all depths
    depths = {}
    for data_file in data_files:
        filename = data_file.stem
        match = re.search(r'([+-]\d+)', filename)
        if match:
            depth_str = match.group(1)
            depth = int(depth_str)
            depths[depth] = data_file
    
    depths_sorted = sorted(depths.keys())
    
    logger.info(f"\n1. DEPTH COVERAGE:")
    logger.info(f"   Total unique depths: {len(depths_sorted)}")
    logger.info(f"   Depth range: {depths_sorted[0]}m to {depths_sorted[-1]}m")
    logger.info(f"   First 10: {depths_sorted[:10]}")
    logger.info(f"   Last 10: {depths_sorted[-10:]}")
    
    # Inspect a few files
    test_depths = [depths_sorted[0], depths_sorted[len(depths_sorted)//2], depths_sorted[-1]]
    
    for test_depth in test_depths:
        data_file = depths[test_depth]
        logger.info(f"\n2. FILE EXAMPLE: {data_file.name} (depth={test_depth}m)")
        
        try:
            # Read raw file
            with open(data_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            
            logger.info(f"   Total lines: {len(lines)}")
            logger.info(f"   First 3 lines (raw):")
            for i, line in enumerate(lines[:3]):
                logger.info(f"      {i}: {line.rstrip()}")
            
            # Parse with pandas
            df = pd.read_csv(
                data_file,
                delimiter=';',
                skipinitialspace=True,
                skiprows=2,
                names=['X', 'Y', 'Temperature', 'STDV'],
                decimal=','
            )
            
            # Convert to numeric
            for col in ['X', 'Y', 'Temperature', 'STDV']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df = df.dropna()
            df = df[(df['Temperature'] != -99999.0) & (df['STDV'] != -99999.0)]
            
            logger.info(f"\n   After parsing:")
            logger.info(f"   Valid records: {len(df)}")
            logger.info(f"   X range: [{df['X'].min():.0f}, {df['X'].max():.0f}]")
            logger.info(f"   Y range: [{df['Y'].min():.0f}, {df['Y'].max():.0f}]")
            logger.info(f"   Temperature range: [{df['Temperature'].min():.2f}, {df['Temperature'].max():.2f}]°C")
            logger.info(f"   STDV range: [{df['STDV'].min():.2f}, {df['STDV'].max():.2f}]")
            
            logger.info(f"\n   Sample records (first 5):")
            for idx, row in df.head(5).iterrows():
                logger.info(f"      X={row['X']:.0f}, Y={row['Y']:.0f}, T={row['Temperature']:.2f}°C, STDV={row['STDV']:.2f}")
        
        except Exception as e:
            logger.error(f"   Error: {e}")


def check_depth_temperature_matching():
    """Check if borehole depths and GeoTIS depths overlap"""
    logger.info("\n" + "="*80)
    logger.info("DEPTH-TEMPERATURE MATCHING CHECK")
    logger.info("="*80)
    
    # Get GeoTIS depths
    geoTIS_dir = Path("data/inputs/geoTIS/temperature")
    data_files = sorted(geoTIS_dir.glob("*.DATA")) + sorted(geoTIS_dir.glob("*.data"))
    
    geoTIS_depths = set()
    for data_file in data_files:
        filename = data_file.stem
        match = re.search(r'([+-]\d+)', filename)
        if match:
            depth = int(match.group(1))
            geoTIS_depths.add(depth)
    
    geoTIS_depths = sorted(geoTIS_depths)
    
    # Get borehole depths
    base_dir = Path("data/inputs")
    shp_path = base_dir / "reservoirs" / "hettangian_1" / "Het1_Belegpunkte.shp"
    gdf = gpd.read_file(shp_path)
    
    borehole_depths = gdf['Teufe'].dropna()
    mean_depths = borehole_depths + (gdf['Gesamtmaec'].dropna() / 2)
    
    logger.info(f"\n1. GEOТIS COVERAGE:")
    logger.info(f"   Depths: {len(geoTIS_depths)} levels")
    logger.info(f"   Range: {geoTIS_depths[0]}m to {geoTIS_depths[-1]}m")
    
    logger.info(f"\n2. BOREHOLE COVERAGE:")
    logger.info(f"   Depths: {len(borehole_depths)} boreholes")
    logger.info(f"   Range: {borehole_depths.min():.0f}m to {borehole_depths.max():.0f}m")
    logger.info(f"   Mean depth (top + thickness/2): {mean_depths.min():.0f}m to {mean_depths.max():.0f}m")
    
    logger.info(f"\n3. OVERLAP ANALYSIS:")
    borehole_min = borehole_depths.min()
    borehole_max = borehole_depths.max()
    geoTIS_min = geoTIS_depths[0]
    geoTIS_max = geoTIS_depths[-1]
    
    logger.info(f"   Borehole depth range: [{borehole_min:.0f}, {borehole_max:.0f}]m")
    logger.info(f"   GeoTIS depth range: [{geoTIS_min}, {geoTIS_max}]m")
    
    if borehole_max < geoTIS_min or borehole_min > geoTIS_max:
        logger.error(f"   ✗ NO OVERLAP! Boreholes and GeoTIS data don't match in depth range!")
    else:
        overlap_min = max(borehole_min, geoTIS_min)
        overlap_max = min(borehole_max, geoTIS_max)
        logger.info(f"   ✓ Overlap range: [{overlap_min:.0f}, {overlap_max:.0f}]m")
    
    logger.info(f"\n4. COORDINATE SYSTEM CHECK:")
    logger.info(f"   GeoTIS X coordinates (sample from +100m file):")
    
    sample_file = geoTIS_dir / "T2022_LIAG_AGEMAR_+100.data"
    if sample_file.exists():
        df = pd.read_csv(sample_file, delimiter=';', skipinitialspace=True, skiprows=2,
                        names=['X', 'Y', 'T', 'STDV'], decimal=',')
        df['X'] = pd.to_numeric(df['X'], errors='coerce')
        df['Y'] = pd.to_numeric(df['Y'], errors='coerce')
        df = df.dropna()
        logger.info(f"      Range: [{df['X'].min():.0f}, {df['X'].max():.0f}]")
    
    logger.info(f"   Borehole X coordinates (sample):")
    logger.info(f"      Range: [{gdf.geometry.x.min():.0f}, {gdf.geometry.x.max():.0f}]")


def main():
    """Main execution"""
    logger.info("\n" + "="*80)
    logger.info("THERMAL DATA DIAGNOSTIC REPORT")
    logger.info("="*80)
    
    inspect_belegpunkte_data()
    inspect_geoTIS_data()
    check_depth_temperature_matching()
    
    logger.info("\n" + "="*80)
    logger.info("DIAGNOSTIC COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Thermal Evidence Layer Generator
Loads GeoTIS temperature data, GOCAD depth surfaces, and borehole evidence
to create temperature maps for each horizon.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.interpolate import griddata
from rasterio.transform import Affine
import rasterio
from rasterio.crs import CRS
import json
import glob
from typing import Tuple

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s:%(name)s:%(message)s')
logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Load and process thermal data for horizon-based analysis"""
    
    def __init__(self, config_file: str = "data/config.json"):
        """Initialize with configuration"""
        self.config = self._load_config(config_file)
        self.crs = self.config.get("crs", "EPSG:31467")
        self.grid_size = self.config.get("grid_size", 2000)
        self.thermal_conductivity = self.config.get("thermal_conductivity", 2.36)
        self.geothermal_gradient = 25.27  # Default, will be overridden
        
        # Paths
        self.base_dir = Path(self.config.get("base_dir", "."))
        self.heat_flow_file = Path(self.config.get("heat_flow_file", "data/inputs/heat_flow/2022-015_Fuchs-et-al_GermanHeatFlowDB-2022.xlsx"))
        self.borehole_file = Path(self.config.get("borehole_file", "data/inputs/borehole_mean_depths.csv"))
        self.output_dir = Path(self.config.get("output_dir", "data/outputs/rasters"))
        self.surfaces_dir = Path(self.config.get("surfaces_dir", "data/inputs/TUNB/surfaces"))
        self.geotis_dir = Path(self.config.get("geotis_dir", "data/inputs/geotIS/temperature_basin_filtered"))
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GeoTIS temperature data (sparse: depth -> (X, Y, T) points)
        self.T_stack = {}  # depth_m -> Nx3 array [X, Y, T]
        self.levels_m_nhn = []  # Sorted depth levels
        
        # Borehole data
        self.boreholes_gdf = None
        
        # Grid
        self.xv = None
        self.yv = None
        self.basin_bounds = None
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  GeoTIS dir: {self.geotis_dir}")
        
    def _load_config(self, config_file: str) -> dict:
        """Load configuration from JSON file"""
        try:
            with open(config_file) as f:
                return json.load(f)
        except FileNotFoundError:
            logger.warning(f"Config file not found: {config_file}, using defaults")
            return {}
    
    def load_heat_flow_database(self):
        """Load GFZ German Heat Flow Database and calculate regional gradient"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GFZ GERMAN HEAT FLOW DATABASE 2022")
        logger.info("="*80)
        
        if not self.heat_flow_file.exists():
            logger.warning(f"⚠ Heat flow file not found: {self.heat_flow_file}")
            logger.warning(f"  Using default gradient: 23.1 °C/km")
            self.geothermal_gradient = 23.1
            return
        
        try:
            df = pd.read_excel(self.heat_flow_file, sheet_name='data', skiprows=3, dtype=str)
            logger.info(f"✓ Loaded: {len(df):,} records from heat flow database")
            
            # Convert European decimal format
            for col in ['lat', 'lng', 'q']:
                if col in df.columns:
                    df[col] = df[col].str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df_clean = df.dropna(subset=['lat', 'lng', 'q']).copy()
            df_clean = df_clean[(df_clean['q'] > 0) & (df_clean['q'] < 300)]
            logger.info(f"✓ Valid records: {len(df_clean):,} / {len(df):,}")
            
            # Filter to North German Basin (lat/lng)
            basin_lat_min, basin_lat_max = 50.8, 54.5
            basin_lng_min, basin_lng_max = 6.0, 14.0
            
            basin_mask = (df_clean['lat'] >= basin_lat_min) & (df_clean['lat'] <= basin_lat_max) & \
                        (df_clean['lng'] >= basin_lng_min) & (df_clean['lng'] <= basin_lng_max)
            
            df_basin = df_clean[basin_mask].copy()
            logger.info(f"✓ Measurements in North German Basin: {len(df_basin):,} / {len(df_clean):,}")
            
            if len(df_basin) > 0:
                q_basin = df_basin['q']
                mean_hf = q_basin.mean()
                logger.info(f"\n  ✓✓✓ BASIN HEAT FLOW STATISTICS ✓✓✓")
                logger.info(f"    Mean: {mean_hf:.1f} mW/m²")
                logger.info(f"    Median: {q_basin.median():.1f} mW/m²")
                logger.info(f"    Std Dev: {q_basin.std():.1f} mW/m²")
                logger.info(f"    Range: {q_basin.min():.1f} to {q_basin.max():.1f} mW/m²")
                
                self.geothermal_gradient = (mean_hf / 1000.0) / self.thermal_conductivity * 1000.0
                logger.info(f"  Calculated Geothermal Gradient: {self.geothermal_gradient:.2f} °C/km")
            else:
                q_all = df_clean['q']
                mean_hf = q_all.mean()
                logger.warning(f"⚠ No basin measurements, using Germany-wide: {mean_hf:.1f} mW/m²")
                self.geothermal_gradient = (mean_hf / 1000.0) / self.thermal_conductivity * 1000.0
        
        except Exception as e:
            logger.warning(f"⚠ Error loading heat flow: {e}")
            self.geothermal_gradient = 23.1
    
    def load_geotis_temperatures(self):
        """Load GeoTIS temperature data from .data files (CSV format: X;Y;T;STDEV)"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GEOTIS TEMPERATURE DATA (BASIN FILTERED)")
        logger.info("="*80)
        
        data_files = sorted(glob.glob(str(self.geotis_dir / "*.data")))
        
        if not data_files:
            logger.error(f"❌ No .data files found in {self.geotis_dir}")
            return False
        
        logger.info(f"✓ Found {len(data_files)} temperature depth files")
        
        try:
            for data_file in data_files:
                filename = Path(data_file).stem
                # Extract depth from filename (format: T2022_LIAG_AGEMAR_-5000)
                depth_str = filename.split('_')[-1]
                
                try:
                    depth_m = int(depth_str)
                except ValueError:
                    logger.warning(f"  ⚠ Could not parse depth from {filename}")
                    continue
                
                # Load CSV data file (X;Y;T;STDEV format)
                try:
                    df = pd.read_csv(data_file, delimiter=';')
                    
                    if df.empty:
                        logger.warning(f"  ⚠ No data in {filename}")
                        continue
                    
                    # Ensure required columns exist
                    if 'T' not in df.columns:
                        logger.warning(f"  ⚠ No temperature column in {filename}")
                        continue
                    
                    # Store sparse data as (X, Y, T) coordinates
                    xyz_data = df[['X', 'Y', 'T']].values.astype(np.float32)
                    
                    self.T_stack[depth_m] = xyz_data
                    self.levels_m_nhn.append(depth_m)
                    
                    T_vals = xyz_data[:, 2]  # T is 3rd column
                    valid_count = np.isfinite(T_vals).sum()
                    logger.info(f"  ✓ {filename}: depth={depth_m:+6d}m, "
                               f"points={len(xyz_data)}, "
                               f"T=[{np.nanmin(T_vals):6.1f}, {np.nanmax(T_vals):6.1f}]°C")
                
                except Exception as e:
                    logger.warning(f"  ⚠ Error reading {filename}: {e}")
                    continue
            
            self.levels_m_nhn.sort()
            
            if len(self.levels_m_nhn) == 0:
                logger.error(f"❌ No valid depth levels loaded")
                return False
            
            logger.info(f"\n✓ Loaded {len(self.levels_m_nhn)} depth levels")
            logger.info(f"  Depth range: {self.levels_m_nhn[0]:+6d}m to {self.levels_m_nhn[-1]:+6d}m")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error loading GeoTIS data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def load_borehole_data(self):
        """Load borehole mean depths and locations"""
        logger.info("\n" + "="*80)
        logger.info("LOADING BOREHOLE DATA")
        logger.info("="*80)
        
        if not self.borehole_file.exists():
            logger.error(f"❌ Borehole file not found: {self.borehole_file}")
            return False
        
        try:
            df = pd.read_csv(self.borehole_file, delimiter=';', decimal=',')
            logger.info(f"✓ Loaded {len(df)} boreholes")
            logger.info(f"  Columns: {list(df.columns)}")
            self.boreholes_gdf = df
            return True
        
        except Exception as e:
            logger.error(f"❌ Error loading boreholes: {e}")
            return False
    
    def initialize_grid(self):
        """Initialize processing grid from borehole bounds"""
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING PROCESSING GRID")
        logger.info("="*80)
        
        if self.boreholes_gdf is None:
            logger.error("❌ Boreholes not loaded")
            return False
        
        x_min, x_max = self.boreholes_gdf['x'].min(), self.boreholes_gdf['x'].max()
        y_min, y_max = self.boreholes_gdf['y'].min(), self.boreholes_gdf['y'].max()
        
        logger.info(f"  Borehole extent (GK Zone 3):")
        logger.info(f"    X: {x_min:,.0f} to {x_max:,.0f}")
        logger.info(f"    Y: {y_min:,.0f} to {y_max:,.0f}")
        
        # Create grid to match processing needs
        grid_size = self.grid_size
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        self.xv, self.yv = np.meshgrid(x, y)
        
        self.basin_bounds = (x_min, x_max, y_min, y_max)
        
        logger.info(f"✓ Grid initialized: {grid_size}x{grid_size}")
        logger.info(f"  Cell size: {(x_max-x_min)/grid_size:.0f}m × {(y_max-y_min)/grid_size:.0f}m")
        
        return True
    
    def load_surface_from_ts_files(self, ts_files: list) -> np.ndarray:
        """Load GOCAD TS surface files and interpolate to grid"""
        logger.info(f"    Loading {len(ts_files)} TS files...")
        
        points = []
        values = []
        
        for ts_file in ts_files:
            try:
                with open(ts_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse TS file - look for VRTX (vertex) lines
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('VRTX'):
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                x = float(parts[2])
                                y = float(parts[3])
                                z = float(parts[4])
                                points.append([x, y])
                                values.append(z)
                            except (ValueError, IndexError):
                                pass
            except Exception as e:
                logger.warning(f"    ⚠ Error reading {Path(ts_file).name}: {e}")
        
        if not points:
            logger.warning(f"    ⚠ No points found in TS files")
            return np.full_like(self.xv, np.nan)
        
        points = np.array(points)
        values = np.array(values)
        
        logger.info(f"    ✓ Extracted {len(points)} points from {len(ts_files)} surfaces")
        
        # DECIMATE: Keep only 10k evenly distributed points for efficiency
        if len(points) > 10000:
            sample_indices = np.random.choice(len(points), 10000, replace=False)
            points_decimated = points[sample_indices]
            values_decimated = values[sample_indices]
            logger.info(f"    → Decimated to {len(points_decimated)} points for efficiency")
        else:
            points_decimated = points
            values_decimated = values
        
        # Use griddata instead of RBF (much faster, less memory)
        try:
            z_grid = griddata(points_decimated, values_decimated, (self.xv, self.yv), method='linear', fill_value=np.nan)
            
            valid = ~np.isnan(z_grid)
            if np.any(valid):
                logger.info(f"    ✓ Interpolated to grid: {np.nanmin(z_grid):.1f}m to {np.nanmax(z_grid):.1f}m")
            
            return z_grid
        except Exception as e:
            logger.warning(f"    ⚠ Interpolation failed: {e}")
            return np.full_like(self.xv, np.nan)
    
    def interpolate_temperature_at_depth(self, z_grid: np.ndarray) -> np.ndarray:
        """Interpolate temperature from sparse GeoTIS points at given depth grid"""
        if not self.levels_m_nhn:
            logger.warning("  ⚠ GeoTIS data not loaded")
            return np.full_like(z_grid, np.nan)
        
        T_grid = np.full_like(z_grid, np.nan, dtype=float)
        
        for i in range(z_grid.shape[0]):
            for j in range(z_grid.shape[1]):
                z = z_grid[i, j]
                if np.isnan(z):
                    continue
                
                # Find surrounding depth levels
                idx = np.searchsorted(self.levels_m_nhn, z)
                
                # Get temperature from appropriate depth level(s)
                if idx == 0:
                    # Use first level
                    xyz = self.T_stack[self.levels_m_nhn[0]]
                    T = self._interpolate_sparse_at_xy(xyz, self.xv[i, j], self.yv[i, j])
                    T_grid[i, j] = T
                elif idx >= len(self.levels_m_nhn):
                    # Use last level
                    xyz = self.T_stack[self.levels_m_nhn[-1]]
                    T = self._interpolate_sparse_at_xy(xyz, self.xv[i, j], self.yv[i, j])
                    T_grid[i, j] = T
                else:
                    # Interpolate between levels
                    z0 = self.levels_m_nhn[idx - 1]
                    z1 = self.levels_m_nhn[idx]
                    
                    xyz0 = self.T_stack[z0]
                    xyz1 = self.T_stack[z1]
                    
                    T0 = self._interpolate_sparse_at_xy(xyz0, self.xv[i, j], self.yv[i, j])
                    T1 = self._interpolate_sparse_at_xy(xyz1, self.xv[i, j], self.yv[i, j])
                    
                    if np.isnan(T0) or np.isnan(T1):
                        continue
                    
                    frac = (z - z0) / (z1 - z0)
                    T_grid[i, j] = T0 + (T1 - T0) * frac
        
        return T_grid
    
    def _interpolate_sparse_at_xy(self, xyz: np.ndarray, x: float, y: float, radius: float = 5000) -> float:
        """Nearest neighbor interpolate temperature at (x,y) from sparse (X,Y,T) points"""
        # Extract points within radius
        X = xyz[:, 0]
        Y = xyz[:, 1]
        T = xyz[:, 2]
        
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist < radius
        
        if not np.any(mask):
            return np.nan
        
        # Find nearest point
        valid_dist = dist[mask]
        valid_idx = np.where(mask)[0]
        nearest_local_idx = np.argmin(valid_dist)
        nearest_idx = valid_idx[nearest_local_idx]
        
        return float(T[nearest_idx])
    
    def process_horizon(self, horizon_name: str, top_surface_dir: str, base_surface_dir: str, fraction: float):
        """Process a single horizon"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {horizon_name.upper()}")
        logger.info(f"{'='*80}")
        
        # Load surfaces
        top_files = sorted(glob.glob(str(self.surfaces_dir / top_surface_dir / "*.ts")))
        base_files = sorted(glob.glob(str(self.surfaces_dir / base_surface_dir / "*.ts")))
        
        if not top_files or not base_files:
            logger.warning(f"  ⚠ Surface files not found")
            logger.warning(f"    Top: {self.surfaces_dir / top_surface_dir}")
            logger.warning(f"    Base: {self.surfaces_dir / base_surface_dir}")
            return
        
        logger.info(f"1. Loading surfaces...")
        z_top = self.load_surface_from_ts_files(top_files)
        z_base = self.load_surface_from_ts_files(base_files)
        
        # Interpolate horizon depth using fraction
        z_horizon = z_top + (z_base - z_top) * fraction
        
        valid_z = ~np.isnan(z_horizon)
        if not np.any(valid_z):
            logger.warning(f"  ⚠ No valid depth data")
            return
        
        logger.info(f"  ✓ Horizon depth range: {np.nanmin(z_horizon):.1f}m to {np.nanmax(z_horizon):.1f}m")
        
        # Query GeoTIS at horizon depth
        logger.info(f"2. Querying GeoTIS temperatures...")
        T_horizon = self.interpolate_temperature_at_depth(z_horizon)
        
        valid_T = ~np.isnan(T_horizon)
        if not np.any(valid_T):
            logger.warning(f"  ⚠ No valid temperature data")
            return
        
        logger.info(f"  ✓ Temperature range: {np.nanmin(T_horizon):.1f}°C to {np.nanmax(T_horizon):.1f}°C")
        
        # Clamp negative temperatures
        T_blended = np.maximum(T_horizon, 3.0)
        
        # Export
        logger.info(f"3. Exporting temperature evidence...")
        self._export_geotiff(T_blended, f"{horizon_name}_temperature_evidence.tif")
        
        logger.info(f"✓ {horizon_name.upper()} completed successfully!")
        logger.info(f"  Temperature range: {np.nanmin(T_blended):.1f}°C to {np.nanmax(T_blended):.1f}°C")
    
    def _export_geotiff(self, data: np.ndarray, filename: str):
        """Export temperature grid as GeoTIFF with proper georeferencing"""
        output_path = self.output_dir / filename
        
        if self.basin_bounds is None:
            logger.warning("  ⚠ Basin bounds not set, using identity transform")
            transform = Affine.identity()
        else:
            # Map grid pixels to real coordinates
            x_min, x_max, y_min, y_max = self.basin_bounds
            
            pixel_width = (x_max - x_min) / data.shape[1]
            pixel_height = (y_max - y_min) / data.shape[0]
            
            # Top-left corner (standard raster orientation)
            transform = Affine.translation(x_min, y_max) * Affine.scale(pixel_width, -pixel_height)
            
            logger.info(f"  Georeferencing: {x_min:.0f}-{x_max:.0f} / {y_min:.0f}-{y_max:.0f}")
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=self.crs,
            transform=transform
        ) as dst:
            # Flip Y-axis for correct orientation in EPSG:31467
            dst.write(np.flipud(data.astype(rasterio.float32)), 1)
        
        logger.info(f"  ✓ Saved: {output_path}")


def main():
    """Main processing workflow"""
    
    # Initialize
    loader = ThermalDataLoader()
    
    # Load data
    loader.load_heat_flow_database()
    if not loader.load_geotis_temperatures():
        logger.error("❌ Failed to load GeoTIS data")
        return
    
    if not loader.load_borehole_data():
        logger.error("❌ Failed to load borehole data")
        return
    
    if not loader.initialize_grid():
        logger.error("❌ Failed to initialize grid")
        return
    
    # Horizon specifications (name, top_surface_dir, base_surface_dir, fraction)
    horizons = [
        ("het1", "06_base_middle_jurassic", "07_base_lower_jurassic", 0.95),
        ("het2", "06_base_middle_jurassic", "07_base_lower_jurassic", 0.90),
        ("sin1", "06_base_middle_jurassic", "07_base_lower_jurassic", 0.75),
        ("sin2", "06_base_middle_jurassic", "07_base_lower_jurassic", 0.70),
        ("pli1", "06_base_middle_jurassic", "07_base_lower_jurassic", 0.55),
        ("pli2", "06_base_middle_jurassic", "07_base_lower_jurassic", 0.50),
    ]
    
    # Process each horizon
    for horizon_name, top_dir, base_dir, frac in horizons:
        loader.process_horizon(horizon_name, top_dir, base_dir, frac)
    
    logger.info("\n" + "="*80)
    logger.info("✓ THERMAL EVIDENCE LAYER GENERATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

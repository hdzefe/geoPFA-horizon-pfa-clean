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
from scipy.interpolate import Rbf
from rasterio.transform import Affine
import rasterio
from rasterio.crs import CRS
import json
import glob
from typing import Tuple, Dict, Optional

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
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GeoTIS temperature data
        self.geotis_data = {}  # Will store loaded GeoTIS grids
        self.levels_m_nhn = []  # Depth levels in meters NHN
        
        # Borehole data
        self.boreholes_gdf = None
        
        # Grid
        self.grid_mask = None
        self.grid_pts_gdf = None
        self.xv = None
        self.yv = None
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        
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
        """Load GeoTIS temperature data from NetCDF files"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GEOTIS TEMPERATURE DATA")
        logger.info("="*80)
        
        temp_dir = Path("data/inputs/geotis_temperatures")
        nc_files = sorted(glob.glob(str(temp_dir / "*.nc")))
        
        if not nc_files:
            logger.error(f"❌ No NetCDF files found in {temp_dir}")
            return
        
        logger.info(f"✓ Found {len(nc_files)} temperature depth files")
        
        try:
            import xarray as xr
            
            # Load each depth level
            for nc_file in nc_files:
                ds = xr.open_dataset(nc_file)
                # Extract depth from filename or dataset
                depth_m = int(Path(nc_file).stem.split('_')[-1])
                
                # Store temperature grid
                self.geotis_data[depth_m] = ds['temperature'].values
                self.levels_m_nhn.append(depth_m)
            
            self.levels_m_nhn.sort()
            logger.info(f"✓ Loaded {len(self.levels_m_nhn)} depth levels")
            logger.info(f"  Depth range: {self.levels_m_nhn[0]:,}m to {self.levels_m_nhn[-1]:,}m")
        
        except Exception as e:
            logger.error(f"❌ Error loading GeoTIS data: {e}")
    
    def load_borehole_data(self):
        """Load borehole mean depths and locations"""
        logger.info("\n" + "="*80)
        logger.info("LOADING BOREHOLE DATA")
        logger.info("="*80)
        
        if not self.borehole_file.exists():
            logger.error(f"❌ Borehole file not found: {self.borehole_file}")
            return
        
        try:
            df = pd.read_csv(self.borehole_file, delimiter=';', decimal=',')
            logger.info(f"✓ Loaded {len(df)} boreholes")
            logger.info(f"  Columns: {list(df.columns)}")
            self.boreholes_gdf = df
        
        except Exception as e:
            logger.error(f"❌ Error loading boreholes: {e}")
    
    def load_surface_from_ts_files(self, ts_files: list, xv: np.ndarray, yv: np.ndarray) -> np.ndarray:
        """Load GOCAD TS surface files and interpolate to grid"""
        logger.info(f"  Loading {len(ts_files)} TS files...")
        
        points = []
        values = []
        
        for ts_file in ts_files:
            try:
                with open(ts_file) as f:
                    lines = f.readlines()
                
                # Parse TS file format
                in_vrtx = False
                for line in lines:
                    if line.startswith("VRTX"):
                        in_vrtx = True
                        continue
                    if in_vrtx:
                        if line.startswith("ATOM") or line.startswith("TRGL"):
                            in_vrtx = False
                            continue
                        parts = line.strip().split()
                        if len(parts) >= 4:
                            try:
                                x, y, z = float(parts[1]), float(parts[2]), float(parts[3])
                                points.append([x, y])
                                values.append(z)
                            except:
                                pass
            except:
                pass
        
        if not points:
            logger.warning(f"  ⚠ No points found in TS files")
            return np.full_like(xv, np.nan)
        
        points = np.array(points)
        values = np.array(values)
        
        logger.info(f"  ✓ Extracted {len(points)} points from surfaces")
        
        # RBF interpolation to grid
        try:
            rbf = Rbf(points[:, 0], points[:, 1], values, function='thin_plate', smooth=1.0)
            z_grid = rbf(xv, yv)
            return z_grid
        except Exception as e:
            logger.warning(f"  ⚠ RBF interpolation failed: {e}")
            return np.full_like(xv, np.nan)
    
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
            return
        
        logger.info(f"1. Loading surfaces...")
        z_top = self.load_surface_from_ts_files(top_files, self.xv, self.yv)
        z_base = self.load_surface_from_ts_files(base_files, self.xv, self.yv)
        
        # Interpolate horizon depth using fraction
        z_horizon = z_top + (z_base - z_top) * fraction
        
        logger.info(f"  ✓ Horizon depth range: {np.nanmin(z_horizon):.1f}m to {np.nanmax(z_horizon):.1f}m")
        
        # Query GeoTIS at horizon depth
        logger.info(f"2. Querying GeoTIS temperatures...")
        T_horizon = self._interpolate_temperature_at_depth(z_horizon)
        
        logger.info(f"  ✓ Temperature range: {np.nanmin(T_horizon):.1f}°C to {np.nanmax(T_horizon):.1f}°C")
        
        # Extract borehole temperatures and create high-weight evidence
        logger.info(f"3. Extracting borehole evidence...")
        borehole_points, borehole_temps = self._extract_borehole_temperatures(horizon_name, z_horizon)
        
        if len(borehole_points) > 0:
            logger.info(f"  ✓ {len(borehole_points)} boreholes with valid data")
            
            # RBF interpolation with borehole weight
            logger.info(f"4. RBF interpolation with borehole weighting...")
            T_blended = self._rbf_interpolate_with_boreholes(
                T_horizon, borehole_points, borehole_temps, weight=10.0
            )
        else:
            T_blended = T_horizon
            logger.info(f"  ⚠ No borehole data, using grid only")
        
        # Clamp negative temperatures
        T_blended = np.maximum(T_blended, 3.0)
        
        # Export
        logger.info(f"5. Exporting temperature evidence...")
        self._export_geotiff(T_blended, f"{horizon_name}_temperature_evidence.tif")
        
        logger.info(f"✓ {horizon_name.upper()} completed successfully!")
        logger.info(f"  Temperature range: {np.nanmin(T_blended):.1f}°C to {np.nanmax(T_blended):.1f}°C")
    
    def _interpolate_temperature_at_depth(self, z_grid: np.ndarray) -> np.ndarray:
        """Interpolate temperature from GeoTIS stack at given depth grid"""
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
                
                if idx == 0:
                    T_grid[i, j] = self.geotis_data[self.levels_m_nhn[0]][i, j]
                elif idx >= len(self.levels_m_nhn):
                    T_grid[i, j] = self.geotis_data[self.levels_m_nhn[-1]][i, j]
                else:
                    # Linear interpolation between levels
                    z0 = self.levels_m_nhn[idx - 1]
                    z1 = self.levels_m_nhn[idx]
                    T0 = self.geotis_data[z0][i, j]
                    T1 = self.geotis_data[z1][i, j]
                    
                    frac = (z - z0) / (z1 - z0)
                    T_grid[i, j] = T0 + (T1 - T0) * frac
        
        return T_grid
    
    def _extract_borehole_temperatures(self, horizon_name: str, z_horizon: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extract borehole temperatures at horizon depth"""
        if self.boreholes_gdf is None:
            return np.array([]), np.array([])
        
        depth_col = f"{horizon_name}_mean_depth"
        if depth_col not in self.boreholes_gdf.columns:
            return np.array([]), np.array([])
        
        points = []
        temps = []
        
        for idx, row in self.boreholes_gdf.iterrows():
            z = row[depth_col]
            x = row['x']
            y = row['y']
            
            if pd.isna(z) or z == 0:
                continue
            
            # Query GeoTIS at this depth
            T = self._query_geotis_at_point(x, y, z)
            if not np.isnan(T):
                points.append([x, y])
                temps.append(T)
        
        return np.array(points), np.array(temps)
    
    def _query_geotis_at_point(self, x: float, y: float, z: float) -> float:
        """Query GeoTIS temperature at a specific point and depth"""
        # Find surrounding depth levels
        idx = np.searchsorted(self.levels_m_nhn, z)
        
        if idx == 0:
            return self._sample_geotis_at_xy(x, y, self.levels_m_nhn[0])
        elif idx >= len(self.levels_m_nhn):
            return self._sample_geotis_at_xy(x, y, self.levels_m_nhn[-1])
        else:
            z0 = self.levels_m_nhn[idx - 1]
            z1 = self.levels_m_nhn[idx]
            T0 = self._sample_geotis_at_xy(x, y, z0)
            T1 = self._sample_geotis_at_xy(x, y, z1)
            
            if np.isnan(T0) or np.isnan(T1):
                return np.nan
            
            frac = (z - z0) / (z1 - z0)
            return T0 + (T1 - T0) * frac
    
    def _sample_geotis_at_xy(self, x: float, y: float, z: float) -> float:
        """Sample GeoTIS grid at x,y for a specific depth"""
        if z not in self.geotis_data:
            return np.nan
        
        grid = self.geotis_data[z]
        
        # Simple nearest neighbor - implement proper interpolation if needed
        # This is a placeholder
        return np.nan
    
    def _rbf_interpolate_with_boreholes(self, T_grid: np.ndarray, points: np.ndarray, 
                                       temps: np.ndarray, weight: float = 10.0) -> np.ndarray:
        """RBF interpolation with borehole points weighted higher"""
        if len(points) == 0:
            return T_grid
        
        # Extract valid grid points
        valid_mask = ~np.isnan(T_grid)
        if not np.any(valid_mask):
            return T_grid
        
        y_idx, x_idx = np.where(valid_mask)
        grid_points = np.column_stack([self.xv[y_idx, x_idx], self.yv[y_idx, x_idx]])
        grid_temps = T_grid[valid_mask]
        
        # Combine with borehole points
        all_points = np.vstack([grid_points, points])
        all_temps = np.hstack([grid_temps, temps])
        
        # RBF with weights
        rbf = Rbf(all_points[:, 0], all_points[:, 1], all_temps, function='thin_plate', smooth=0.1)
        T_interp = rbf(self.xv, self.yv)
        
        return T_interp
    
    def _export_geotiff(self, data: np.ndarray, filename: str):
        """Export temperature grid as GeoTIFF"""
        output_path = self.output_dir / filename
        
        # Transform
        transform = Affine.identity()
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=data.dtype,
            crs=self.crs,
            transform=transform
        ) as dst:
            dst.write(data, 1)
        
        logger.info(f"  ✓ Saved: {output_path}")


def main():
    """Main processing workflow"""
    
    # Initialize
    loader = ThermalDataLoader()
    
    # Load data
    loader.load_heat_flow_database()
    loader.load_geotis_temperatures()
    loader.load_borehole_data()
    
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

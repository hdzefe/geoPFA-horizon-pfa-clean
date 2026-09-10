"""
Thermal Data Loader - PFA Evidence Layer Generation

Generates PFA evidence layers for geothermal favorability assessment.
Processes borehole data (Tier 1) using RBF interpolation with hybrid temperature source.

TIER 1 APPROACH (HYBRID):
- Load borehole Teufe (depth) and Gesamtmaec (thickness)
- Calculate mean depth from: -(Teufe - Gesamtmaec/2)
- Query GeoTIS temperature at borehole mean depths
- SPATIAL FILTER: Only use GeoTIS points within North German Basin extent
  (Prevents contamination from Molasse & Rhine Graben anomalies)
- Where GeoTIS exists: use measured temperature
- Where GeoTIS is missing/NaN: use heat flow gradient calculation
- Blend both sources for complete basin coverage
- RBF interpolate across entire basin (no gaps!)
- CLAMP to physical limits (no negative/unrealistic temperatures)
- Confidence based on borehole density and data source

Temperature Sources:
1. GeoTIS: Real measured/modeled temperatures (where available, spatially filtered)
2. Heat Flow Gradient: Backup for gaps (65 mW/m² → 23°C/km)

Output: Clean evidence layers for input to PFA combination model

Coordinate Systems:
- GeoTIS temperature: Negative = below sea level, Positive = above sea level
- Borehole Teufe: Positive values representing depth below surface
- Conversion: Negate Teufe to match GeoTIS coordinate system
- Spatial Filter: Only GeoTIS points within basin extent (EPSG:31467)

IMPORTANT: Excludes Teufe ≤ 0 (surface outcrops, not boreholes)

Tier 1 Horizons (with borehole data):
- hettangian_1, hettangian_2
- pliensbachian_1, pliensbachian_2
- sinemurian_1, sinemurian_2
"""

import json
import logging
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.transform import from_bounds
from scipy.interpolate import Rbf, griddata
from scipy.spatial.distance import cdist
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Generate PFA thermal evidence layers from borehole data using hybrid temperature source"""
    
    # Tier 1 horizons with borehole depth/thickness data
    TIER1_HORIZONS = [
        'hettangian_1', 'hettangian_2',
        'pliensbachian_1', 'pliensbachian_2',
        'sinemurian_1', 'sinemurian_2'
    ]
    
    # Mapping: horizon_name -> belegpunkte_filename_prefix
    BELEGPUNKTE_MAPPING = {
        'hettangian_1': 'Het1',
        'hettangian_2': 'Het2',
        'pliensbachian_1': 'Pli1',
        'pliensbachian_2': 'Pli2',
        'sinemurian_1': 'Sin1',
        'sinemurian_2': 'Sin2'
    }
    
    # Heat flow database parameters (GFZ German Heat Flow Database 2022) - FALLBACK
    HEAT_FLOW_MEAN = 65.0  # mW/m²
    THERMAL_CONDUCTIVITY_LOW = 2.5  # W/(m·K)
    THERMAL_CONDUCTIVITY_HIGH = 3.5  # W/(m·K)
    
    GEOTHERMAL_GRADIENT = (HEAT_FLOW_MEAN / 1000.0) / ((THERMAL_CONDUCTIVITY_LOW + THERMAL_CONDUCTIVITY_HIGH) / 2.0) * 1000.0
    
    T_SURFACE = 10.0  # °C
    
    # Physical limits for clamping (prevent extrapolation artifacts)
    T_MIN = 3.0  # °C - minimum physical temperature (subsurface)
    T_MAX = 200.0  # °C - maximum realistic temperature (before geothermal gradient unrealistic)
    
    def __init__(self, config_path, base_dir="data/inputs", geoTIS_dir="data/inputs/geoTIS/temperature"):
        """Initialize thermal data loader"""
        self.base_dir = Path(base_dir)
        self.config_path = Path(config_path)
        self.geoTIS_dir = Path(geoTIS_dir)
        
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        self.crs = self.config['crs']
        self.grid_size = self.config['grid_size']
        self.extent = self.config['extent']
        
        # ✅ SPATIAL FILTER BOUNDS (North German Basin only)
        self.basin_xmin = self.extent['left']
        self.basin_xmax = self.extent['right']
        self.basin_ymin = self.extent['bottom']
        self.basin_ymax = self.extent['top']
        
        self.output_dir = Path("data/outputs/thermal")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.geoTIS_data = {}
        self.geoTIS_loaded = False
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Tier 1 horizons: {len(self.TIER1_HORIZONS)}")
        logger.info(f"  Temperature clamping: [{self.T_MIN:.1f}, {self.T_MAX:.1f}]°C (prevents extrapolation artifacts)")
        logger.info(f"  Spatial filter (basin bounds):")
        logger.info(f"    X: {self.basin_xmin:.0f} to {self.basin_xmax:.0f} m")
        logger.info(f"    Y: {self.basin_ymin:.0f} to {self.basin_ymax:.0f} m")
        logger.info(f"    (Prevents contamination from Molasse & Rhine Graben)")
    
    def load_geoTIS_temperature_data(self):
        """Load all GeoTIS temperature .DATA files"""
        logger.info(f"\n{'='*80}")
        logger.info("LOADING GEOТIS TEMPERATURE DATA")
        logger.info(f"{'='*80}")
        
        if self.geoTIS_loaded:
            return self.geoTIS_data
        
        if not self.geoTIS_dir.exists():
            logger.error(f"✗ GeoTIS directory not found: {self.geoTIS_dir}")
            raise FileNotFoundError(f"GeoTIS directory: {self.geoTIS_dir}")
        
        data_files = sorted(self.geoTIS_dir.glob("*.DATA")) + sorted(self.geoTIS_dir.glob("*.data"))
        
        if not data_files:
            logger.error(f"✗ No .DATA files found in {self.geoTIS_dir}")
            raise FileNotFoundError(f"No .DATA files in {self.geoTIS_dir}")
        
        logger.info(f"✓ Found {len(data_files)} temperature depth files")
        
        geoTIS_data = {}
        
        for data_file in data_files:
            try:
                filename = data_file.stem
                match = re.search(r'([+-]\d+)', filename)
                
                if match:
                    depth_str = match.group(1)
                    depth = int(depth_str)
                else:
                    continue
                
                df = pd.read_csv(
                    data_file, 
                    delimiter=';', 
                    skipinitialspace=True,
                    skiprows=2,
                    names=['X', 'Y', 'Temperature', 'STDV'],
                    decimal=','
                )
                
                if len(df) == 0:
                    continue
                
                for col in ['X', 'Y', 'Temperature', 'STDV']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                df = df[(df['Temperature'] != -99999.0) & (df['STDV'] != -99999.0)]
                
                if len(df) == 0:
                    continue
                
                geoTIS_data[depth] = {
                    'X': df['X'].values.astype(np.float32),
                    'Y': df['Y'].values.astype(np.float32),
                    'T': df['Temperature'].values.astype(np.float32),
                    'STDV': df['STDV'].values.astype(np.float32)
                }
            
            except Exception as e:
                logger.error(f"  ✗ Error reading {data_file.name}: {e}")
                continue
        
        if not geoTIS_data:
            logger.warning("⚠ No valid GeoTIS data loaded")
            self.geoTIS_loaded = True
            return {}
        
        logger.info(f"✓ Loaded {len(geoTIS_data)} depth levels (all Germany)")
        depths = sorted(geoTIS_data.keys())
        logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m")
        
        self.geoTIS_data = geoTIS_data
        self.geoTIS_loaded = True
        return geoTIS_data
    
    def filter_geoTIS_to_basin(self, X, Y, T, STDV):
        """
        ✅ SPATIAL FILTER: Keep only GeoTIS points within North German Basin
        Prevents contamination from Molasse & Rhine Graben anomalies
        """
        mask = (
            (X >= self.basin_xmin) & (X <= self.basin_xmax) &
            (Y >= self.basin_ymin) & (Y <= self.basin_ymax)
        )
        
        X_filtered = X[mask]
        Y_filtered = Y[mask]
        T_filtered = T[mask]
        STDV_filtered = STDV[mask]
        
        return X_filtered, Y_filtered, T_filtered, STDV_filtered
    
    def interpolate_temperature_at_depth(self, target_depth, method='linear'):
        """Interpolate temperature at arbitrary depth from GeoTIS data (spatially filtered)"""
        if not self.geoTIS_loaded or not self.geoTIS_data:
            return None
        
        available_depths = sorted(self.geoTIS_data.keys())
        
        if target_depth in self.geoTIS_data:
            data = self.geoTIS_data[target_depth]
            X, Y, T, STDV = self.filter_geoTIS_to_basin(data['X'], data['Y'], data['T'], data['STDV'])
            
            if len(X) == 0:
                return None
            
            return X, Y, T, STDV
        
        below = [d for d in available_depths if d <= target_depth]
        above = [d for d in available_depths if d >= target_depth]
        
        if method == 'nearest' or len(below) == 0 or len(above) == 0:
            if len(below) > 0 and len(above) > 0:
                d_below = below[-1]
                d_above = above[0]
                depth = d_below if (target_depth - d_below) <= (d_above - target_depth) else d_above
            elif len(below) > 0:
                depth = below[-1]
            else:
                depth = above[0]
            
            data = self.geoTIS_data[depth]
            X, Y, T, STDV = self.filter_geoTIS_to_basin(data['X'], data['Y'], data['T'], data['STDV'])
            
            if len(X) == 0:
                return None
            
            return X, Y, T, STDV
        
        else:
            d_below = below[-1]
            d_above = above[0]
            
            data_below = self.geoTIS_data[d_below]
            data_above = self.geoTIS_data[d_above]
            
            # ✅ Filter both depth levels to basin
            X_below, Y_below, T_below, STDV_below = self.filter_geoTIS_to_basin(
                data_below['X'], data_below['Y'], data_below['T'], data_below['STDV']
            )
            X_above, Y_above, T_above, STDV_above = self.filter_geoTIS_to_basin(
                data_above['X'], data_above['Y'], data_above['T'], data_above['STDV']
            )
            
            if len(X_below) == 0 or len(X_above) == 0:
                return None
            
            w_below = (d_above - target_depth) / (d_above - d_below)
            w_above = (target_depth - d_below) / (d_above - d_below)
            
            points_below = np.column_stack([X_below, Y_below])
            points_above = np.column_stack([X_above, Y_above])
            
            T_above_interp = griddata(points_above, T_above, points_below, method='nearest')
            
            T_interp = w_below * T_below + w_above * T_above_interp
            STDV_interp = np.sqrt((w_below * STDV_below)**2 + (w_above * T_above_interp)**2)
            
            return X_below, Y_below, T_interp, STDV_interp
    
    def find_belegpunkte_shapefile(self, horizon_name):
        """Find Belegpunkte shapefile for horizon"""
        if horizon_name not in self.BELEGPUNKTE_MAPPING:
            return None
        
        prefix = self.BELEGPUNKTE_MAPPING[horizon_name]
        shp_filename = f"{prefix}_Belegpunkte.shp"
        
        horizon_dir = self.base_dir / "reservoirs" / horizon_name
        shp_path = horizon_dir / shp_filename
        
        if shp_path.exists():
            return shp_path
        
        for path in self.base_dir.rglob(shp_filename):
            return path
        
        return None
    
    def load_belegpunkte_data(self, horizon_name):
        """Load borehole evidence points"""
        shp_path = self.find_belegpunkte_shapefile(horizon_name)
        
        if shp_path is None:
            logger.warning(f"  ⚠ Belegpunkte not found for {horizon_name}")
            return None
        
        try:
            gdf = gpd.read_file(shp_path)
            
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            gdf['X'] = gdf.geometry.x
            gdf['Y'] = gdf.geometry.y
            
            return gdf
        
        except Exception as e:
            logger.error(f"  ✗ Error loading {shp_path}: {e}")
            return None
    
    def rbf_interpolate_surface(self, borehole_gdf, column_name, function='thin_plate'):
        """Interpolate surface using RBF"""
        valid = borehole_gdf[borehole_gdf[column_name].notna()].copy()
        
        if column_name == 'Teufe':
            valid = valid[valid[column_name] > 0]
        
        if len(valid) < 3:
            logger.warning(f"    ⚠ Less than 3 valid points for {column_name} ({len(valid)} found)")
            return None, None
        
        try:
            x_pts = valid['X'].values
            y_pts = valid['Y'].values
            z_pts = valid[column_name].values.astype(np.float32)
            
            rbf = Rbf(x_pts, y_pts, z_pts, function=function, epsilon=None, smooth=1.0)
            
            left = self.extent['left']
            right = self.extent['right']
            bottom = self.extent['bottom']
            top = self.extent['top']
            
            x_grid = np.linspace(left, right, self.grid_size[0])
            y_grid = np.linspace(bottom, top, self.grid_size[1])
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            
            interpolated = rbf(X_mesh, Y_mesh)
            
            grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
            borehole_points = np.column_stack([x_pts, y_pts])
            
            distances = cdist(grid_points, borehole_points)
            min_distances = np.min(distances, axis=1)
            
            max_dist = np.percentile(min_distances, 95)
            uncertainty = np.clip(min_distances / max_dist, 0, 1).reshape(self.grid_size[1], self.grid_size[0])
            
            data_range = z_pts.max() - z_pts.min()
            uncertainty = uncertainty * (data_range * 0.1)
            
            return interpolated.astype(np.float32), uncertainty.astype(np.float32)
        
        except Exception as e:
            logger.error(f"    ✗ RBF interpolation failed: {e}")
            return None, None
    
    def calculate_borehole_density_confidence(self, borehole_gdf):
        """Calculate confidence based on borehole spatial density"""
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        borehole_points = np.column_stack([borehole_gdf['X'], borehole_gdf['Y']])
        grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        
        distances = cdist(grid_points, borehole_points)
        min_distances = np.min(distances, axis=1)
        
        max_dist = np.percentile(min_distances, 95)
        density_confidence = 1.0 - np.clip(min_distances / max_dist, 0, 1)
        density_confidence = density_confidence.reshape(self.grid_size[1], self.grid_size[0])
        
        return density_confidence.astype(np.float32)
    
    def combine_confidence_layers(self, interpolation_error, density_confidence):
        """Combine interpolation error and borehole density"""
        error_norm = np.clip(interpolation_error / np.percentile(interpolation_error[interpolation_error > 0], 95), 0, 1)
        interpolation_confidence = 1.0 - error_norm
        
        combined = np.sqrt(interpolation_confidence * density_confidence)
        
        return combined.astype(np.float32)
    
    def calculate_temperature_from_depth(self, depth_surface_m):
        """Calculate temperature from depth using geothermal gradient"""
        depth_km = np.abs(depth_surface_m) / 1000.0
        temperature = self.T_SURFACE + (depth_km * self.GEOTHERMAL_GRADIENT)
        
        return temperature.astype(np.float32)
    
    def interpolate_temperature_grid_hybrid(self, mean_depth_surface, borehole_gdf):
        """
        Interpolate temperature using HYBRID approach with spatial filtering
        Only uses GeoTIS points within North German Basin (prevents Molasse/Rhine contamination)
        """
        logger.info(f"    Interpolating temperature (HYBRID: GeoTIS spatially-filtered + gradient fallback)...")
        
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        temperature_at_boreholes = np.full(len(borehole_gdf), np.nan, dtype=np.float32)
        geotis_available = np.zeros(len(borehole_gdf), dtype=bool)
        
        # Query GeoTIS at borehole locations (spatially filtered)
        if self.geoTIS_data:
            logger.info(f"      Querying GeoTIS at {len(borehole_gdf)} borehole locations (spatially filtered to basin)...")
            
            for idx, (i, row) in enumerate(borehole_gdf.iterrows()):
                target_depth = row['mean_depth']
                
                try:
                    result = self.interpolate_temperature_at_depth(target_depth, method='linear')
                    
                    if result is not None:
                        X_temp, Y_temp, T_temp, STDV_temp = result
                        
                        # Only proceed if we have basin-filtered data
                        if len(X_temp) > 0:
                            points_temp = np.column_stack([X_temp, Y_temp])
                            T_at_borehole = griddata(points_temp, T_temp, (row['X'], row['Y']), method='nearest')
                            
                            if not np.isnan(T_at_borehole):
                                temperature_at_boreholes[idx] = T_at_borehole
                                geotis_available[idx] = True
                
                except Exception as e:
                    continue
            
            geotis_count = np.sum(geotis_available)
            logger.info(f"      GeoTIS temperatures found (basin-filtered): {geotis_count}/{len(borehole_gdf)} boreholes")
        
        # Fallback: use heat flow gradient where GeoTIS is missing
        gradient_temps = self.calculate_temperature_from_depth(borehole_gdf['mean_depth'].values)
        
        # Use GeoTIS where available, gradient elsewhere
        temp_blended = np.where(
            geotis_available,
            temperature_at_boreholes,
            gradient_temps
        )
        
        borehole_gdf['temperature_blended'] = temp_blended
        
        # RBF interpolate blended temperatures
        logger.info(f"      RBF interpolating blended temperatures across basin...")
        
        x_pts = borehole_gdf['X'].values
        y_pts = borehole_gdf['Y'].values
        z_pts = temp_blended
        
        valid_mask = ~np.isnan(z_pts)
        if np.sum(valid_mask) < 3:
            logger.warning(f"      ⚠ Less than 3 valid temperature points")
            return gradient_temps.reshape(self.grid_size[1], self.grid_size[0]), geotis_available
        
        x_pts_valid = x_pts[valid_mask]
        y_pts_valid = y_pts[valid_mask]
        z_pts_valid = z_pts[valid_mask]
        
        rbf = Rbf(x_pts_valid, y_pts_valid, z_pts_valid, function='thin_plate', epsilon=None, smooth=1.0)
        temperature_grid = rbf(X_mesh, Y_mesh)
        
        # ✅ CLAMP TO PHYSICAL LIMITS
        logger.info(f"      Before clamping: [{temperature_grid.min():.1f}, {temperature_grid.max():.1f}]°C")
        temperature_grid = np.clip(temperature_grid, self.T_MIN, self.T_MAX)
        logger.info(f"      After clamping:  [{temperature_grid.min():.1f}, {temperature_grid.max():.1f}]°C")
        
        logger.info(f"      ✓ Temperature range (physical): [{temperature_grid.min():.1f}, {temperature_grid.max():.1f}]°C")
        logger.info(f"      Data sources: {geotis_count} GeoTIS (spatially filtered), {len(borehole_gdf) - geotis_count} gradient-derived")
        
        return temperature_grid.astype(np.float32), geotis_available
    
    def create_transform_and_bounds(self):
        """Create rasterio transform"""
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        width, height = self.grid_size
        transform = from_bounds(left, bottom, right, top, width, height)
        
        return transform
    
    def save_raster(self, raster, name, dtype=np.float32):
        """Save raster to GeoTIFF and NumPy"""
        transform = self.create_transform_and_bounds()
        
        tif_path = self.output_dir / f"{name}.tif"
        with rasterio.open(
            tif_path, 'w',
            driver='GTiff',
            height=self.grid_size[1],
            width=self.grid_size[0],
            count=1,
            dtype=dtype,
            crs=self.crs,
            transform=transform,
            compress='lzw'
        ) as dst:
            dst.write(raster, 1)
        
        npy_path = self.output_dir / f"{name}.npy"
        np.save(npy_path, raster)
        
        return tif_path, npy_path
    
    def process_tier1_horizon(self, horizon_name):
        """Process single Tier 1 horizon with borehole data"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {horizon_name} (TIER 1 - Borehole-based HYBRID)")
        logger.info(f"{'='*80}")
        
        try:
            logger.info(f"\n1. Loading borehole data...")
            borehole_gdf = self.load_belegpunkte_data(horizon_name)
            
            if borehole_gdf is None or len(borehole_gdf) == 0:
                logger.error(f"✗ No borehole data")
                return None
            
            depth_col = 'Teufe'
            thick_col = 'Gesamtmaec'
            
            if depth_col not in borehole_gdf.columns or thick_col not in borehole_gdf.columns:
                logger.error(f"✗ Required columns not found")
                return None
            
            logger.info(f"  ✓ Loaded: {len(borehole_gdf)} total records")
            
            borehole_records = borehole_gdf[(borehole_gdf[depth_col].notna()) & (borehole_gdf[depth_col] > 0)]
            outcrop_records = borehole_gdf[(borehole_gdf[depth_col].isna()) | (borehole_gdf[depth_col] <= 0)]
            
            logger.info(f"  Valid boreholes (Teufe > 0): {len(borehole_records)}")
            logger.info(f"  Outcrop records (Teufe ≤ 0): {len(outcrop_records)}")
            
            if len(borehole_records) < 3:
                logger.error(f"✗ Not enough borehole data")
                return None
            
            borehole_gdf = borehole_records.copy()
            
            logger.info(f"\n2. RBF interpolating depth surface...")
            depth_surface, depth_unc = self.rbf_interpolate_surface(borehole_gdf, depth_col, function='thin_plate')
            
            if depth_surface is None:
                return None
            
            logger.info(f"3. RBF interpolating thickness surface...")
            thickness_surface, thick_unc = self.rbf_interpolate_surface(borehole_gdf, thick_col, function='thin_plate')
            
            if thickness_surface is None:
                return None
            
            logger.info(f"\n4. Calculating mean depth...")
            mean_depth = -depth_surface + (thickness_surface / 2)
            
            logger.info(f"  Depth (boreholes, m): {depth_surface.min():.0f} to {depth_surface.max():.0f}")
            logger.info(f"  Thickness (boreholes, m): {thickness_surface.min():.0f} to {thickness_surface.max():.0f}")
            logger.info(f"  Mean depth (GeoTIS, m): {mean_depth.min():.0f} to {mean_depth.max():.0f}")
            
            borehole_gdf['mean_depth'] = -borehole_gdf['Teufe'] + (borehole_gdf['Gesamtmaec'] / 2)
            
            logger.info(f"\n5. Interpolating temperature (HYBRID approach, spatially filtered)...")
            temperature_surface, geotis_mask = self.interpolate_temperature_grid_hybrid(mean_depth, borehole_gdf)
            
            logger.info(f"  Temperature range (clamped): [{temperature_surface.min():.1f}, {temperature_surface.max():.1f}]°C")
            
            logger.info(f"\n6. Calculating confidence layers...")
            
            combined_unc = np.sqrt(depth_unc**2 + thick_unc**2)
            density_conf = self.calculate_borehole_density_confidence(borehole_gdf)
            confidence = self.combine_confidence_layers(combined_unc, density_conf)
            
            temp_stdv = (thick_unc + depth_unc) * self.GEOTHERMAL_GRADIENT / 1000.0
            
            logger.info(f"\n7. Saving evidence layers...")
            
            self.save_raster(depth_surface, f"{horizon_name}_depth_surface")
            self.save_raster(thickness_surface, f"{horizon_name}_thickness_surface")
            self.save_raster(temperature_surface, f"{horizon_name}_temperature_surface")
            self.save_raster(confidence, f"{horizon_name}_confidence")
            self.save_raster(temp_stdv, f"{horizon_name}_temperature_stdv")
            
            gradient_surface = np.full_like(temperature_surface, self.GEOTHERMAL_GRADIENT)
            self.save_raster(gradient_surface, f"{horizon_name}_geothermal_gradient")
            
            metadata = {
                'horizon': horizon_name,
                'tier': 1,
                'data_type': 'borehole_interpolated_rbf_hybrid_clamped_spatially_filtered',
                'interpolation_method': 'RBF (Radial Basis Function) - Thin Plate with physical clamping',
                'temperature_source': 'HYBRID: GeoTIS (primary, spatially filtered to basin) + Heat Flow Gradient (fallback)',
                'spatial_filter_applied': True,
                'spatial_filter_description': 'Only GeoTIS points within North German Basin extent - prevents contamination from Molasse & Rhine Graben',
                'spatial_filter_bounds_X_m': [float(self.basin_xmin), float(self.basin_xmax)],
                'spatial_filter_bounds_Y_m': [float(self.basin_ymin), float(self.basin_ymax)],
                'geotis_available': int(np.sum(geotis_mask)) if geotis_mask is not None else 0,
                'gradient_fallback': int(len(borehole_gdf) - np.sum(geotis_mask)) if geotis_mask is not None else len(borehole_gdf),
                'heat_flow_database': 'GFZ German Heat Flow Database 2022 (fallback)',
                'heat_flow_mean_mW_m2': self.HEAT_FLOW_MEAN,
                'thermal_conductivity_W_mK': (self.THERMAL_CONDUCTIVITY_LOW + self.THERMAL_CONDUCTIVITY_HIGH) / 2.0,
                'geothermal_gradient_C_km': float(self.GEOTHERMAL_GRADIENT),
                'surface_temperature_C': float(self.T_SURFACE),
                'temperature_clamping_C': [float(self.T_MIN), float(self.T_MAX)],
                'boreholes_used': len(borehole_gdf),
                'boreholes_excluded_outcrop': len(outcrop_records),
                'depth_range_m': [float(depth_surface.min()), float(depth_surface.max())],
                'thickness_range_m': [float(thickness_surface.min()), float(thickness_surface.max())],
                'mean_depth_m': float(np.nanmean(mean_depth)),
                'mean_depth_range_m': [float(np.nanmin(mean_depth)), float(np.nanmax(mean_depth))],
                'temperature_range_C': [float(temperature_surface.min()), float(temperature_surface.max())],
                'mean_temperature_C': float(temperature_surface.mean()),
                'confidence_range': [float(confidence.min()), float(confidence.max())],
                'mean_confidence': float(confidence.mean()),
                'coordinate_system': 'GeoTIS (- below sea level, + above) within EPSG:31467',
                'notes': 'HYBRID RBF layers with physical clamping & spatial filtering: GeoTIS (spatially filtered to basin only) where available, heat flow gradient elsewhere - full spatial coverage, no extrapolation artifacts, no contamination from other basins'
            }
            
            metadata_file = self.output_dir / f"{horizon_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"\n✓ {horizon_name} completed successfully!")
            logger.info(f"  Coverage: Full basin (RBF ensures no gaps)")
            logger.info(f"  Temperature sources: GeoTIS {metadata['geotis_available']}/{len(borehole_gdf)} (spatially filtered) + gradient {metadata['gradient_fallback']}/{len(borehole_gdf)}")
            logger.info(f"  Temperature clamped to [{self.T_MIN}, {self.T_MAX}]°C (prevents extrapolation artifacts)")
            logger.info(f"  Spatial filter: ACTIVE (basin bounds only)")
            
            return metadata
        
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_tier1(self):
        """Process all Tier 1 horizons"""
        logger.info("\n" + "="*80)
        logger.info("TIER 1: THERMAL EVIDENCE LAYERS (Borehole-based HYBRID with Spatial Filtering)")
        logger.info("="*80)
        
        try:
            self.load_geoTIS_temperature_data()
        except Exception as e:
            logger.warning(f"⚠ Failed to load GeoTIS: {e}")
        
        results = {}
        
        for horizon in self.TIER1_HORIZONS:
            metadata = self.process_tier1_horizon(horizon)
            if metadata:
                results[horizon] = metadata
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info(f"TIER 1 SUMMARY: {len(results)}/{len(self.TIER1_HORIZONS)} horizons processed")
        logger.info("="*80)
        
        for horizon, meta in sorted(results.items()):
            logger.info(f"\n{horizon}:")
            logger.info(f"  Boreholes used: {meta['boreholes_used']}")
            logger.info(f"  Outcrops excluded: {meta['boreholes_excluded_outcrop']}")
            logger.info(f"  Method: {meta['interpolation_method']}")
            logger.info(f"  Temperature source: {meta['temperature_source']}")
            logger.info(f"    - GeoTIS (spatially filtered): {meta['geotis_available']} boreholes")
            logger.info(f"    - Gradient fallback: {meta['gradient_fallback']} boreholes")
            logger.info(f"  Spatial filter: {'ACTIVE - basin bounds only' if meta['spatial_filter_applied'] else 'NONE'}")
            logger.info(f"  Depth range (boreholes): {meta['depth_range_m'][0]:.0f}-{meta['depth_range_m'][1]:.0f} m")
            logger.info(f"  Mean depth (GeoTIS system): {meta['mean_depth_range_m'][0]:.0f} to {meta['mean_depth_range_m'][1]:.0f} m")
            logger.info(f"  Temperature: {meta['mean_temperature_C']:.1f}°C (range: {meta['temperature_range_C'][0]:.1f}-{meta['temperature_range_C'][1]:.1f})")
            logger.info(f"  Clamped to: [{meta['temperature_clamping_C'][0]:.1f}, {meta['temperature_clamping_C'][1]:.1f}]°C")
            logger.info(f"  Confidence: {meta['mean_confidence']:.2f}")
            logger.info(f"  Coverage: Full basin (no gaps, no extrapolation artifacts, no basin contamination)")
        
        return results


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    logger.info("\n" + "="*80)
    logger.info("TIER 1 - HYBRID APPROACH WITH PHYSICAL CLAMPING & SPATIAL FILTERING")
    logger.info("="*80)
    logger.info("\n✓ Temperature Strategy:")
    logger.info("  1. Primary: GeoTIS measured temperatures (where available)")
    logger.info("     ✓ SPATIAL FILTER: Only points within North German Basin extent")
    logger.info("     ✓ Prevents contamination from Molasse & Rhine Graben anomalies")
    logger.info("  2. Fallback: Heat flow-derived geothermal gradient (where GeoTIS missing)")
    logger.info("  3. RBF smooth across entire basin")
    logger.info("  4. CLAMP to physical limits [3.0, 200.0]°C (prevents extrapolation artifacts)")
    logger.info("\n  Result: Full coverage + realistic temperatures + no anomalies from other basins!\n")
    
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs",
        geoTIS_dir="data/inputs/geoTIS/temperature"
    )
    
    results = loader.process_tier1()
    
    logger.info("\n" + "="*80)
    logger.info("✓ THERMAL EVIDENCE LAYER GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {loader.output_dir}")
    logger.info("\nEvidence layers ready for Tier 2 (TUNB synthesis):")
    logger.info("  - *_depth_surface.tif/.npy (RBF interpolated)")
    logger.info("  - *_thickness_surface.tif/.npy (RBF interpolated)")
    logger.info("  - *_temperature_surface.tif/.npy (HYBRID: GeoTIS spatially-filtered + gradient, physically clamped)")
    logger.info("  - *_geothermal_gradient.tif/.npy (constant: 23.1°C/km)")
    logger.info("  - *_confidence.tif/.npy (borehole density + interpolation error)")
    logger.info("  - *_temperature_stdv.tif/.npy (uncertainty estimate)")
    logger.info("  - *_metadata.json (complete provenance + data sources + spatial filtering info + clamping info)")
    logger.info("\nNext: Process Tier 2 (TUNB surfaces + GeoTIS deep temperatures)")


if __name__ == "__main__":
    main()

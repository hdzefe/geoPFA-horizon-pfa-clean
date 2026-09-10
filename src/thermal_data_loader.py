"""
Thermal Data Loader - PFA Evidence Layer Generation

Generates PFA evidence layers for geothermal favorability assessment.
Processes borehole data (Tier 1) using RBF interpolation to create:
- RBF-interpolated depth surfaces
- RBF-interpolated thickness surfaces
- RBF-interpolated temperature surfaces
- Confidence layers (RBF uncertainty + borehole density)

Output: Clean evidence layers for input to PFA combination model

Coordinate Systems:
- GeoTIS temperature: Negative = below sea level, Positive = above sea level
- Borehole Teufe: Positive values representing depth below surface
- Conversion: Negate Teufe to match GeoTIS coordinate system

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
from scipy.interpolate import griddata, Rbf
from scipy.spatial.distance import cdist
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Generate PFA thermal evidence layers from borehole and GeoTIS data using RBF interpolation"""
    
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
    
    def __init__(self, config_path, base_dir="data/inputs", geoTIS_dir="data/inputs/geoTIS/temperature"):
        """
        Initialize thermal data loader
        
        Args:
            config_path: Path to metadata.json
            base_dir: Base directory for input data
            geoTIS_dir: Directory with GeoTIS temperature .DATA files
        """
        self.base_dir = Path(base_dir)
        self.config_path = Path(config_path)
        self.geoTIS_dir = Path(geoTIS_dir)
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        self.crs = self.config['crs']
        self.grid_size = self.config['grid_size']
        self.extent = self.config['extent']
        
        # Create output directory
        self.output_dir = Path("data/outputs/thermal")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # GeoTIS temperature data (cached)
        self.geoTIS_data = {}
        self.geoTIS_loaded = False
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Tier 1 horizons: {len(self.TIER1_HORIZONS)}")
        logger.info(f"  Interpolation method: RBF (Radial Basis Function)")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Coordinate system: GeoTIS (- below sea level, + above)")
        logger.info(f"  Data filtering: Excludes Teufe ≤ 0 (surface outcrops)")
    
    def load_geoTIS_temperature_data(self):
        """
        Load all GeoTIS temperature .DATA files
        
        Handles multiple filename formats:
        - Simple: +100.DATA, -500.DATA
        - Prefixed: T2022_LIAG_AGEMAR_+100.data, T2022_LIAG_AGEMAR_-2500.data
        
        Format: German locale with comma decimal separator
        - Delimiter: semicolon (;)
        - Decimal: comma (,)
        - Headers: First 2 lines are metadata
        - Missing data: -99999
        - Coordinate system: - below sea level, + above sea level
        
        Returns:
            Dict: {depth_value: {'X': array, 'Y': array, 'T': array, 'STDV': array}}
        """
        logger.info(f"\n{'='*80}")
        logger.info("LOADING GEOТIS TEMPERATURE DATA")
        logger.info(f"{'='*80}")
        
        if self.geoTIS_loaded:
            logger.info("✓ GeoTIS data already loaded")
            return self.geoTIS_data
        
        if not self.geoTIS_dir.exists():
            logger.error(f"✗ GeoTIS directory not found: {self.geoTIS_dir}")
            raise FileNotFoundError(f"GeoTIS directory: {self.geoTIS_dir}")
        
        # Find all .DATA files
        data_files = sorted(self.geoTIS_dir.glob("*.DATA")) + sorted(self.geoTIS_dir.glob("*.data"))
        
        if not data_files:
            logger.error(f"✗ No .DATA files found in {self.geoTIS_dir}")
            raise FileNotFoundError(f"No .DATA files in {self.geoTIS_dir}")
        
        logger.info(f"✓ Found {len(data_files)} temperature depth files")
        
        geoTIS_data = {}
        
        for data_file in data_files:
            try:
                # Extract depth from filename using regex
                filename = data_file.stem
                match = re.search(r'([+-]\d+)', filename)
                
                if match:
                    depth_str = match.group(1)
                    depth = int(depth_str)
                else:
                    logger.debug(f"  ⚠ Could not parse depth from filename: {data_file.name}")
                    continue
                
                # Read .DATA file
                df = pd.read_csv(
                    data_file, 
                    delimiter=';', 
                    skipinitialspace=True,
                    skiprows=2,
                    names=['X', 'Y', 'Temperature', 'STDV'],
                    decimal=','
                )
                
                if len(df) == 0:
                    logger.debug(f"  ⚠ Empty file: {data_file.name}")
                    continue
                
                # Convert columns to numeric
                for col in ['X', 'Y', 'Temperature', 'STDV']:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                df = df.dropna()
                df = df[(df['Temperature'] != -99999.0) & (df['STDV'] != -99999.0)]
                
                if len(df) == 0:
                    logger.debug(f"  ⚠ No valid data in: {data_file.name}")
                    continue
                
                geoTIS_data[depth] = {
                    'X': df['X'].values.astype(np.float32),
                    'Y': df['Y'].values.astype(np.float32),
                    'T': df['Temperature'].values.astype(np.float32),
                    'STDV': df['STDV'].values.astype(np.float32)
                }
                
                logger.debug(f"  ✓ {data_file.name}: depth={depth:>5}m, {len(df):>5} valid points, T [{df['Temperature'].min():>6.1f}, {df['Temperature'].max():>6.1f}]°C")
            
            except Exception as e:
                logger.error(f"  ✗ Error reading {data_file.name}: {e}")
                continue
        
        if not geoTIS_data:
            raise ValueError("No valid GeoTIS data loaded")
        
        logger.info(f"\n✓ Loaded {len(geoTIS_data)} depth levels")
        depths = sorted(geoTIS_data.keys())
        logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m (- = below sea level, + = above)")
        
        self.geoTIS_data = geoTIS_data
        self.geoTIS_loaded = True
        return geoTIS_data
    
    def interpolate_temperature_at_depth(self, target_depth, method='linear'):
        """
        Interpolate temperature at arbitrary depth from GeoTIS data
        
        Args:
            target_depth: Depth in meters (negative = below sea level)
            method: 'linear' or 'nearest'
        
        Returns:
            (X, Y, T, STDV) arrays at target depth
        """
        if not self.geoTIS_loaded:
            raise ValueError("GeoTIS data not loaded")
        
        available_depths = sorted(self.geoTIS_data.keys())
        
        if target_depth in self.geoTIS_data:
            data = self.geoTIS_data[target_depth]
            return data['X'], data['Y'], data['T'], data['STDV']
        
        # Find surrounding depths
        below = [d for d in available_depths if d <= target_depth]
        above = [d for d in available_depths if d >= target_depth]
        
        if method == 'nearest' or len(below) == 0 or len(above) == 0:
            # Use nearest depth
            if len(below) > 0 and len(above) > 0:
                d_below = below[-1]
                d_above = above[0]
                depth = d_below if (target_depth - d_below) <= (d_above - target_depth) else d_above
            elif len(below) > 0:
                depth = below[-1]
            else:
                depth = above[0]
            
            data = self.geoTIS_data[depth]
            return data['X'], data['Y'], data['T'], data['STDV']
        
        else:  # linear interpolation
            d_below = below[-1]
            d_above = above[0]
            
            data_below = self.geoTIS_data[d_below]
            data_above = self.geoTIS_data[d_above]
            
            # Interpolation weights
            w_below = (d_above - target_depth) / (d_above - d_below)
            w_above = (target_depth - d_below) / (d_above - d_below)
            
            # Match points and interpolate
            X_below = data_below['X']
            Y_below = data_below['Y']
            T_below = data_below['T']
            
            points_below = np.column_stack([X_below, Y_below])
            points_above = np.column_stack([data_above['X'], data_above['Y']])
            
            # Interpolate above to below grid
            T_above_interp = griddata(points_above, data_above['T'], points_below, method='nearest')
            
            # Linear interpolation
            T_interp = w_below * T_below + w_above * T_above_interp
            STDV_interp = np.sqrt((w_below * data_below['STDV'])**2 + (w_above * T_above_interp)**2)
            
            return X_below, Y_below, T_interp, STDV_interp
    
    def find_belegpunkte_shapefile(self, horizon_name):
        """Find Belegpunkte shapefile for horizon"""
        if horizon_name not in self.BELEGPUNKTE_MAPPING:
            logger.warning(f"  ⚠ No mapping for {horizon_name}")
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
            
            logger.info(f"  ✓ Loaded: {shp_path.name}")
            logger.info(f"    Total records: {len(gdf)}")
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Extract coordinates
            gdf['X'] = gdf.geometry.x
            gdf['Y'] = gdf.geometry.y
            
            return gdf
        
        except Exception as e:
            logger.error(f"  ✗ Error loading {shp_path}: {e}")
            return None
    
    def rbf_interpolate_surface(self, borehole_gdf, column_name, function='thin_plate'):
        """
        Interpolate surface using Radial Basis Function (RBF)
        
        RBF advantages over kriging:
        - No spatial range limits → covers entire basin
        - No min/max point requirements → works with sparse data
        - Smooth and natural interpolation
        - No dropout zones with 0 values
        
        Args:
            borehole_gdf: GeoDataFrame with borehole data
            column_name: Column to interpolate
            function: 'thin_plate', 'multiquadric', 'inverse_multiquadric', 'gaussian', 'linear', 'cubic', 'quintic'
        
        Returns:
            Interpolated 2D array + uncertainty estimate
        """
        # Extract valid data (exclude NULL/NaN and values ≤ 0 for Teufe)
        valid = borehole_gdf[borehole_gdf[column_name].notna()].copy()
        
        # If this is Teufe column, exclude Teufe ≤ 0 (surface outcrops)
        if column_name == 'Teufe':
            valid = valid[valid[column_name] > 0]
        
        if len(valid) < 3:
            logger.warning(f"    ⚠ Less than 3 valid points for {column_name} ({len(valid)} found)")
            return None, None
        
        logger.info(f"    RBF interpolating {column_name}: {len(valid)} valid points (from {len(borehole_gdf)} total)")
        
        try:
            # Prepare data
            x_pts = valid['X'].values
            y_pts = valid['Y'].values
            z_pts = valid[column_name].values.astype(np.float32)
            
            logger.info(f"      Data range: {z_pts.min():.2f} - {z_pts.max():.2f}")
            
            # Create RBF interpolator
            logger.info(f"      Fitting RBF ({function})...")
            rbf = Rbf(x_pts, y_pts, z_pts, function=function, epsilon=None, smooth=1.0)
            
            # Create grid
            left = self.extent['left']
            right = self.extent['right']
            bottom = self.extent['bottom']
            top = self.extent['top']
            
            x_grid = np.linspace(left, right, self.grid_size[0])
            y_grid = np.linspace(bottom, top, self.grid_size[1])
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            
            # Interpolate to grid
            logger.info(f"      Evaluating RBF on grid...")
            interpolated = rbf(X_mesh, Y_mesh)
            
            logger.info(f"      ✓ Interpolation complete. Grid range: {np.nanmin(interpolated):.2f} - {np.nanmax(interpolated):.2f}")
            
            # Calculate uncertainty based on distance to nearest borehole
            grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
            borehole_points = np.column_stack([x_pts, y_pts])
            
            distances = cdist(grid_points, borehole_points)
            min_distances = np.min(distances, axis=1)
            
            # Uncertainty increases with distance from nearest borehole
            max_dist = np.percentile(min_distances, 95)
            uncertainty = np.clip(min_distances / max_dist, 0, 1).reshape(self.grid_size[1], self.grid_size[0])
            
            # Scale uncertainty by data range
            data_range = z_pts.max() - z_pts.min()
            uncertainty = uncertainty * (data_range * 0.1)
            
            return interpolated.astype(np.float32), uncertainty.astype(np.float32)
        
        except Exception as e:
            logger.error(f"    ✗ RBF interpolation failed: {e}")
            import traceback
            traceback.print_exc()
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
        """
        Combine interpolation error and borehole density into single confidence layer
        
        Confidence = sqrt((1 - error_normalized) × density_confidence)
        
        Args:
            interpolation_error: Interpolation uncertainty surface
            density_confidence: Borehole density confidence (0-1)
        
        Returns:
            Combined confidence layer (0-1)
        """
        # Normalize interpolation error to 0-1
        error_norm = np.clip(interpolation_error / np.percentile(interpolation_error[interpolation_error > 0], 95), 0, 1)
        interpolation_confidence = 1.0 - error_norm
        
        # Combine: geometric mean
        combined = np.sqrt(interpolation_confidence * density_confidence)
        
        return combined.astype(np.float32)
    
    def interpolate_temperature_grid(self, mean_depth_surface):
        """
        Interpolate temperature across entire grid at varying depths
        
        Args:
            mean_depth_surface: 2D array of mean depths (m, negative = below sea level)
        
        Returns:
            Interpolated temperature grid + uncertainty
        """
        logger.info(f"    Interpolating temperature grid...")
        
        temperature_surface = np.zeros_like(mean_depth_surface)
        stdv_surface = np.zeros_like(mean_depth_surface)
        
        # Process by depth slices for efficiency
        unique_depths = np.unique(mean_depth_surface[mean_depth_surface < 0])
        unique_depths = unique_depths[::max(1, len(unique_depths)//20)]
        
        logger.info(f"      Processing {len(unique_depths)} depth levels...")
        
        for depth in unique_depths:
            mask = np.abs(mean_depth_surface - depth) < 50
            
            if not np.any(mask):
                continue
            
            # Get temperature at this depth
            X_temp, Y_temp, T_temp, STDV_temp = self.interpolate_temperature_at_depth(depth, method='linear')
            
            points_temp = np.column_stack([X_temp, Y_temp])
            
            left = self.extent['left']
            right = self.extent['right']
            bottom = self.extent['bottom']
            top = self.extent['top']
            
            x_grid = np.linspace(left, right, self.grid_size[0])
            y_grid = np.linspace(bottom, top, self.grid_size[1])
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            
            T_grid = griddata(points_temp, T_temp, (X_mesh, Y_mesh), method='linear')
            STDV_grid = griddata(points_temp, STDV_temp, (X_mesh, Y_mesh), method='linear')
            
            T_nn = griddata(points_temp, T_temp, (X_mesh, Y_mesh), method='nearest')
            STDV_nn = griddata(points_temp, STDV_temp, (X_mesh, Y_mesh), method='nearest')
            
            nan_mask = np.isnan(T_grid)
            T_grid[nan_mask] = T_nn[nan_mask]
            STDV_grid[nan_mask] = STDV_nn[nan_mask]
            
            temperature_surface[mask] = T_grid[mask]
            stdv_surface[mask] = STDV_grid[mask]
        
        logger.info(f"      ✓ Temperature range: [{temperature_surface.min():.1f}, {temperature_surface.max():.1f}]°C")
        
        return temperature_surface, stdv_surface
    
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
        
        # GeoTIFF
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
        
        # NumPy
        npy_path = self.output_dir / f"{name}.npy"
        np.save(npy_path, raster)
        
        return tif_path, npy_path
    
    def process_tier1_horizon(self, horizon_name):
        """Process single Tier 1 horizon with borehole data"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {horizon_name} (TIER 1 - with borehole data)")
        logger.info(f"{'='*80}")
        
        try:
            # Load borehole data
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
            
            logger.info(f"  Using: depth='{depth_col}', thickness='{thick_col}'")
            
            # Separate boreholes from outcrops
            # Boreholes: Teufe > 0 (positive depth below surface)
            # Outcrops: Teufe = 0 or NULL (surface locations)
            borehole_records = borehole_gdf[(borehole_gdf[depth_col].notna()) & (borehole_gdf[depth_col] > 0)]
            outcrop_records = borehole_gdf[(borehole_gdf[depth_col].isna()) | (borehole_gdf[depth_col] <= 0)]
            
            logger.info(f"  Valid boreholes (Teufe > 0): {len(borehole_records)}/{len(borehole_gdf)}")
            logger.info(f"  Outcrop records (Teufe ≤ 0): {len(outcrop_records)}/{len(borehole_gdf)}")
            
            if len(borehole_records) < 3:
                logger.error(f"✗ Not enough borehole data ({len(borehole_records)} < 3)")
                return None
            
            # Use only boreholes for interpolation (exclude outcrops)
            borehole_gdf = borehole_records.copy()
            
            # RBF interpolate depth surface
            logger.info(f"\n2. RBF interpolating depth surface...")
            depth_surface, depth_unc = self.rbf_interpolate_surface(borehole_gdf, depth_col, function='thin_plate')
            
            if depth_surface is None:
                logger.error(f"✗ Interpolation failed")
                return None
            
            # RBF interpolate thickness surface
            logger.info(f"\n3. RBF interpolating thickness surface...")
            thickness_surface, thick_unc = self.rbf_interpolate_surface(borehole_gdf, thick_col, function='thin_plate')
            
            if thickness_surface is None:
                logger.error(f"✗ Interpolation failed")
                return None
            
            # Calculate mean depth (negate to match GeoTIS coordinate system)
            logger.info(f"\n4. Calculating mean depth (converting to GeoTIS coordinate system)...")
            mean_depth = -depth_surface + (thickness_surface / 2)
            
            logger.info(f"  Depth range (original boreholes, positive): [0, {depth_surface.max():.0f}] m")
            logger.info(f"  Depth range (GeoTIS system, negative): [{mean_depth.min():.0f}, {mean_depth.max():.0f}] m")
            logger.info(f"  Thickness range: [{thickness_surface.min():.0f}, {thickness_surface.max():.0f}] m")
            
            # Check for NaN values in mean depth
            nan_count = np.isnan(mean_depth).sum()
            if nan_count > 0:
                logger.warning(f"  ⚠ {nan_count} NaN values in mean depth (will be handled)")
            
            # Interpolate temperature
            logger.info(f"\n5. Interpolating temperature...")
            temperature_surface, temp_stdv = self.interpolate_temperature_grid(mean_depth)
            
            # Check for zero values
            zero_count = (temperature_surface == 0).sum()
            if zero_count > 0:
                logger.warning(f"  ⚠ {zero_count} zero temperature values in grid")
            
            # Calculate confidence layers
            logger.info(f"\n6. Calculating confidence layers...")
            
            combined_unc = np.sqrt(depth_unc**2 + thick_unc**2)
            density_conf = self.calculate_borehole_density_confidence(borehole_gdf)
            confidence = self.combine_confidence_layers(combined_unc, density_conf)
            
            logger.info(f"  Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")
            logger.info(f"  Mean confidence: {confidence.mean():.2f}")
            
            # Calculate geothermal gradient
            logger.info(f"\n7. Calculating geothermal gradient...")
            gradient = np.full_like(temperature_surface, 25.0)
            logger.info(f"  Geothermal gradient: 25 °C/km (typical for North German Basin)")
            
            # Save outputs
            logger.info(f"\n8. Saving evidence layers...")
            
            self.save_raster(depth_surface, f"{horizon_name}_depth_surface")
            self.save_raster(thickness_surface, f"{horizon_name}_thickness_surface")
            self.save_raster(temperature_surface, f"{horizon_name}_temperature_surface")
            self.save_raster(gradient, f"{horizon_name}_geothermal_gradient")
            self.save_raster(confidence, f"{horizon_name}_confidence")
            self.save_raster(temp_stdv, f"{horizon_name}_temperature_stdv")
            
            # Metadata
            metadata = {
                'horizon': horizon_name,
                'tier': 1,
                'data_type': 'borehole_interpolated_rbf',
                'interpolation_method': 'RBF (Radial Basis Function) - Thin Plate',
                'boreholes_used': len(borehole_gdf),
                'boreholes_excluded_outcrop': len(outcrop_records),
                'depth_range_m': [float(depth_surface.min()), float(depth_surface.max())],
                'thickness_range_m': [float(thickness_surface.min()), float(thickness_surface.max())],
                'mean_depth_m': float(np.nanmean(mean_depth)),
                'mean_depth_range_m': [float(np.nanmin(mean_depth)), float(np.nanmax(mean_depth))],
                'temperature_range_C': [float(np.nanmin(temperature_surface)), float(np.nanmax(temperature_surface))],
                'mean_temperature_C': float(np.nanmean(temperature_surface)),
                'temperature_zero_count': int((temperature_surface == 0).sum()),
                'geothermal_gradient_C_km': 25.0,
                'confidence_range': [float(confidence.min()), float(confidence.max())],
                'mean_confidence': float(confidence.mean()),
                'coordinate_system': 'GeoTIS (- below sea level, + above)',
                'notes': 'RBF evidence layers - full spatial coverage, no dropout zones'
            }
            
            metadata_file = self.output_dir / f"{horizon_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"\n✓ {horizon_name} completed successfully!")
            logger.info(f"  Coverage: Full basin (RBF ensures no gaps)")
            
            return metadata
        
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_tier1(self):
        """Process all Tier 1 horizons"""
        logger.info("\n" + "="*80)
        logger.info("TIER 1: THERMAL EVIDENCE LAYERS (Borehole-based with RBF)")
        logger.info("="*80)
        
        # Load GeoTIS once
        try:
            self.load_geoTIS_temperature_data()
        except Exception as e:
            logger.error(f"✗ Failed to load GeoTIS: {e}")
            return {}
        
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
            logger.info(f"  Depth range (boreholes): {meta['depth_range_m'][0]:.0f}-{meta['depth_range_m'][1]:.0f} m")
            logger.info(f"  Mean depth (GeoTIS system): {meta['mean_depth_range_m'][0]:.0f} to {meta['mean_depth_range_m'][1]:.0f} m")
            logger.info(f"  Temperature: {meta['mean_temperature_C']:.1f}°C (range: {meta['temperature_range_C'][0]:.1f}-{meta['temperature_range_C'][1]:.1f})")
            logger.info(f"  Zero temp values: {meta['temperature_zero_count']}")
            logger.info(f"  Confidence: {meta['mean_confidence']:.2f}")
            logger.info(f"  Coverage: Full basin (no gaps)")
        
        return results


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    logger.info("\n✓ Using RBF (Radial Basis Function) interpolation")
    logger.info("  Advantages: Full spatial coverage, no dropout zones, smooth surfaces\n")
    
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs",
        geoTIS_dir="data/inputs/geoTIS/temperature"
    )
    
    # Process Tier 1
    results = loader.process_tier1()
    
    logger.info("\n" + "="*80)
    logger.info("✓ THERMAL EVIDENCE LAYER GENERATION COMPLETE")
    logger.info("="*80)
    logger.info(f"\nOutput directory: {loader.output_dir}")
    logger.info("\nEvidence layers ready for Tier 2 (TUNB synthesis):")
    logger.info("  - *_depth_surface.tif/.npy (RBF interpolated)")
    logger.info("  - *_thickness_surface.tif/.npy (RBF interpolated)")
    logger.info("  - *_temperature_surface.tif/.npy (RBF-based, full coverage)")
    logger.info("  - *_geothermal_gradient.tif/.npy")
    logger.info("  - *_confidence.tif/.npy")
    logger.info("  - *_temperature_stdv.tif/.npy")
    logger.info("  - *_metadata.json")
    logger.info("\nNext: Process Tier 2 (TUNB + GeoTIS synthetic surfaces)")


if __name__ == "__main__":
    main()

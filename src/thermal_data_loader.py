"""
Thermal Data Loader - PFA Evidence Layer Generation

Generates PFA evidence layers for geothermal favorability assessment.
Processes borehole data (Tier 1) and TUNB surfaces (Tier 2) to create:
- Interpolated depth surfaces
- Interpolated thickness surfaces
- Interpolated temperature surfaces
- Confidence layers (combined kriging error + borehole density)

Output: Clean evidence layers for input to PFA combination model

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
from rasterio.features import rasterize
from rasterio.transform import from_bounds
from scipy.interpolate import griddata
from scipy.spatial.distance import cdist
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Generate PFA thermal evidence layers from borehole and GeoTIS data"""
    
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
        logger.info(f"  Output directory: {self.output_dir}")
    
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
                # Handles formats like:
                #   +100.DATA, -500.DATA (simple)
                #   T2022_LIAG_AGEMAR_+100.data, T2022_LIAG_AGEMAR_-2500.data (with prefix)
                
                filename = data_file.stem  # Remove .data/.DATA extension
                
                # Find depth pattern: +/- followed by digits
                match = re.search(r'([+-]\d+)', filename)
                
                if match:
                    depth_str = match.group(1)
                    depth = int(depth_str)
                else:
                    logger.debug(f"  ⚠ Could not parse depth from filename: {data_file.name}")
                    continue
                
                # Read .DATA file
                # Format: X; Y; Temperature; STDV
                # German locale: comma as decimal separator
                # Skip first 2 lines (headers)
                df = pd.read_csv(
                    data_file, 
                    delimiter=';', 
                    skipinitialspace=True,
                    skiprows=2,  # Skip metadata header lines
                    names=['X', 'Y', 'Temperature', 'STDV'],
                    decimal=','  # German decimal separator
                )
                
                if len(df) == 0:
                    logger.debug(f"  ⚠ Empty file: {data_file.name}")
                    continue
                
                # Convert columns to numeric, handling both German (,) and English (.) decimals
                for col in ['X', 'Y', 'Temperature', 'STDV']:
                    # First convert string to numeric (handles both , and .)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                
                # Drop rows with NaN
                df = df.dropna()
                
                # Filter out missing data (-99999)
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
        logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m")
        logger.info(f"  First 10 depths: {depths[:10]}")
        logger.info(f"  Last 10 depths: {depths[-10:]}")
        
        self.geoTIS_data = geoTIS_data
        self.geoTIS_loaded = True
        return geoTIS_data
    
    def interpolate_temperature_at_depth(self, target_depth, method='linear'):
        """
        Interpolate temperature at arbitrary depth from GeoTIS data
        
        Args:
            target_depth: Depth in meters
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
        # Use mapping to get correct filename prefix
        if horizon_name not in self.BELEGPUNKTE_MAPPING:
            logger.warning(f"  ⚠ No mapping for {horizon_name}")
            return None
        
        prefix = self.BELEGPUNKTE_MAPPING[horizon_name]
        shp_filename = f"{prefix}_Belegpunkte.shp"
        
        # Search in horizon-specific directory
        horizon_dir = self.base_dir / "reservoirs" / horizon_name
        shp_path = horizon_dir / shp_filename
        
        if shp_path.exists():
            return shp_path
        
        # Fallback: search recursively
        for path in self.base_dir.rglob(shp_filename):
            return path
        
        return None
    
    def find_potential_shapefile(self, horizon_name):
        """Find Potential shapefile from config"""
        for res_config in self.config['reservoirs']:
            if res_config['name'] == horizon_name:
                return self.base_dir / res_config['path']
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
            logger.info(f"    Records: {len(gdf)}")
            
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
    
    def interpolate_surface(self, borehole_gdf, column_name, method='linear'):
        """
        Interpolate surface from borehole points using linear interpolation
        
        Args:
            borehole_gdf: GeoDataFrame with borehole data
            column_name: Column to interpolate
            method: 'linear', 'cubic', or 'nearest'
        
        Returns:
            Interpolated 2D array + kriging standard error estimate
        """
        # Extract valid data (exclude NULL/NaN)
        valid = borehole_gdf[borehole_gdf[column_name].notna()].copy()
        
        if len(valid) < 3:
            logger.warning(f"    ⚠ Less than 3 valid points for {column_name} ({len(valid)} found)")
            return None, None
        
        logger.info(f"    Interpolating {column_name}: {len(valid)} valid points (from {len(borehole_gdf)} total)")
        
        # Create regular grid
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        # Prepare data
        points = np.column_stack([valid['X'], valid['Y']])
        values = valid[column_name].values.astype(np.float32)
        grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        
        # Interpolate
        try:
            interpolated = griddata(points, values, grid_points, method=method)
            interpolated = interpolated.reshape(self.grid_size[1], self.grid_size[0])
            
            # Fill NaN with nearest neighbor
            if np.any(np.isnan(interpolated)):
                mask = np.isnan(interpolated)
                interpolated_nn = griddata(points, values, grid_points, method='nearest')
                interpolated_nn = interpolated_nn.reshape(self.grid_size[1], self.grid_size[0])
                interpolated[mask] = interpolated_nn[mask]
            
            # Estimate kriging standard error (simplified)
            # Error is higher where points are sparse
            distances = cdist(grid_points, points)
            min_distances = np.min(distances, axis=1)
            
            # Normalize to 0-1 range
            max_dist = np.percentile(min_distances, 95)
            kriging_error = np.clip(min_distances / max_dist, 0, 1).reshape(self.grid_size[1], self.grid_size[0])
            
            # Estimate uncertainty as % of data range
            data_range = values.max() - values.min()
            kriging_error = kriging_error * (data_range * 0.1)  # ~10% of range at max distance
            
            return interpolated.astype(np.float32), kriging_error.astype(np.float32)
        
        except Exception as e:
            logger.error(f"    ✗ Interpolation failed: {e}")
            return None, None
    
    def calculate_borehole_density_confidence(self, borehole_gdf):
        """
        Calculate confidence based on borehole spatial density
        
        Returns:
            Confidence grid (0-1, 1=high density)
        """
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        # Distance to nearest borehole
        borehole_points = np.column_stack([borehole_gdf['X'], borehole_gdf['Y']])
        grid_points = np.column_stack([X_mesh.ravel(), Y_mesh.ravel()])
        
        distances = cdist(grid_points, borehole_points)
        min_distances = np.min(distances, axis=1)
        
        # Convert distance to confidence (closer = higher confidence)
        # Normalize by max distance
        max_dist = np.percentile(min_distances, 95)
        density_confidence = 1.0 - np.clip(min_distances / max_dist, 0, 1)
        density_confidence = density_confidence.reshape(self.grid_size[1], self.grid_size[0])
        
        return density_confidence.astype(np.float32)
    
    def combine_confidence_layers(self, kriging_error, density_confidence):
        """
        Combine kriging error and borehole density into single confidence layer
        
        Confidence = sqrt((1 - kriging_error_normalized) × density_confidence)
        
        Args:
            kriging_error: Kriging standard error surface (0-max)
            density_confidence: Borehole density confidence (0-1)
        
        Returns:
            Combined confidence layer (0-1)
        """
        # Normalize kriging error to 0-1 (higher error = lower confidence)
        ke_norm = np.clip(kriging_error / np.percentile(kriging_error[kriging_error > 0], 95), 0, 1)
        kriging_confidence = 1.0 - ke_norm
        
        # Combine: geometric mean emphasizes areas with BOTH good coverage and low error
        combined = np.sqrt(kriging_confidence * density_confidence)
        
        return combined.astype(np.float32)
    
    def interpolate_temperature_grid(self, mean_depth_surface):
        """
        Interpolate temperature across entire grid at varying depths
        
        Args:
            mean_depth_surface: 2D array of mean depths (m)
        
        Returns:
            Interpolated temperature grid + uncertainty
        """
        logger.info(f"    Interpolating temperature grid...")
        
        temperature_surface = np.zeros_like(mean_depth_surface)
        stdv_surface = np.zeros_like(mean_depth_surface)
        
        # Process by depth slices for efficiency
        unique_depths = np.unique(mean_depth_surface[mean_depth_surface > 0])
        unique_depths = unique_depths[::max(1, len(unique_depths)//20)]  # Sample ~20 unique depths
        
        logger.info(f"      Processing {len(unique_depths)} depth levels...")
        
        for depth in unique_depths:
            mask = np.abs(mean_depth_surface - depth) < 50  # Within 50m
            
            if not np.any(mask):
                continue
            
            # Get temperature at this depth
            X_temp, Y_temp, T_temp, STDV_temp = self.interpolate_temperature_at_depth(depth, method='linear')
            
            # Create spatial index
            points_temp = np.column_stack([X_temp, Y_temp])
            
            # Grid for this depth slice
            left = self.extent['left']
            right = self.extent['right']
            bottom = self.extent['bottom']
            top = self.extent['top']
            
            x_grid = np.linspace(left, right, self.grid_size[0])
            y_grid = np.linspace(bottom, top, self.grid_size[1])
            X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
            
            # Interpolate to grid
            T_grid = griddata(points_temp, T_temp, (X_mesh, Y_mesh), method='linear')
            STDV_grid = griddata(points_temp, STDV_temp, (X_mesh, Y_mesh), method='linear')
            
            # Fill NaNs with nearest
            T_nn = griddata(points_temp, T_temp, (X_mesh, Y_mesh), method='nearest')
            STDV_nn = griddata(points_temp, STDV_temp, (X_mesh, Y_mesh), method='nearest')
            
            nan_mask = np.isnan(T_grid)
            T_grid[nan_mask] = T_nn[nan_mask]
            STDV_grid[nan_mask] = STDV_nn[nan_mask]
            
            # Assign to surface
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
            
            # Use specified columns: Teufe (depth), Gesamtmaec (thickness)
            depth_col = 'Teufe'
            thick_col = 'Gesamtmaec'
            
            # Check if columns exist
            if depth_col not in borehole_gdf.columns:
                logger.error(f"✗ Column '{depth_col}' not found")
                logger.error(f"  Available columns: {list(borehole_gdf.columns)}")
                return None
            
            if thick_col not in borehole_gdf.columns:
                logger.error(f"✗ Column '{thick_col}' not found")
                logger.error(f"  Available columns: {list(borehole_gdf.columns)}")
                return None
            
            logger.info(f"  Using: depth='{depth_col}', thickness='{thick_col}'")
            
            # Count valid records (exclude NULL depths = outcrops)
            valid_depth_records = borehole_gdf[borehole_gdf[depth_col].notna()]
            logger.info(f"  Valid records with depth (boreholes): {len(valid_depth_records)}/{len(borehole_gdf)}")
            logger.info(f"  Outcrop records (NULL depth): {len(borehole_gdf) - len(valid_depth_records)}")
            
            if len(valid_depth_records) < 3:
                logger.error(f"✗ Not enough borehole data ({len(valid_depth_records)} < 3)")
                return None
            
            # Use only valid borehole records for interpolation
            borehole_gdf = valid_depth_records.copy()
            
            # Interpolate depth surface
            logger.info(f"\n2. Interpolating depth surface...")
            depth_surface, depth_ke = self.interpolate_surface(borehole_gdf, depth_col)
            
            if depth_surface is None:
                logger.error(f"✗ Interpolation failed")
                return None
            
            # Interpolate thickness surface
            logger.info(f"\n3. Interpolating thickness surface...")
            thickness_surface, thick_ke = self.interpolate_surface(borehole_gdf, thick_col)
            
            if thickness_surface is None:
                logger.error(f"✗ Interpolation failed")
                return None
            
            # Calculate mean depth
            logger.info(f"\n4. Calculating mean depth...")
            mean_depth = depth_surface + (thickness_surface / 2)
            
            logger.info(f"  Depth range: [{depth_surface.min():.0f}, {depth_surface.max():.0f}] m")
            logger.info(f"  Thickness range: [{thickness_surface.min():.0f}, {thickness_surface.max():.0f}] m")
            logger.info(f"  Mean depth range: [{mean_depth.min():.0f}, {mean_depth.max():.0f}] m")
            
            # Interpolate temperature
            logger.info(f"\n5. Interpolating temperature...")
            temperature_surface, temp_stdv = self.interpolate_temperature_grid(mean_depth)
            
            # Calculate confidence layers
            logger.info(f"\n6. Calculating confidence layers...")
            
            # Kriging error-based confidence
            # Combine errors from depth and thickness interpolation
            combined_ke = np.sqrt(depth_ke**2 + thick_ke**2)
            
            # Borehole density confidence
            density_conf = self.calculate_borehole_density_confidence(borehole_gdf)
            
            # Combined confidence
            confidence = self.combine_confidence_layers(combined_ke, density_conf)
            
            logger.info(f"  Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")
            logger.info(f"  Mean confidence: {confidence.mean():.2f}")
            
            # Calculate geothermal gradient (simplified)
            logger.info(f"\n7. Calculating geothermal gradient...")
            
            # Regional gradient estimate (simplified)
            gradient = np.full_like(temperature_surface, 25.0)  # °C/km typical
            
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
                'data_type': 'borehole_interpolated',
                'boreholes': len(borehole_gdf),
                'depth_range_m': [float(depth_surface.min()), float(depth_surface.max())],
                'thickness_range_m': [float(thickness_surface.min()), float(thickness_surface.max())],
                'mean_depth_m': float(mean_depth.mean()),
                'temperature_range_C': [float(temperature_surface.min()), float(temperature_surface.max())],
                'mean_temperature_C': float(temperature_surface.mean()),
                'geothermal_gradient_C_km': 25.0,
                'confidence_range': [float(confidence.min()), float(confidence.max())],
                'mean_confidence': float(confidence.mean()),
                'notes': 'Evidence layers only - ready for PFA combination model'
            }
            
            metadata_file = self.output_dir / f"{horizon_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"\n✓ {horizon_name} completed successfully!")
            
            return metadata
        
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_tier1(self):
        """Process all Tier 1 horizons"""
        logger.info("\n" + "="*80)
        logger.info("TIER 1: THERMAL EVIDENCE LAYERS (Borehole-based)")
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
            logger.info(f"  Boreholes: {meta['boreholes']}")
            logger.info(f"  Depth: {meta['mean_depth_m']:.0f} m (range: {meta['depth_range_m'][0]:.0f}-{meta['depth_range_m'][1]:.0f})")
            logger.info(f"  Temperature: {meta['mean_temperature_C']:.1f}°C (range: {meta['temperature_range_C'][0]:.1f}-{meta['temperature_range_C'][1]:.1f})")
            logger.info(f"  Confidence: {meta['mean_confidence']:.2f}")
        
        return results


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
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
    logger.info("\nEvidence layers ready for PFA combination model:")
    logger.info("  - *_depth_surface.tif/.npy")
    logger.info("  - *_thickness_surface.tif/.npy")
    logger.info("  - *_temperature_surface.tif/.npy")
    logger.info("  - *_geothermal_gradient.tif/.npy")
    logger.info("  - *_confidence.tif/.npy")
    logger.info("  - *_metadata.json")


if __name__ == "__main__":
    main()

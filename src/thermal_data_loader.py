"""
Thermal Data Loader - Tier 1 Horizons (6 boreholes-based reservoirs)

Loads borehole evidence points (Belegpunkte) with depth/thickness data,
interpolates depth and thickness surfaces, calculates temperature at each horizon,
and classifies thermal use potential.

Tier 1 Horizons:
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
from scipy.spatial import cKDTree
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Load and process thermal data for Tier 1 reservoir horizons"""
    
    # Tier 1 horizons with borehole data
    TIER1_HORIZONS = [
        'hettangian_1', 'hettangian_2',
        'pliensbachian_1', 'pliensbachian_2',
        'sinemurian_1', 'sinemurian_2'
    ]
    
    # Thermal class thresholds (°C)
    THERMAL_CLASSES = {
        0: (0, 30, "Too Cold", 0.0),          # < 30°C
        1: (30, 50, "Low-Intermediate", 0.3), # 30-50°C (heat pump + ATES)
        2: (50, 75, "Hydrothermal Direct", 0.6),  # 50-75°C (direct use)
        3: (75, 150, "High Hydrothermal", 0.9)    # > 75°C (industrial)
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
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Tier 1 horizons: {len(self.TIER1_HORIZONS)}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def load_geoTIS_temperature_data(self):
        """
        Load all GeoTIS temperature .DATA files
        
        Returns:
            Dict: {depth_value: (X, Y, T, STDV) arrays}
        """
        logger.info(f"\n{'='*80}")
        logger.info("LOADING GEOТIS TEMPERATURE DATA")
        logger.info(f"{'='*80}")
        
        if not self.geoTIS_dir.exists():
            logger.error(f"✗ GeoTIS directory not found: {self.geoTIS_dir}")
            raise FileNotFoundError(f"GeoTIS directory: {self.geoTIS_dir}")
        
        # Find all .DATA files
        data_files = sorted(self.geoTIS_dir.glob("*.DATA")) + sorted(self.geoTIS_dir.glob("*.data"))
        
        if not data_files:
            logger.error(f"✗ No .DATA files found in {self.geoTIS_dir}")
            raise FileNotFoundError(f"No .DATA files in {self.geoTIS_dir}")
        
        logger.info(f"\n✓ Found {len(data_files)} temperature depth files")
        
        geoTIS_data = {}
        
        for data_file in data_files:
            try:
                # Extract depth from filename (e.g., "+100.DATA", "-500.DATA")
                depth_str = data_file.stem  # Remove .DATA extension
                
                # Parse depth value
                try:
                    depth = int(depth_str)
                except ValueError:
                    logger.warning(f"  ⚠ Could not parse depth from filename: {data_file.name}")
                    continue
                
                # Read .DATA file
                # Format: X; Y; Subsurface temperature; STDV
                df = pd.read_csv(data_file, delimiter=';', skipinitialspace=True,
                                 names=['X', 'Y', 'Temperature', 'STDV'])
                
                if len(df) == 0:
                    logger.warning(f"  ⚠ Empty file: {data_file.name}")
                    continue
                
                # Convert to arrays
                X = df['X'].values
                Y = df['Y'].values
                T = df['Temperature'].values
                STDV = df['STDV'].values
                
                geoTIS_data[depth] = {
                    'X': X,
                    'Y': Y,
                    'T': T,
                    'STDV': STDV
                }
                
                logger.info(f"  ✓ {data_file.name}: {len(df)} points, T range [{T.min():.1f}, {T.max():.1f}]°C")
            
            except Exception as e:
                logger.error(f"  ✗ Error reading {data_file.name}: {e}")
                continue
        
        if not geoTIS_data:
            raise ValueError("No valid GeoTIS data loaded")
        
        logger.info(f"\n✓ Loaded {len(geoTIS_data)} depth levels")
        depths = sorted(geoTIS_data.keys())
        logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m")
        
        self.geoTIS_data = geoTIS_data
        return geoTIS_data
    
    def interpolate_temperature_at_depth(self, target_depth, method='linear'):
        """
        Interpolate temperature at arbitrary depth from GeoTIS data
        
        Args:
            target_depth: Depth in meters (positive = below surface)
            method: 'linear' or 'nearest'
        
        Returns:
            (X_grid, Y_grid, T_grid, STDV_grid) interpolated to study area grid
        """
        if not self.geoTIS_data:
            raise ValueError("GeoTIS data not loaded. Call load_geoTIS_temperature_data() first")
        
        # Find nearest depths
        available_depths = sorted(self.geoTIS_data.keys())
        
        if target_depth in self.geoTIS_data:
            # Exact depth available
            data = self.geoTIS_data[target_depth]
            return data['X'], data['Y'], data['T'], data['STDV']
        
        # Find surrounding depths
        below = [d for d in available_depths if d <= target_depth]
        above = [d for d in available_depths if d >= target_depth]
        
        if method == 'nearest':
            # Use nearest depth
            if len(below) > 0 and len(above) > 0:
                d_below = below[-1]
                d_above = above[0]
                dist_below = target_depth - d_below
                dist_above = d_above - target_depth
                
                if dist_below <= dist_above:
                    depth = d_below
                else:
                    depth = d_above
            elif len(below) > 0:
                depth = below[-1]
            else:
                depth = above[0]
            
            data = self.geoTIS_data[depth]
            return data['X'], data['Y'], data['T'], data['STDV']
        
        else:  # linear interpolation
            if len(below) == 0 or len(above) == 0:
                # Use nearest available
                if len(below) > 0:
                    depth = below[-1]
                else:
                    depth = above[0]
                
                data = self.geoTIS_data[depth]
                return data['X'], data['Y'], data['T'], data['STDV']
            
            d_below = below[-1]
            d_above = above[0]
            
            data_below = self.geoTIS_data[d_below]
            data_above = self.geoTIS_data[d_above]
            
            # Linear interpolation weight
            w_below = (d_above - target_depth) / (d_above - d_below)
            w_above = (target_depth - d_below) / (d_above - d_below)
            
            # Combine data
            # Need to match points - use nearest neighbor
            X_below = data_below['X']
            Y_below = data_below['Y']
            T_below = data_below['T']
            
            X_above = data_above['X']
            Y_above = data_above['Y']
            T_above = data_above['T']
            
            # Use below data as reference, interpolate to matching points
            points_below = np.column_stack([X_below, Y_below])
            points_above = np.column_stack([X_above, Y_above])
            
            # Interpolate above data to below points
            T_above_interp = griddata(points_above, T_above, points_below, method='nearest')
            
            # Interpolate temperature
            T_interp = w_below * T_below + w_above * T_above_interp
            STDV_interp = np.sqrt((w_below * data_below['STDV'])**2 + (w_above * T_above_interp)**2)
            
            return X_below, Y_below, T_interp, STDV_interp
    
    def find_belegpunkte_shapefile(self, horizon_name):
        """Find Belegpunkte shapefile for horizon"""
        # Build possible filenames
        possible_names = [
            f"{horizon_name}_Belegpunkte.shp",
            f"{horizon_name.replace('_', '')}Belegpunkte.shp",
            f"{horizon_name.upper()}_Belegpunkte.shp"
        ]
        
        for name in possible_names:
            for path in self.base_dir.rglob(name):
                return path
        
        return None
    
    def find_potential_shapefile(self, horizon_name):
        """Find Potential shapefile for masking"""
        # Extract base name from config
        for res_config in self.config['reservoirs']:
            if res_config['name'] == horizon_name:
                return self.base_dir / res_config['path']
        
        return None
    
    def load_belegpunkte_data(self, horizon_name):
        """
        Load borehole evidence points (Belegpunkte) for a horizon
        
        Returns:
            GeoDataFrame with columns: borehole_id, X, Y, top_depth, thickness, geometry
        """
        shp_path = self.find_belegpunkte_shapefile(horizon_name)
        
        if shp_path is None:
            logger.warning(f"  ⚠ Belegpunkte not found for {horizon_name}")
            return None
        
        try:
            gdf = gpd.read_file(shp_path)
            
            logger.info(f"  ✓ Loaded: {shp_path.name}")
            logger.info(f"    Records: {len(gdf)}")
            logger.info(f"    Columns: {list(gdf.columns)[:15]}")
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Extract coordinates
            gdf['X'] = gdf.geometry.x
            gdf['Y'] = gdf.geometry.y
            
            # Find depth/thickness columns (common names)
            depth_cols = [c for c in gdf.columns if any(kw in c.lower() for kw in 
                         ['top', 'tiefe', 'depth', 'oben'])]
            thick_cols = [c for c in gdf.columns if any(kw in c.lower() for kw in 
                         ['thick', 'mächt', 'maechtigkeit', 'dicke'])]
            
            logger.info(f"    Potential depth columns: {depth_cols}")
            logger.info(f"    Potential thickness columns: {thick_cols}")
            
            return gdf
        
        except Exception as e:
            logger.error(f"  ✗ Error loading {shp_path}: {e}")
            return None
    
    def interpolate_depth_surface(self, borehole_gdf, column_name, method='linear'):
        """
        Interpolate depth or thickness values to grid
        
        Args:
            borehole_gdf: GeoDataFrame with borehole points
            column_name: Column containing depth/thickness values
            method: 'linear', 'cubic', or 'nearest'
        
        Returns:
            2D interpolated array
        """
        # Extract valid data
        valid = borehole_gdf[borehole_gdf[column_name].notna()].copy()
        
        if len(valid) < 3:
            logger.warning(f"    ⚠ Less than 3 valid points for {column_name}, skipping interpolation")
            return None
        
        logger.info(f"    Interpolating {column_name}: {len(valid)} valid points")
        
        # Create regular grid
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        # Prepare interpolation
        points = np.column_stack([valid['X'], valid['Y']])
        values = valid[column_name].values
        
        # Flatten grid for interpolation
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
            
            return interpolated.astype(np.float32)
        
        except Exception as e:
            logger.error(f"    ✗ Interpolation failed: {e}")
            return None
    
    def classify_thermal_use(self, temperature):
        """
        Classify temperature to thermal use class
        
        Args:
            temperature: Temperature in °C (scalar or array)
        
        Returns:
            thermal_class, thermal_value, class_name
        """
        temp_scalar = np.isscalar(temperature)
        temp = np.atleast_1d(temperature)
        
        thermal_class = np.zeros_like(temp, dtype=int)
        thermal_value = np.zeros_like(temp, dtype=float)
        
        # Class 0: < 30°C
        mask = temp < 30
        thermal_class[mask] = 0
        thermal_value[mask] = 0.0
        
        # Class 1: 30-50°C
        mask = (temp >= 30) & (temp < 50)
        thermal_class[mask] = 1
        thermal_value[mask] = 0.3 + (temp[mask] - 30) / 200  # 0.3-0.5
        
        # Class 2: 50-75°C
        mask = (temp >= 50) & (temp < 75)
        thermal_class[mask] = 2
        thermal_value[mask] = 0.6 + (temp[mask] - 50) / 250  # 0.6-0.8
        
        # Class 3: >= 75°C
        mask = temp >= 75
        thermal_class[mask] = 3
        thermal_value[mask] = 0.9 + np.minimum((temp[mask] - 75) / 100, 0.1)  # 0.9-1.0
        
        if temp_scalar:
            return int(thermal_class[0]), float(thermal_value[0])
        else:
            return thermal_class, thermal_value
    
    def load_potential_mask(self, horizon_name):
        """Load potential shapefile as raster mask"""
        shp_path = self.find_potential_shapefile(horizon_name)
        
        if shp_path is None or not shp_path.exists():
            logger.warning(f"    ⚠ Potential shapefile not found: {horizon_name}")
            return None
        
        try:
            gdf = gpd.read_file(shp_path)
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Rasterize to binary mask
            transform, _ = self.create_transform_and_bounds()
            width, height = self.grid_size
            
            mask = rasterize(
                gdf.geometry,
                out_shape=(height, width),
                transform=transform,
                default_value=0,
                fill=1
            ).astype(np.float32)
            
            logger.info(f"    ✓ Loaded potential mask")
            logger.info(f"      Reservoir pixels: {np.count_nonzero(mask)}")
            
            return mask
        
        except Exception as e:
            logger.error(f"    ✗ Error loading potential mask: {e}")
            return None
    
    def create_transform_and_bounds(self):
        """Create rasterio transform and bounds"""
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        width, height = self.grid_size
        transform = from_bounds(left, bottom, right, top, width, height)
        bounds = (left, bottom, right, top)
        
        return transform, bounds
    
    def save_raster(self, raster, name, dtype=np.float32):
        """Save raster to GeoTIFF and NumPy"""
        transform, _ = self.create_transform_and_bounds()
        
        # Save as GeoTIFF
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
        
        # Save as NumPy
        npy_path = self.output_dir / f"{name}.npy"
        np.save(npy_path, raster)
        
        logger.info(f"    ✓ Saved: {name}")
        return tif_path, npy_path
    
    def process_horizon(self, horizon_name):
        """
        Process single Tier 1 horizon
        
        Returns:
            Dict with processed data
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {horizon_name} (TIER 1)")
        logger.info(f"{'='*80}")
        
        try:
            # Load borehole data
            logger.info(f"\n1. Loading borehole data (Belegpunkte)...")
            borehole_gdf = self.load_belegpunkte_data(horizon_name)
            
            if borehole_gdf is None or len(borehole_gdf) == 0:
                logger.error(f"✗ No borehole data for {horizon_name}")
                return None
            
            # Find depth and thickness columns
            # Common patterns: top_tiefe, Top, tiefe_top, depth_top, etc.
            depth_col = None
            thick_col = None
            
            for col in borehole_gdf.columns:
                col_lower = col.lower()
                if depth_col is None and any(kw in col_lower for kw in ['top', 'tiefe_top', 'depth_top', 'oben']):
                    depth_col = col
                if thick_col is None and any(kw in col_lower for kw in ['thick', 'mächt', 'maechtigkeit', 'maechtig', 'dicke']):
                    if 'sand' not in col_lower:  # Skip sand_thickness for now
                        thick_col = col
            
            if depth_col is None or thick_col is None:
                logger.error(f"✗ Could not find depth/thickness columns")
                logger.error(f"  Available: {list(borehole_gdf.columns)}")
                return None
            
            logger.info(f"  ✓ Using columns: depth='{depth_col}', thickness='{thick_col}'")
            
            # Interpolate depth surface
            logger.info(f"\n2. Interpolating depth surface...")
            depth_surface = self.interpolate_depth_surface(borehole_gdf, depth_col, method='linear')
            
            if depth_surface is None:
                logger.error(f"✗ Failed to interpolate depth surface")
                return None
            
            # Interpolate thickness surface
            logger.info(f"\n3. Interpolating thickness surface...")
            thickness_surface = self.interpolate_depth_surface(borehole_gdf, thick_col, method='linear')
            
            if thickness_surface is None:
                logger.error(f"✗ Failed to interpolate thickness surface")
                return None
            
            # Calculate mean depth
            logger.info(f"\n4. Calculating mean depth...")
            mean_depth = depth_surface + (thickness_surface / 2)
            
            logger.info(f"  Mean depth range: [{mean_depth.min():.0f}, {mean_depth.max():.0f}] m")
            logger.info(f"  Thickness range: [{thickness_surface.min():.0f}, {thickness_surface.max():.0f}] m")
            
            # Interpolate temperature at mean depth
            logger.info(f"\n5. Interpolating temperature at mean depth...")
            
            # Create temperature surface by interpolating at each pixel
            temperature_surface = np.zeros_like(mean_depth)
            stdv_surface = np.zeros_like(mean_depth)
            
            # Get unique depth values and their counts
            unique_depths = np.unique(mean_depth[mean_depth > 0])
            
            logger.info(f"  Processing {len(unique_depths)} unique depths...")
            
            for depth in unique_depths:
                # Get all pixels at this depth (approximately)
                mask = np.abs(mean_depth - depth) < 5  # Within 5m
                
                if np.any(mask):
                    # Interpolate temperature at this depth
                    X_temp, Y_temp, T_temp, STDV_temp = self.interpolate_temperature_at_depth(depth, method='linear')
                    
                    # Create spatial index for temperature data
                    points_temp = np.column_stack([X_temp, Y_temp])
                    
                    # Get grid points that need interpolation
                    grid_x = np.linspace(self.extent['left'], self.extent['right'], self.grid_size[0])
                    grid_y = np.linspace(self.extent['bottom'], self.extent['top'], self.grid_size[1])
                    X_mesh, Y_mesh = np.meshgrid(grid_x, grid_y)
                    
                    # Interpolate temperature to grid
                    T_grid = griddata(points_temp, T_temp, (X_mesh, Y_mesh), method='linear')
                    STDV_grid = griddata(points_temp, STDV_temp, (X_mesh, Y_mesh), method='linear')
                    
                    # Fill NaNs
                    T_grid_nn = griddata(points_temp, T_temp, (X_mesh, Y_mesh), method='nearest')
                    STDV_grid_nn = griddata(points_temp, STDV_temp, (X_mesh, Y_mesh), method='nearest')
                    
                    nan_mask = np.isnan(T_grid)
                    T_grid[nan_mask] = T_grid_nn[nan_mask]
                    STDV_grid[nan_mask] = STDV_grid_nn[nan_mask]
                    
                    # Assign to surface
                    temperature_surface[mask] = T_grid[mask]
                    stdv_surface[mask] = STDV_grid[mask]
            
            logger.info(f"  ✓ Temperature range: [{temperature_surface.min():.1f}, {temperature_surface.max():.1f}]°C")
            
            # Calculate geothermal gradient
            logger.info(f"\n6. Calculating geothermal gradient...")
            
            # Get top and base temperatures (approximate from surface data)
            top_depth = depth_surface
            base_depth = depth_surface + thickness_surface
            
            # This is simplified - ideally we'd interpolate at exact top/base depths
            gradient = np.zeros_like(temperature_surface)
            valid = thickness_surface > 0
            
            # Estimate from regional average gradient
            # For now, use a typical value and refine if needed
            regional_gradient = 25  # °C/km typical for North German Basin
            gradient[valid] = regional_gradient
            
            # Classify thermal use
            logger.info(f"\n7. Classifying thermal use...")
            thermal_class, thermal_use = self.classify_thermal_use(temperature_surface)
            
            # Load potential mask
            logger.info(f"\n8. Masking to reservoir extent...")
            potential_mask = self.load_potential_mask(horizon_name)
            
            # Apply mask (multiply)
            if potential_mask is not None:
                thermal_use = thermal_use * potential_mask
                thermal_class = thermal_class.astype(float) * potential_mask
                thermal_class = thermal_class.astype(int)
            
            # Count by class
            logger.info(f"\n  Thermal use distribution:")
            for class_id, (t_min, t_max, desc, _) in self.THERMAL_CLASSES.items():
                count = np.count_nonzero(thermal_class == class_id)
                pct = 100 * count / thermal_class.size if thermal_class.size > 0 else 0
                logger.info(f"    Class {class_id} ({desc:20s}): {count:>8} pixels ({pct:>5.1f}%)")
            
            # Save outputs
            logger.info(f"\n9. Saving outputs...")
            
            self.save_raster(depth_surface, f"{horizon_name}_depth_to_horizon")
            self.save_raster(thickness_surface, f"{horizon_name}_thickness")
            self.save_raster(temperature_surface, f"{horizon_name}_temperature")
            self.save_raster(gradient, f"{horizon_name}_geothermal_gradient")
            self.save_raster(thermal_class.astype(np.float32), f"{horizon_name}_thermal_class")
            self.save_raster(thermal_use, f"{horizon_name}_thermal_use")
            self.save_raster(stdv_surface, f"{horizon_name}_temperature_stdv")
            
            # Save metadata
            metadata = {
                'horizon': horizon_name,
                'tier': 1,
                'boreholes': len(borehole_gdf),
                'depth_range': [float(depth_surface.min()), float(depth_surface.max())],
                'thickness_range': [float(thickness_surface.min()), float(thickness_surface.max())],
                'mean_depth': float(mean_depth.mean()),
                'temperature_range': [float(temperature_surface.min()), float(temperature_surface.max())],
                'mean_temperature': float(temperature_surface.mean()),
                'geothermal_gradient': float(regional_gradient),
                'thermal_use_range': [float(thermal_use.min()), float(thermal_use.max())],
                'mean_thermal_use': float(thermal_use.mean()),
                'class_distribution': {
                    int(cid): int(np.count_nonzero(thermal_class == cid))
                    for cid in range(4)
                }
            }
            
            metadata_file = self.output_dir / f"{horizon_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"  ✓ Metadata saved: {metadata_file.name}")
            
            logger.info(f"\n✓ {horizon_name} completed successfully!\n")
            
            return metadata
        
        except Exception as e:
            logger.error(f"✗ Error processing {horizon_name}: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_tier1_horizons(self):
        """Process all Tier 1 horizons"""
        logger.info("\n" + "="*80)
        logger.info("TIER 1 THERMAL DATA PROCESSING (6 HORIZONS WITH BOREHOLE DATA)")
        logger.info("="*80)
        
        # Load GeoTIS data once
        try:
            self.load_geoTIS_temperature_data()
        except Exception as e:
            logger.error(f"✗ Failed to load GeoTIS data: {e}")
            return {}
        
        # Process each horizon
        results = {}
        
        for horizon in self.TIER1_HORIZONS:
            metadata = self.process_horizon(horizon)
            
            if metadata is not None:
                results[horizon] = metadata
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("TIER 1 PROCESSING SUMMARY")
        logger.info("="*80)
        logger.info(f"\nSuccessfully processed: {len(results)}/{len(self.TIER1_HORIZONS)} horizons")
        
        if results:
            logger.info(f"\nHorizon Summary:")
            for horizon, metadata in sorted(results.items()):
                logger.info(f"\n  {horizon}:")
                logger.info(f"    Depth: {metadata['depth_range'][0]:.0f}-{metadata['depth_range'][1]:.0f} m (mean: {metadata['mean_depth']:.0f} m)")
                logger.info(f"    Thickness: {metadata['thickness_range'][0]:.0f}-{metadata['thickness_range'][1]:.0f} m")
                logger.info(f"    Temperature: {metadata['temperature_range'][0]:.1f}-{metadata['temperature_range'][1]:.1f}°C (mean: {metadata['mean_temperature']:.1f}°C)")
                logger.info(f"    Thermal use: {metadata['mean_thermal_use']:.2f}")
                logger.info(f"    Class distribution: {metadata['class_distribution']}")
        
        return results


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    # Initialize loader
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs",
        geoTIS_dir="data/inputs/geoTIS/temperature"
    )
    
    # Process Tier 1 horizons
    results = loader.process_tier1_horizons()
    
    logger.info("\n✓ Tier 1 thermal processing complete!")
    logger.info(f"  Output directory: {loader.output_dir}")


if __name__ == "__main__":
    main()

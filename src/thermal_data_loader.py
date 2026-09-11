"""
Thermal Data Loader for geoPFA
Loads borehole data, calculates mean depths, queries GeoTIS, and interpolates temperature grids
"""

import json
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
from scipy.interpolate import griddata, Rbf
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Load and process thermal data for PFA analysis"""
    
    def __init__(self, config_path, base_dir="data/inputs", geoTIS_dir="data/inputs/geotIS/temperature_basin_filtered"):
        """Initialize thermal data loader"""
        self.base_dir = Path(base_dir)
        self.geoTIS_dir = Path(geoTIS_dir)
        
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.crs = self.config['crs']
        self.grid_size = self.config['grid_size']
        self.extent = self.config['extent']
        
        # Temperature physical limits
        self.T_MIN = 3.0      # Minimum realistic temperature (°C)
        self.T_MAX = 200.0    # Maximum realistic temperature (°C)
        
        # Heat flow gradient for fallback
        self.geothermal_gradient = 23.1  # °C/km (standard for stable continental crust)
        self.surface_temp = 10.0         # °C
        
        # GeoTIS data storage
        self.geoTIS_data = {}
        
        # Load GeoTIS temperature data
        self.load_geoTIS_temperature_data()
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Temperature clamping: [{self.T_MIN}, {self.T_MAX}]°C (prevents extrapolation artifacts)")
        logger.info(f"  Spatial filter (basin bounds):")
        logger.info(f"    X: {self.extent['left']} to {self.extent['right']} m")
        logger.info(f"    Y: {self.extent['bottom']} to {self.extent['top']} m")
        logger.info(f"    (Prevents contamination from Molasse & Rhine Graben)")

    def load_geoTIS_temperature_data(self):
        """Load GeoTIS temperature data from basin-filtered files"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GEOТIS TEMPERATURE DATA")
        logger.info("="*80)
        
        if not self.geoTIS_dir.exists():
            logger.warning(f"⚠ GeoTIS directory not found: {self.geoTIS_dir}")
            return
        
        # Find all .data files
        data_files = sorted(self.geoTIS_dir.glob("*.data"))
        logger.info(f"✓ Found {len(data_files)} temperature depth files")
        
        # Extract depth levels from filenames
        depths = []
        for f in data_files:
            # Filename format: T2022_LIAG_AGEMAR_±DEPTH.data
            parts = f.stem.split('_')
            try:
                depth_str = parts[-1]
                depth = int(depth_str)
                depths.append(depth)
                
                # Load data
                df = pd.read_csv(f, delimiter=';')
                if len(df) > 0:
                    # Column 0: X, Column 1: Y, Column 2: Temperature
                    self.geoTIS_data[depth] = {
                        'X': df.iloc[:, 0].values,
                        'Y': df.iloc[:, 1].values,
                        'T': pd.to_numeric(df.iloc[:, 2], errors='coerce').values,
                    }
            except Exception as e:
                continue
        
        depths = sorted(self.geoTIS_data.keys())
        logger.info(f"✓ Loaded {len(depths)} depth levels (all Germany)")
        if depths:
            logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m")

    def load_borehole_data_all_horizons(self):
        """
        Load borehole data from ALL 6 horizons and find boreholes with complete coverage
        Returns: Dict of horizon -> GeoDataFrame, and list of boreholes in ALL horizons
        """
        horizons = {
            'hettangian_1': 'Het1_Belegpunkte.shp',
            'hettangian_2': 'Het2_Belegpunkte.shp',
            'sinemurian_1': 'Sin1_Belegpunkte.shp',
            'sinemurian_2': 'Sin2_Belegpunkte.shp',
            'pliensbachian_1': 'Pli1_Belegpunkte.shp',
            'pliensbachian_2': 'Pli2_Belegpunkte.shp',
        }
        
        gdfs = {}
        borehole_names_per_horizon = {}
        
        # Load all horizons
        for horizon, filename in horizons.items():
            shp_path = self.base_dir / "reservoirs" / horizon / filename
            try:
                gdf = gpd.read_file(shp_path)
                gdf['borehole_name'] = gdf.iloc[:, 0]
                gdf['horizon'] = horizon
                gdfs[horizon] = gdf
                
                # Store borehole names (where Teufe > 0)
                borehole_names_per_horizon[horizon] = set(gdf[gdf['Teufe'] > 0]['borehole_name'].values)
            except Exception as e:
                logger.warning(f"⚠ Could not load {horizon}: {e}")
                gdfs[horizon] = None
                borehole_names_per_horizon[horizon] = set()
        
        # Find boreholes that appear in ALL 6 horizons with valid data
        all_names = set.intersection(*[names for names in borehole_names_per_horizon.values() if names])
        
        logger.info(f"\n✓ Borehole coverage analysis:")
        logger.info(f"  Total boreholes with complete 6-horizon coverage: {len(all_names)}")
        
        return gdfs, all_names

    def calculate_temperature_from_depth(self, depths_m):
        """Calculate temperature using geothermal gradient"""
        # depth_m is negative (below surface), convert to positive for gradient
        abs_depths = np.abs(depths_m)
        temps = self.surface_temp + (abs_depths / 1000.0) * self.geothermal_gradient
        return np.clip(temps, self.T_MIN, self.T_MAX)

    def interpolate_temperature_at_depth(self, target_depth, method='linear'):
        """
        Interpolate temperature at a specific depth using available GeoTIS data
        
        Args:
            target_depth: Depth in meters (negative = below surface)
            method: 'linear', 'nearest'
        
        Returns:
            Tuple (X, Y, T, STDV) or None if no data available
        """
        if not self.geoTIS_data:
            return None
        
        depths = sorted(self.geoTIS_data.keys())
        
        # Clamp to available depth range
        if target_depth < depths[0]:
            target_depth = depths[0]
        elif target_depth > depths[-1]:
            target_depth = depths[-1]
        
        # Find bracketing depths
        if target_depth in self.geoTIS_data:
            # Exact match
            data = self.geoTIS_data[target_depth]
            X, Y, T = data['X'], data['Y'], data['T']
        else:
            # Interpolate between two depths
            upper_depth = min([d for d in depths if d >= target_depth], default=depths[-1])
            lower_depth = max([d for d in depths if d <= target_depth], default=depths[0])
            
            if upper_depth == lower_depth:
                data = self.geoTIS_data[upper_depth]
                X, Y, T = data['X'], data['Y'], data['T']
            else:
                # Linear interpolation
                data_lower = self.geoTIS_data[lower_depth]
                data_upper = self.geoTIS_data[upper_depth]
                
                weight = (target_depth - lower_depth) / (upper_depth - lower_depth)
                
                X = data_lower['X']
                Y = data_lower['Y']
                T = data_lower['T'] * (1 - weight) + data_upper['T'] * weight
        
        # Remove invalid temperatures (-99999.0) and NaNs
        mask = (T > -99999) & (~np.isnan(T))
        X = X[mask]
        Y = Y[mask]
        T = T[mask]
        
        if len(T) == 0:
            return None
        
        # Standard deviation (simplified: use 5% of values)
        STDV = np.std(T)
        
        return X, Y, T, STDV

    def interpolate_temperature_grid_hybrid(self, mean_depth_surface, borehole_gdf):
        """
        Interpolate temperature using HYBRID approach with spatial filtering
        ✅ FILTER: Only use boreholes with complete 6-horizon stratigraphic coverage
        
        Args:
            mean_depth_surface: Mean depth surface (for reference)
            borehole_gdf: Borehole GeoDataFrame for current horizon
        
        Returns:
            Tuple (temperature_grid, geotis_available_mask)
        """
        logger.info(f"    Interpolating temperature (HYBRID: GeoTIS spatially-filtered + gradient fallback)...")
        
        # ✅ FILTER: Load all horizons and find boreholes with COMPLETE coverage
        all_gdfs, complete_boreholes = self.load_borehole_data_all_horizons()
        
        # Filter current horizon to ONLY boreholes with complete 6-horizon coverage
        borehole_gdf_filtered = borehole_gdf[borehole_gdf['borehole_name'].isin(complete_boreholes)].copy()
        
        logger.info(f"      ✓ Filtered to {len(borehole_gdf_filtered)}/{len(borehole_gdf)} boreholes with complete 6-horizon stratigraphic coverage")
        
        if len(borehole_gdf_filtered) < 3:
            logger.warning(f"      ⚠ Less than 3 boreholes with complete coverage")
            logger.info(f"      Using all boreholes (may have stratigraphic inconsistencies)")
            borehole_gdf_filtered = borehole_gdf.copy()
        
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        temperature_at_boreholes = np.full(len(borehole_gdf_filtered), np.nan, dtype=np.float32)
        geotis_available = np.zeros(len(borehole_gdf_filtered), dtype=bool)
        
        # Query GeoTIS at borehole locations (spatially filtered)
        if self.geoTIS_data:
            logger.info(f"      Querying GeoTIS at {len(borehole_gdf_filtered)} borehole locations (spatially filtered to basin)...")
            
            for idx, (i, row) in enumerate(borehole_gdf_filtered.iterrows()):
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
            logger.info(f"      GeoTIS temperatures found (basin-filtered): {geotis_count}/{len(borehole_gdf_filtered)} boreholes")
        
        # Fallback: use heat flow gradient where GeoTIS is missing
        gradient_temps = self.calculate_temperature_from_depth(borehole_gdf_filtered['mean_depth'].values)
        
        # Use GeoTIS where available, gradient elsewhere
        temp_blended = np.where(
            geotis_available,
            temperature_at_boreholes,
            gradient_temps
        )
        
        borehole_gdf_filtered['temperature_blended'] = temp_blended
        
        # RBF interpolate blended temperatures
        logger.info(f"      RBF interpolating blended temperatures across basin (complete-coverage boreholes only)...")
        
        x_pts = borehole_gdf_filtered['X'].values
        y_pts = borehole_gdf_filtered['Y'].values
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
        logger.info(f"      Data sources: {geotis_count} GeoTIS (spatially filtered), {len(borehole_gdf_filtered) - geotis_count} gradient-derived")
        
        return temperature_grid.astype(np.float32), geotis_available


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs",
        geoTIS_dir="data/inputs/geotIS/temperature_basin_filtered"
    )


if __name__ == "__main__":
    main()

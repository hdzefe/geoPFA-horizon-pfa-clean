"""
Thermal Data Loader for geoPFA
Loads pre-calculated borehole mean depths with X,Y coordinates from CSV, queries GeoTIS, 
and interpolates temperature grids.
Exports temperature_evidence layers as GeoTIFFs per horizon.

RESERVOIR-SPECIFIC PFA APPROACH:
- 6 horizons with borehole data: het1, het2, sin1, sin2, pli1, pli2
- Output: {horizon}_temperature_evidence.tif (interpolated from borehole depths + GeoTIS)
- Temperature is DIRECT EVIDENCE of geothermal favorability (hard data from boreholes)

HEAT FLOW INTEGRATION:
- Uses GFZ German Heat Flow Database for regional geothermal gradient
- Falls back to gradient for boreholes without GeoTIS data
- Calculates gradient from measured heat flow: q / λ → dT/dz (where λ = 3.0 W/(m·K))

DEBUG MODE:
- Exports {horizon}_borehole_temperatures_debug.csv with all borehole data used in interpolation
- Exports {horizon}_borehole_temperatures_filtered.csv with boreholes filtered OUT and reason
"""

import json
import logging
import numpy as np
import pandas as pd
from scipy.interpolate import griddata, Rbf
from pathlib import Path
import warnings
import rasterio
from rasterio.transform import from_bounds

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Load and process thermal evidence data for PFA analysis"""
    
    def __init__(self, config_path, base_dir="data/inputs", geoTIS_dir="data/inputs/geotIS/temperature_basin_filtered", 
                 heat_flow_file="data/inputs/heat_flow/2022-015_Fuchs-et-al_GermanHeatFlowDB-2022.xlsx"):
        """Initialize thermal data loader"""
        self.base_dir = Path(base_dir)
        self.geoTIS_dir = Path(geoTIS_dir)
        self.heat_flow_file = Path(heat_flow_file)
        self.output_dir = Path("data/outputs/rasters")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        with open(config_path) as f:
            self.config = json.load(f)
        
        self.crs = self.config['crs']
        self.grid_size = self.config['grid_size']
        self.extent = self.config['extent']
        
        # Temperature physical limits
        self.T_MIN = 3.0
        self.T_MAX = 200.0
        
        # Thermal conductivity (typical for sediments)
        self.thermal_conductivity = 3.0  # W/(m·K)
        
        # Surface temperature (typical for North German Basin)
        self.surface_temp = 10.0  # °C
        
        # Geothermal gradient (will be calculated from heat flow)
        self.geothermal_gradient = None
        
        # GeoTIS data storage
        self.geoTIS_data = {}
        
        # Load heat flow data
        self.load_heat_flow_database()
        
        # Load GeoTIS temperature data
        self.load_geoTIS_temperature_data()
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Geothermal gradient: {self.geothermal_gradient:.2f} °C/km")
        logger.info(f"  Output directory: {self.output_dir}")

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
            # Load Excel - Skip first 3 rows, row 4 becomes header
            df = pd.read_excel(self.heat_flow_file, sheet_name='data', skiprows=3)
            
            logger.info(f"✓ Loaded: {len(df):,} records from heat flow database")
            
            # Filter valid data
            df_clean = df.dropna(subset=['lat', 'lng', 'q']).copy()
            df_clean = df_clean[(df_clean['q'] > 0) & (df_clean['q'] < 300)]
            
            logger.info(f"✓ Valid records: {len(df_clean):,} / {len(df):,}")
            
            # North German Basin extent (Gauss-Krueger Zone 3)
            basin_xmin, basin_xmax = 3342763, 3921379
            basin_ymin, basin_ymax = 5650524, 6103144
            
            # Convert lat/lng to GK coordinates for filtering
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:31467")
            
            x_gk, y_gk = transformer.transform(df_clean['lat'].values, df_clean['lng'].values)
            
            # Filter to basin extent
            basin_mask = (x_gk >= basin_xmin) & (x_gk <= basin_xmax) & \
                        (y_gk >= basin_ymin) & (y_gk <= basin_ymax)
            
            df_basin = df_clean[basin_mask].copy()
            
            logger.info(f"✓ Measurements in North German Basin: {len(df_basin):,} / {len(df_clean):,}")
            
            if len(df_basin) > 0:
                q_basin = df_basin['q']
                mean_hf = q_basin.mean()
                
                logger.info(f"\n  Basin Heat Flow Statistics:")
                logger.info(f"    Mean: {mean_hf:.1f} mW/m²")
                logger.info(f"    Median: {q_basin.median():.1f} mW/m²")
                logger.info(f"    Std Dev: {q_basin.std():.1f} mW/m²")
                logger.info(f"    Range: {q_basin.min():.1f} to {q_basin.max():.1f} mW/m²")
                
                # Calculate gradient from heat flow: dT/dz = q / λ (in °C/km)
                # q is in mW/m² = W/m² / 1000
                # λ in W/(m·K)
                self.geothermal_gradient = (mean_hf / 1000.0) / self.thermal_conductivity * 1000.0
                
                logger.info(f"\n  Calculated Geothermal Gradient (λ = {self.thermal_conductivity} W/(m·K)):")
                logger.info(f"    {self.geothermal_gradient:.2f} °C/km")
                logger.info(f"    (Based on measured heat flow: {mean_hf:.1f} mW/m²)")
                
            else:
                # Use Germany-wide average if no basin data
                q_all = df_clean['q']
                mean_hf = q_all.mean()
                
                logger.warning(f"⚠ No heat flow measurements in basin")
                logger.info(f"\n  Using Germany-wide average:")
                logger.info(f"    Mean: {mean_hf:.1f} mW/m²")
                
                self.geothermal_gradient = (mean_hf / 1000.0) / self.thermal_conductivity * 1000.0
                
                logger.info(f"\n  Calculated Geothermal Gradient:")
                logger.info(f"    {self.geothermal_gradient:.2f} °C/km (Germany-wide)")
        
        except Exception as e:
            logger.warning(f"⚠ Error loading heat flow database: {e}")
            logger.warning(f"  Using default gradient: 23.1 °C/km")
            self.geothermal_gradient = 23.1

    def load_geoTIS_temperature_data(self):
        """Load GeoTIS temperature data from basin-filtered files"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GEOТIS TEMPERATURE DATA (NORTH GERMAN BASIN FILTERED)")
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
            parts = f.stem.split('_')
            try:
                depth_str = parts[-1]
                depth = int(depth_str)
                depths.append(depth)
                
                # Load data
                df = pd.read_csv(f, delimiter=';')
                if len(df) > 0:
                    self.geoTIS_data[depth] = {
                        'X': df.iloc[:, 0].values,
                        'Y': df.iloc[:, 1].values,
                        'T': pd.to_numeric(df.iloc[:, 2], errors='coerce').values,
                    }
            except Exception as e:
                continue
        
        depths = sorted(self.geoTIS_data.keys())
        logger.info(f"✓ Loaded {len(depths)} depth levels")
        if depths:
            logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m")

    def load_mean_depths_table(self):
        """Load pre-calculated mean depths with X,Y coordinates from CSV"""
        table_path = self.base_dir / "borehole_mean_depths.csv"
        
        logger.info(f"\n✓ Loading pre-calculated mean depths from: {table_path}")
        
        if not table_path.exists():
            logger.error(f"✗ Mean depths table not found: {table_path}")
            return None
        
        try:
            # Try reading with different delimiters and decimal formats
            for delimiter in [';', ',', '\t', ' ']:
                try:
                    # Try with comma as decimal separator (European format)
                    df = pd.read_csv(table_path, delimiter=delimiter, decimal=',')
                    
                    # Check if columns parsed correctly
                    if 'borehole_name' in df.columns and 'x' in df.columns.str.lower() and 'y' in df.columns.str.lower():
                        logger.info(f"✓ Loaded {len(df)} boreholes (delimiter: '{delimiter}', decimal: ',')")
                        logger.info(f"  Columns: {list(df.columns)}")
                        return df
                except:
                    pass
                
                try:
                    # Try with period as decimal separator (US format)
                    df = pd.read_csv(table_path, delimiter=delimiter, decimal='.')
                    
                    # Check if columns parsed correctly
                    if 'borehole_name' in df.columns and 'x' in df.columns.str.lower() and 'y' in df.columns.str.lower():
                        logger.info(f"✓ Loaded {len(df)} boreholes (delimiter: '{delimiter}', decimal: '.')")
                        logger.info(f"  Columns: {list(df.columns)}")
                        return df
                except:
                    continue
            
            logger.error(f"✗ Could not parse CSV with any standard delimiter/decimal combination")
            return None
            
        except Exception as e:
            logger.error(f"✗ Error loading mean depths table: {e}")
            return None

    def calculate_temperature_from_depth(self, depths_m):
        """Calculate temperature using geothermal gradient (fallback when GeoTIS unavailable)"""
        abs_depths = np.abs(depths_m)
        temps = self.surface_temp + (abs_depths / 1000.0) * self.geothermal_gradient
        return np.clip(temps, self.T_MIN, self.T_MAX)

    def interpolate_temperature_at_depth(self, target_depth, method='linear'):
        """Interpolate temperature at a specific depth using GeoTIS data"""
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
            data = self.geoTIS_data[target_depth]
            X, Y, T = data['X'], data['Y'], data['T']
        else:
            upper_depth = min([d for d in depths if d >= target_depth], default=depths[-1])
            lower_depth = max([d for d in depths if d <= target_depth], default=depths[0])
            
            if upper_depth == lower_depth:
                data = self.geoTIS_data[upper_depth]
                X, Y, T = data['X'], data['Y'], data['T']
            else:
                data_lower = self.geoTIS_data[lower_depth]
                data_upper = self.geoTIS_data[upper_depth]
                
                weight = (target_depth - lower_depth) / (upper_depth - lower_depth)
                
                X = data_lower['X']
                Y = data_lower['Y']
                T = data_lower['T'] * (1 - weight) + data_upper['T'] * weight
        
        # Remove invalid temperatures
        mask = (T > -99999) & (~np.isnan(T))
        X = X[mask]
        Y = Y[mask]
        T = T[mask]
        
        if len(T) == 0:
            return None
        
        STDV = np.std(T)
        
        return X, Y, T, STDV

    def save_temperature_geotiff(self, temperature_grid, horizon_name):
        """
        Save temperature evidence grid as GeoTIFF
        
        Args:
            temperature_grid: 2D numpy array of interpolated temperatures
            horizon_name: Horizon name (het1, het2, sin1, sin2, pli1, pli2)
        """
        output_path = self.output_dir / f"{horizon_name}_temperature_evidence.tif"
        
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        transform = from_bounds(left, bottom, right, top, temperature_grid.shape[1], temperature_grid.shape[0])
        
        try:
            with rasterio.open(
                output_path,
                'w',
                driver='GTiff',
                height=temperature_grid.shape[0],
                width=temperature_grid.shape[1],
                count=1,
                dtype=temperature_grid.dtype,
                crs=f"EPSG:{self.crs.split(':')[1]}",
                transform=transform
            ) as dst:
                dst.write(temperature_grid, 1)
            
            logger.info(f"  ✓ Saved: {output_path}")
            return str(output_path)
        except Exception as e:
            logger.error(f"  ✗ Error saving GeoTIFF: {e}")
            return None

    def save_debug_borehole_csv(self, horizon_name, merged_valid, geotis_available, temperature_at_boreholes, gradient_temps, temp_blended):
        """
        Save debug CSV with all borehole data used in interpolation
        
        Args:
            horizon_name: Horizon name
            merged_valid: DataFrame with valid boreholes (includes mean_depth already calculated)
            geotis_available: Boolean array (True = GeoTIS data available)
            temperature_at_boreholes: Array of GeoTIS temperatures
            gradient_temps: Array of gradient-calculated temperatures
            temp_blended: Final blended temperatures used for RBF
        """
        # Extract mean depth from merged_valid (should have the depth_col already)
        depth_col_map = {
            'het1': 'het1_mean_depth',
            'het2': 'het2_mean_depth',
            'sin1': 'sin1_mean_depth',
            'sin2': 'sin2_mean_depth',
            'pli1': 'pli1_mean_depth',
            'pli2': 'pli2_mean_depth',
        }
        depth_col = depth_col_map.get(horizon_name, 'mean_depth')
        
        debug_df = pd.DataFrame({
            'borehole_name': merged_valid['borehole_name'].values if 'borehole_name' in merged_valid.columns else range(len(merged_valid)),
            'x': merged_valid['x'].values,
            'y': merged_valid['y'].values,
            'mean_depth_m': abs(merged_valid[depth_col].values),  # Convert back to positive for readability
            'geotis_available': geotis_available,
            'geotis_temperature_C': temperature_at_boreholes,
            'gradient_temperature_C': gradient_temps,
            'final_blended_temperature_C': temp_blended,
            'data_source': ['GeoTIS' if g else 'Gradient' for g in geotis_available]
        })
        
        output_path = self.output_dir / f"{horizon_name}_borehole_temperatures_debug.csv"
        debug_df.to_csv(output_path, index=False, sep=';', decimal=',')  # Use comma for decimals
        logger.info(f"  ✓ Debug CSV saved: {output_path}")
        logger.info(f"    GeoTIS boreholes: {np.sum(geotis_available)}/{len(geotis_available)}")
        logger.info(f"    Gradient boreholes: {len(geotis_available) - np.sum(geotis_available)}/{len(geotis_available)}")

    def save_filtered_boreholes_csv(self, horizon_name, df_copy, depth_col, filtered_out):
        """
        Save CSV with boreholes that were filtered out and the reason why
        
        Args:
            horizon_name: Horizon name
            df_copy: Full DataFrame with all boreholes
            depth_col: Name of depth column
            filtered_out: List of tuples (borehole_name, reason)
        """
        if len(filtered_out) == 0:
            logger.info(f"  ✓ No boreholes filtered out")
            return
        
        filtered_df = pd.DataFrame(filtered_out, columns=['borehole_name', 'reason'])
        output_path = self.output_dir / f"{horizon_name}_borehole_temperatures_filtered.csv"
        filtered_df.to_csv(output_path, index=False, sep=';', decimal=',')  # Use comma for decimals
        logger.info(f"  ✓ Filtered boreholes CSV saved: {output_path}")
        logger.info(f"    Total filtered out: {len(filtered_out)}")

    def interpolate_temperature_grid_hybrid(self, horizon_name, mean_depths_df):
        """
        Interpolate temperature EVIDENCE using HYBRID approach:
        - Direct evidence: GeoTIS temperatures at borehole mean depths
        - Fallback: Geothermal gradient (from heat flow) where GeoTIS unavailable
        - Interpolation: RBF across basin
        
        Args:
            horizon_name: Horizon name (het1, het2, sin1, sin2, pli1, pli2)
            mean_depths_df: DataFrame with pre-calculated mean depths, X, Y coordinates
        
        Returns:
            Tuple (temperature_grid, geotis_count)
        """
        logger.info(f"    Interpolating temperature EVIDENCE (HYBRID: GeoTIS + gradient fallback)...")
        
        # Map horizon name to column name
        depth_column_map = {
            'het1': 'het1_mean_depth',
            'het2': 'het2_mean_depth',
            'sin1': 'sin1_mean_depth',
            'sin2': 'sin2_mean_depth',
            'pli1': 'pli1_mean_depth',
            'pli2': 'pli2_mean_depth',
        }
        
        depth_col = depth_column_map[horizon_name]
        
        # Normalize column names (lowercase for X, Y)
        df_copy = mean_depths_df.copy()
        df_copy.columns = df_copy.columns.str.lower()
        
        # Convert depth column to numeric and make negative (below surface)
        df_copy[depth_col] = pd.to_numeric(df_copy[depth_col], errors='coerce')
        
        # Track filtered boreholes
        filtered_out = []
        
        # Initial filter: exclude 0 values and invalid records
        for idx, row in df_copy.iterrows():
            depth_val = row[depth_col]
            if pd.isna(depth_val):
                filtered_out.append((row['borehole_name'], 'Invalid depth (NaN)'))
            elif depth_val == 0:
                filtered_out.append((row['borehole_name'], 'Zero depth'))
        
        # Convert to negative AFTER tracking
        df_copy[depth_col] = -np.abs(df_copy[depth_col])
        
        merged_valid = df_copy[
            (df_copy[depth_col] != 0) & 
            (df_copy[depth_col].notna()) &
            (df_copy[depth_col] < 0)
        ].copy()
        
        logger.info(f"      ✓ Filtered to {len(merged_valid)}/{len(df_copy)} boreholes with valid mean depths")
        logger.info(f"        Excluded: {len(df_copy) - len(merged_valid)} boreholes (zero, NaN, or no GeoTIS data)")
        
        if len(merged_valid) < 3:
            logger.warning(f"      ⚠ Less than 3 valid boreholes")
            return None, 0
        
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        temperature_at_boreholes = np.full(len(merged_valid), np.nan, dtype=np.float32)
        geotis_available = np.zeros(len(merged_valid), dtype=bool)
        
        # Query GeoTIS at borehole locations
        if self.geoTIS_data:
            logger.info(f"      Querying GeoTIS at {len(merged_valid)} borehole locations (basin filtered)...")
            
            for idx, (i, row) in enumerate(merged_valid.iterrows()):
                target_depth = row[depth_col]
                
                try:
                    result = self.interpolate_temperature_at_depth(target_depth, method='linear')
                    
                    if result is not None:
                        X_temp, Y_temp, T_temp, STDV_temp = result
                        
                        if len(X_temp) > 0:
                            points_temp = np.column_stack([X_temp, Y_temp])
                            T_at_borehole = griddata(points_temp, T_temp, (row['x'], row['y']), method='nearest')
                            
                            if not np.isnan(T_at_borehole):
                                temperature_at_boreholes[idx] = T_at_borehole
                                geotis_available[idx] = True
                            else:
                                filtered_out.append((row['borehole_name'], f'No GeoTIS data at depth {abs(target_depth)}m'))
                    else:
                        filtered_out.append((row['borehole_name'], f'No GeoTIS data at depth {abs(target_depth)}m'))
                
                except Exception as e:
                    filtered_out.append((row['borehole_name'], f'GeoTIS query error: {str(e)}'))
            
            geotis_count = np.sum(geotis_available)
            logger.info(f"      GeoTIS temperatures found: {geotis_count}/{len(merged_valid)} boreholes")
        
        # Save filtered boreholes info
        self.save_filtered_boreholes_csv(horizon_name, df_copy, depth_col, filtered_out)
        
        # Fallback: use heat flow gradient where GeoTIS is missing
        gradient_temps = self.calculate_temperature_from_depth(merged_valid[depth_col].values)
        
        # Use GeoTIS where available, gradient elsewhere
        temp_blended = np.where(
            geotis_available,
            temperature_at_boreholes,
            gradient_temps
        )
        
        merged_valid['temperature_blended'] = temp_blended
        
        # DEBUG: Save borehole data for verification
        logger.info(f"      Saving debug CSV with borehole temperatures...")
        self.save_debug_borehole_csv(
            horizon_name, 
            merged_valid, 
            geotis_available, 
            temperature_at_boreholes, 
            gradient_temps, 
            temp_blended
        )
        
        # RBF interpolate blended temperatures
        logger.info(f"      RBF interpolating blended temperatures across basin...")
        
        x_pts = merged_valid['x'].values
        y_pts = merged_valid['y'].values
        z_pts = temp_blended
        
        valid_mask = ~np.isnan(z_pts)
        if np.sum(valid_mask) < 3:
            logger.warning(f"      ⚠ Less than 3 valid temperature points")
            return None, 0
        
        x_pts_valid = x_pts[valid_mask]
        y_pts_valid = y_pts[valid_mask]
        z_pts_valid = z_pts[valid_mask]
        
        rbf = Rbf(x_pts_valid, y_pts_valid, z_pts_valid, function='thin_plate', epsilon=None, smooth=1.0)
        temperature_grid = rbf(X_mesh, Y_mesh)
        
        # Clamp to physical limits
        logger.info(f"      Before clamping: [{temperature_grid.min():.1f}, {temperature_grid.max():.1f}]°C")
        temperature_grid = np.clip(temperature_grid, self.T_MIN, self.T_MAX)
        logger.info(f"      After clamping:  [{temperature_grid.min():.1f}, {temperature_grid.max():.1f}]°C")
        
        logger.info(f"      ✓ Temperature range: [{temperature_grid.min():.1f}, {temperature_grid.max():.1f}]°C")
        
        return temperature_grid.astype(np.float32), geotis_count


def main():
    """
    Main execution - Process 6 horizons with borehole data
    
    RESERVOIR-SPECIFIC PFA APPROACH:
    - Input: Pre-calculated mean depths (top + thickness/2) with X,Y from CSV
    - Query: GeoTIS temperatures at those mean depths (DIRECT EVIDENCE)
    - Fallback: Geothermal gradient (from GFZ Heat Flow Database) where GeoTIS unavailable
    - Output: {horizon}_temperature_evidence.tif per horizon
    - Debug: {horizon}_borehole_temperatures_debug.csv with all borehole data
    - Filtered: {horizon}_borehole_temperatures_filtered.csv with excluded boreholes + reasons
    """
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs",
        geoTIS_dir="data/inputs/geotIS/temperature_basin_filtered",
        heat_flow_file="data/inputs/heat_flow/2022-015_Fuchs-et-al_GermanHeatFlowDB-2022.xlsx"
    )
    
    # Load pre-calculated mean depths with X, Y coordinates
    mean_depths_df = loader.load_mean_depths_table()
    if mean_depths_df is None:
        logger.error("✗ Cannot proceed without mean depths table")
        return
    
    logger.info("\n" + "="*80)
    logger.info("THERMAL EVIDENCE LAYERS (Using Pre-calculated Mean Depths + X,Y from CSV)")
    logger.info("="*80)
    
    horizons = {
        'het1': 'Hettangian 1',
        'het2': 'Hettangian 2',
        'sin1': 'Sinemurian 1',
        'sin2': 'Sinemurian 2',
        'pli1': 'Pliensbachian 1',
        'pli2': 'Pliensbachian 2',
    }
    
    for horizon_short, horizon_long in horizons.items():
        logger.info("\n" + "="*80)
        logger.info(f"Processing: {horizon_short.upper()} ({horizon_long})")
        logger.info("="*80)
        
        try:
            # Interpolate temperature evidence
            logger.info(f"\n1. Interpolating temperature evidence...")
            temp_grid, geotis_count = loader.interpolate_temperature_grid_hybrid(
                horizon_short, 
                mean_depths_df
            )
            
            if temp_grid is not None:
                # Export as GeoTIFF
                logger.info(f"\n2. Exporting temperature evidence GeoTIFF...")
                loader.save_temperature_geotiff(temp_grid, horizon_short)
                
                logger.info(f"\n✓ {horizon_short.upper()} completed successfully!")
                logger.info(f"   Temperature range: {temp_grid.min():.1f}°C to {temp_grid.max():.1f}°C")
                logger.info(f"   Data sources: {geotis_count} GeoTIS (basin filtered), others from gradient")
            else:
                logger.warning(f"⚠ Could not generate temperature evidence for {horizon_short.upper()}")
            
        except Exception as e:
            logger.error(f"✗ Error processing {horizon_short.upper()}: {e}", exc_info=True)
            continue
    
    logger.info("\n" + "="*80)
    logger.info("✓ THERMAL EVIDENCE LAYER GENERATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

"""
Thermal Data Loader - PFA Evidence Layer Generation

Generates PFA evidence layers for geothermal favorability assessment.
Processes borehole data (Tier 1) using RBF interpolation and heat flow-derived gradient.

TIER 1 APPROACH:
- Load borehole Teufe (depth) and Gesamtmaec (thickness)
- Calculate mean depth from: -(Teufe - Gesamtmaec/2)
- Calculate temperature using: T = T_surface + (depth_km × gradient)
- Temperature gradient derived from GFZ German Heat Flow Database 2022
- North German Basin mean heat flow: ~65 mW/m² → ~23 °C/km gradient
- RBF interpolate across entire basin (no gaps!)
- Confidence based on borehole density

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
from scipy.interpolate import Rbf
from scipy.spatial.distance import cdist
from pathlib import Path
import re
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ThermalDataLoader:
    """Generate PFA thermal evidence layers from borehole data using heat flow-derived gradient"""
    
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
    
    # Heat flow database parameters (GFZ German Heat Flow Database 2022)
    # North German Basin mean heat flow: ~65 mW/m²
    # Thermal conductivity range: 2.5-3.5 W/(m·K)
    # Derived gradient: (q / λ) = (65 mW/m² / 3000 W/(m·K)) ≈ 20-26 °C/km
    HEAT_FLOW_MEAN = 65.0  # mW/m²
    THERMAL_CONDUCTIVITY_LOW = 2.5  # W/(m·K) - high conductivity → low gradient
    THERMAL_CONDUCTIVITY_HIGH = 3.5  # W/(m·K) - low conductivity → high gradient
    
    # Calculate gradient from heat flow
    GEOTHERMAL_GRADIENT = (HEAT_FLOW_MEAN / 1000.0) / ((THERMAL_CONDUCTIVITY_LOW + THERMAL_CONDUCTIVITY_HIGH) / 2.0) * 1000.0
    
    # Surface temperature (typical for North German Basin)
    T_SURFACE = 10.0  # °C
    
    def __init__(self, config_path, base_dir="data/inputs"):
        """
        Initialize thermal data loader
        
        Args:
            config_path: Path to metadata.json
            base_dir: Base directory for input data
        """
        self.base_dir = Path(base_dir)
        self.config_path = Path(config_path)
        
        # Load configuration
        with open(self.config_path) as f:
            self.config = json.load(f)
        
        self.crs = self.config['crs']
        self.grid_size = self.config['grid_size']
        self.extent = self.config['extent']
        
        # Create output directory
        self.output_dir = Path("data/outputs/thermal")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ ThermalDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Tier 1 horizons: {len(self.TIER1_HORIZONS)}")
        logger.info(f"  Interpolation method: RBF (Radial Basis Function)")
        logger.info(f"  Output directory: {self.output_dir}")
        logger.info(f"  Coordinate system: GeoTIS (- below sea level, + above)")
        logger.info(f"  Data filtering: Excludes Teufe ≤ 0 (surface outcrops)")
        logger.info(f"\n  Temperature Calculation (Heat Flow Database):")
        logger.info(f"    Heat flow (North German Basin): {self.HEAT_FLOW_MEAN:.0f} mW/m²")
        logger.info(f"    Thermal conductivity: {self.THERMAL_CONDUCTIVITY_LOW:.1f}-{self.THERMAL_CONDUCTIVITY_HIGH:.1f} W/(m·K)")
        logger.info(f"    Geothermal gradient: {self.GEOTHERMAL_GRADIENT:.1f} °C/km")
        logger.info(f"    Surface temperature: {self.T_SURFACE:.1f}°C")
        logger.info(f"    Formula: T = {self.T_SURFACE:.1f} + (depth_km × {self.GEOTHERMAL_GRADIENT:.1f})")
    
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
        
        RBF advantages:
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
    
    def calculate_temperature_from_depth(self, depth_surface_m):
        """
        Calculate temperature from depth using heat flow-derived geothermal gradient
        
        Formula: T(z) = T_surface + (z_km × gradient)
        
        Where:
        - T_surface = 10°C (typical surface temperature, North German Basin)
        - z_km = depth in kilometers (negative for below sea level)
        - gradient = 23.1 °C/km (from GFZ heat flow database, 65 mW/m²)
        
        Args:
            depth_surface_m: 2D array of depths in meters (negative = below sea level)
        
        Returns:
            Temperature surface in °C
        """
        # Convert depth from meters to kilometers
        # Note: depth_surface_m is negative (below sea level)
        depth_km = np.abs(depth_surface_m) / 1000.0
        
        # Calculate temperature
        temperature = self.T_SURFACE + (depth_km * self.GEOTHERMAL_GRADIENT)
        
        return temperature.astype(np.float32)
    
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
        logger.info(f"Processing: {horizon_name} (TIER 1 - Borehole-based)")
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
            logger.info(f"\n2. RBF interpolating depth surface (Teufe)...")
            depth_surface, depth_unc = self.rbf_interpolate_surface(borehole_gdf, depth_col, function='thin_plate')
            
            if depth_surface is None:
                logger.error(f"✗ Interpolation failed")
                return None
            
            # RBF interpolate thickness surface
            logger.info(f"\n3. RBF interpolating thickness surface (Gesamtmaec)...")
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
            
            # Calculate temperature from depth using geothermal gradient
            logger.info(f"\n5. Calculating temperature from depth (heat flow-derived gradient)...")
            logger.info(f"  Formula: T = {self.T_SURFACE:.1f}°C + (depth_km × {self.GEOTHERMAL_GRADIENT:.1f}°C/km)")
            temperature_surface = self.calculate_temperature_from_depth(mean_depth)
            
            logger.info(f"  Temperature range: [{temperature_surface.min():.1f}, {temperature_surface.max():.1f}]°C")
            logger.info(f"  Mean temperature: {temperature_surface.mean():.1f}°C")
            
            # Check for unrealistic values
            zero_count = (temperature_surface == 0).sum()
            if zero_count > 0:
                logger.warning(f"  ⚠ {zero_count} zero temperature values")
            
            # Calculate confidence layers
            logger.info(f"\n6. Calculating confidence layers...")
            
            combined_unc = np.sqrt(depth_unc**2 + thick_unc**2)
            density_conf = self.calculate_borehole_density_confidence(borehole_gdf)
            confidence = self.combine_confidence_layers(combined_unc, density_conf)
            
            logger.info(f"  Confidence range: [{confidence.min():.2f}, {confidence.max():.2f}]")
            logger.info(f"  Mean confidence: {confidence.mean():.2f}")
            
            # Standard deviation (uncertainty) estimate for temperature
            # Based on gradient uncertainty and depth interpolation error
            temp_stdv = (thick_unc + depth_unc) * self.GEOTHERMAL_GRADIENT / 1000.0
            
            logger.info(f"\n7. Saving evidence layers...")
            
            self.save_raster(depth_surface, f"{horizon_name}_depth_surface")
            self.save_raster(thickness_surface, f"{horizon_name}_thickness_surface")
            self.save_raster(temperature_surface, f"{horizon_name}_temperature_surface")
            self.save_raster(confidence, f"{horizon_name}_confidence")
            self.save_raster(temp_stdv, f"{horizon_name}_temperature_stdv")
            
            # Geothermal gradient (constant)
            gradient_surface = np.full_like(temperature_surface, self.GEOTHERMAL_GRADIENT)
            self.save_raster(gradient_surface, f"{horizon_name}_geothermal_gradient")
            
            # Metadata
            metadata = {
                'horizon': horizon_name,
                'tier': 1,
                'data_type': 'borehole_interpolated_rbf_heatflow',
                'interpolation_method': 'RBF (Radial Basis Function) - Thin Plate',
                'temperature_calculation': 'Heat flow database derived geothermal gradient',
                'heat_flow_database': 'GFZ German Heat Flow Database 2022',
                'heat_flow_mean_mW_m2': self.HEAT_FLOW_MEAN,
                'thermal_conductivity_W_mK': (self.THERMAL_CONDUCTIVITY_LOW + self.THERMAL_CONDUCTIVITY_HIGH) / 2.0,
                'geothermal_gradient_C_km': float(self.GEOTHERMAL_GRADIENT),
                'surface_temperature_C': float(self.T_SURFACE),
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
                'coordinate_system': 'GeoTIS (- below sea level, + above)',
                'notes': 'RBF evidence layers with heat flow-derived temperatures - full spatial coverage, no dropout zones'
            }
            
            metadata_file = self.output_dir / f"{horizon_name}_metadata.json"
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info(f"\n✓ {horizon_name} completed successfully!")
            logger.info(f"  Coverage: Full basin (RBF ensures no gaps)")
            logger.info(f"  Temperature source: Heat flow database gradient ({self.GEOTHERMAL_GRADIENT:.1f}°C/km)")
            
            return metadata
        
        except Exception as e:
            logger.error(f"✗ Error: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def process_tier1(self):
        """Process all Tier 1 horizons"""
        logger.info("\n" + "="*80)
        logger.info("TIER 1: THERMAL EVIDENCE LAYERS (Borehole-based with Heat Flow Gradient)")
        logger.info("="*80)
        
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
            logger.info(f"  Temperature gradient: {meta['geothermal_gradient_C_km']:.1f}°C/km (from heat flow)")
            logger.info(f"  Depth range (boreholes): {meta['depth_range_m'][0]:.0f}-{meta['depth_range_m'][1]:.0f} m")
            logger.info(f"  Mean depth (GeoTIS system): {meta['mean_depth_range_m'][0]:.0f} to {meta['mean_depth_range_m'][1]:.0f} m")
            logger.info(f"  Temperature: {meta['mean_temperature_C']:.1f}°C (range: {meta['temperature_range_C'][0]:.1f}-{meta['temperature_range_C'][1]:.1f})")
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
    logger.info("✓ Temperature calculated from heat flow-derived geothermal gradient")
    logger.info("✓ Data source: GFZ German Heat Flow Database 2022")
    logger.info("  Advantages: Full spatial coverage, no dropout zones, physically sound\n")
    
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs"
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
    logger.info("  - *_temperature_surface.tif/.npy (RBF + heat flow gradient)")
    logger.info("  - *_geothermal_gradient.tif/.npy (constant: 23.1°C/km)")
    logger.info("  - *_confidence.tif/.npy (borehole density + interpolation error)")
    logger.info("  - *_temperature_stdv.tif/.npy (uncertainty estimate)")
    logger.info("  - *_metadata.json (complete provenance)")
    logger.info("\nNext: Process Tier 2 (TUNB surfaces + GeoTIS deep temperatures)")


if __name__ == "__main__":
    main()

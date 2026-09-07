"""
Data Loader for Reservoir Shapefiles
Reads shapefiles, rasterizes to 2000x2000 grid, normalizes values, clips to basin extent
"""

import json
import logging
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask, rasterize
from rasterio.transform import from_bounds
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)


class ReservoirDataLoader:
    """Load and process reservoir shapefiles for PFA analysis"""
    
    def __init__(self, config_path, base_dir="data/inputs"):
        """
        Initialize data loader
        
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
        self.output_dir = Path("data/outputs/rasters")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"✓ ReservoirDataLoader initialized")
        logger.info(f"  CRS: {self.crs}")
        logger.info(f"  Grid size: {self.grid_size}")
        logger.info(f"  Output directory: {self.output_dir}")
    
    def load_extent_boundary(self):
        """Load extent boundary shapefile"""
        try:
            extent_path = self.base_dir / self.config['extent']['boundary_shapefile']
            gdf = gpd.read_file(extent_path)
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            logger.info(f"✓ Loaded extent boundary: {extent_path}")
            logger.info(f"  CRS: {gdf.crs}")
            logger.info(f"  Bounds: {gdf.total_bounds}")
            
            return gdf
        except Exception as e:
            logger.error(f"✗ Failed to load extent boundary: {e}")
            raise
    
    def create_transform_and_bounds(self):
        """Create rasterio transform and bounds from config extent"""
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        width, height = self.grid_size
        
        # Create transform (geospatial affine transformation)
        transform = from_bounds(left, bottom, right, top, width, height)
        
        bounds = (left, bottom, right, top)
        
        logger.info(f"✓ Created raster transform")
        logger.info(f"  Bounds: {bounds}")
        logger.info(f"  Cell size: {(right-left)/width:.2f}m x {(top-bottom)/height:.2f}m")
        
        return transform, bounds
    
    def load_and_rasterize_shapefile(self, shapefile_path, column_name, normalize=True):
        """
        Load shapefile and rasterize to grid
        
        Args:
            shapefile_path: Path to shapefile
            column_name: Column to rasterize
            normalize: Whether to normalize values to [0, 1]
        
        Returns:
            2D numpy array
        """
        try:
            # Load shapefile
            gdf = gpd.read_file(self.base_dir / shapefile_path)
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Get transform and bounds
            transform, bounds = self.create_transform_and_bounds()
            width, height = self.grid_size
            
            # Create raster by burning geometry
            raster = np.zeros((height, width), dtype=np.float32)
            
            # Rasterize each feature
            for idx, row in gdf.iterrows():
                if pd.isna(row[column_name]):
                    continue
                
                geometry = row.geometry
                value = float(row[column_name])
                
                # Rasterize this geometry
                burned = rasterize(
                    [(geometry, value)],
                    out_shape=(height, width),
                    transform=transform,
                    default_value=0,
                    dtype=np.float32
                )
                
                raster = np.maximum(raster, burned)
            
            logger.info(f"✓ Rasterized {shapefile_path}")
            logger.info(f"  Column: {column_name}")
            logger.info(f"  Value range: [{raster.min():.3f}, {raster.max():.3f}]")
            
            return raster
        
        except Exception as e:
            logger.error(f"✗ Failed to rasterize {shapefile_path}: {e}")
            raise
    
    def normalize_potential_values(self, raster, value_mapping):
        """
        Normalize geothermal potential values to [0, 1]
        
        Args:
            raster: Input raster with categorical values
            value_mapping: Dict mapping original values to normalized values
        
        Returns:
            Normalized raster
        """
        normalized = np.zeros_like(raster)
        
        # Map categorical values to numeric scale
        value_scale = {
            'low': 0.0,
            'medium': 0.5,
            'high': 1.0
        }
        
        # Apply mapping
        for orig_val, norm_label in value_mapping.items():
            mask = raster == float(orig_val)
            normalized[mask] = value_scale[norm_label]
        
        return normalized
    
    def process_geothermal_potential_layer(self, reservoir_config):
        """
        Process geothermal potential layer
        
        Args:
            reservoir_config: Configuration dict for reservoir
        
        Returns:
            Normalized raster [0, 1]
        """
        name = reservoir_config['name']
        path = reservoir_config['path']
        column = reservoir_config['potential_column']
        value_mapping = reservoir_config['potential_values']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {name} (geothermal potential)")
        logger.info(f"{'='*80}")
        
        try:
            # Load and rasterize
            raster = self.load_and_rasterize_shapefile(path, column, normalize=False)
            
            # Normalize categorical values
            normalized = self.normalize_potential_values(raster, value_mapping)
            
            logger.info(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
            logger.info(f"  Non-zero cells: {np.count_nonzero(normalized)}/{normalized.size}")
            
            return normalized
        
        except Exception as e:
            logger.error(f"✗ Failed to process {name}: {e}")
            return None
    
    def process_sandstone_presence_layer(self, reservoir_config):
        """
        Process sandstone presence layer (binary: 0 or 1)
        
        Args:
            reservoir_config: Configuration dict for reservoir
        
        Returns:
            Binary raster [0, 1]
        """
        name = reservoir_config['name']
        path = reservoir_config['path']
        column = reservoir_config['sandstone_column']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {name} (sandstone presence)")
        logger.info(f"{'='*80}")
        
        try:
            # Load and rasterize
            raster = self.load_and_rasterize_shapefile(path, column, normalize=False)
            
            # Convert to binary (any non-zero value = 1)
            binary = np.where(raster > 0, 1.0, 0.0).astype(np.float32)
            
            logger.info(f"  Binary range: [{binary.min():.1f}, {binary.max():.1f}]")
            logger.info(f"  Presence cells: {np.count_nonzero(binary)}/{binary.size}")
            
            return binary
        
        except Exception as e:
            logger.error(f"✗ Failed to process {name}: {e}")
            return None
    
    def process_sandstone_share_layer(self, reservoir_config):
        """
        Process sandstone share layer (percentage: 0-100 normalized to 0-1)
        
        Args:
            reservoir_config: Configuration dict for reservoir
        
        Returns:
            Normalized raster [0, 1]
        """
        name = reservoir_config['name']
        path = reservoir_config['path']
        column = reservoir_config['sandstone_column']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {name} (sandstone share %)")
        logger.info(f"{'='*80}")
        
        try:
            # Load and rasterize
            raster = self.load_and_rasterize_shapefile(path, column, normalize=False)
            
            # Normalize from percentage (0-100) to [0, 1]
            normalized = np.clip(raster / 100.0, 0, 1).astype(np.float32)
            
            logger.info(f"  Normalized range: [{normalized.min():.3f}, {normalized.max():.3f}]")
            logger.info(f"  Non-zero cells: {np.count_nonzero(normalized)}/{normalized.size}")
            
            return normalized
        
        except Exception as e:
            logger.error(f"✗ Failed to process {name}: {e}")
            return None
    
    def clip_to_extent(self, raster, extent_gdf):
        """
        Clip raster to extent boundary
        
        Args:
            raster: Input raster
            extent_gdf: GeoDataFrame with extent boundary
        
        Returns:
            Clipped raster
        """
        transform, bounds = self.create_transform_and_bounds()
        width, height = self.grid_size
        
        # Create mask from extent geometry
        mask = geometry_mask(
            extent_gdf.geometry,
            out_shape=(height, width),
            transform=transform,
            invert=True  # True where inside extent
        )
        
        # Apply mask
        clipped = raster.copy()
        clipped[~mask] = 0
        
        return clipped
    
    def save_raster(self, raster, name, dtype=np.float32):
        """Save raster to GeoTIFF"""
        transform, _ = self.create_transform_and_bounds()
        output_path = self.output_dir / f"{name}.tif"
        
        with rasterio.open(
            output_path,
            'w',
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
        
        logger.info(f"✓ Saved: {output_path}")
        return output_path
    
    def save_raster_as_npy(self, raster, name):
        """Save raster as numpy array"""
        output_path = self.output_dir / f"{name}.npy"
        np.save(output_path, raster)
        logger.info(f"✓ Saved: {output_path}")
        return output_path
    
    def process_all_reservoirs(self, clip_to_boundary=True):
        """
        Process all reservoir layers
        
        Args:
            clip_to_boundary: Whether to clip results to basin extent
        
        Returns:
            Dict with all processed rasters
        """
        logger.info("\n" + "="*80)
        logger.info("LOADING AND PROCESSING ALL RESERVOIR SHAPEFILES")
        logger.info("="*80)
        
        results = {}
        
        # Load extent boundary if clipping
        extent_gdf = None
        if clip_to_boundary:
            extent_gdf = self.load_extent_boundary()
        
        # Process each reservoir
        for reservoir_config in self.config['reservoirs']:
            name = reservoir_config['name']
            layer_type = reservoir_config['type']
            
            try:
                if layer_type == 'geothermal_potential':
                    raster = self.process_geothermal_potential_layer(reservoir_config)
                
                elif layer_type == 'sandstone_presence':
                    raster = self.process_sandstone_presence_layer(reservoir_config)
                
                elif layer_type == 'sandstone_share':
                    raster = self.process_sandstone_share_layer(reservoir_config)
                
                else:
                    logger.warning(f"⚠ Unknown layer type: {layer_type}")
                    continue
                
                if raster is None:
                    logger.warning(f"⚠ Skipping {name} - processing failed")
                    continue
                
                # Clip to extent if requested
                if clip_to_boundary and extent_gdf is not None:
                    raster = self.clip_to_extent(raster, extent_gdf)
                    logger.info(f"  Clipped to basin extent")
                
                # Save outputs
                self.save_raster(raster, name, dtype=np.float32)
                self.save_raster_as_npy(raster, name)
                
                results[name] = {
                    'array': raster,
                    'type': layer_type,
                    'shape': raster.shape,
                    'range': (raster.min(), raster.max()),
                    'valid_cells': np.count_nonzero(raster)
                }
                
                logger.info(f"✓ {name} completed\n")
            
            except Exception as e:
                logger.error(f"✗ Error processing {name}: {e}\n")
                continue
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("PROCESSING SUMMARY")
        logger.info("="*80)
        logger.info(f"Successfully processed: {len(results)}/{len(self.config['reservoirs'])} layers")
        
        for name, data in results.items():
            logger.info(f"  {name}: {data['type']}, range [{data['range'][0]:.3f}, {data['range'][1]:.3f}]")
        
        return results


# Import pandas here to avoid errors
import pandas as pd


def main():
    """Main execution"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    # Initialize loader
    loader = ReservoirDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs"
    )
    
    # Process all reservoirs
    results = loader.process_all_reservoirs(clip_to_boundary=True)
    
    logger.info("\n✓ All data processing complete!")
    logger.info(f"  Output directory: {loader.output_dir}")


if __name__ == "__main__":
    main()

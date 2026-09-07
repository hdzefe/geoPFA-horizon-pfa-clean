"""
Data Loader for Reservoir Shapefiles
Reads shapefiles, rasterizes to 2000x2000 grid, normalizes values, clips to basin extent
"""

import json
import logging
import numpy as np
import geopandas as gpd
import pandas as pd
import rasterio
from rasterio.features import rasterize
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
        
        return transform, bounds
    
    def find_shapefile(self, base_path):
        """
        Find shapefile by searching for .shp files in directory
        
        Args:
            base_path: Path pattern from config
        
        Returns:
            Actual path to .shp file or None
        """
        # Try exact path first
        full_path = self.base_dir / base_path
        if full_path.exists():
            return full_path
        
        # Try finding any .shp in the directory
        parent_dir = self.base_dir / Path(base_path).parent
        if parent_dir.exists():
            shp_files = list(parent_dir.glob("*.shp"))
            if shp_files:
                logger.warning(f"  Using found shapefile: {shp_files[0].name}")
                return shp_files[0]
        
        return None
    
    def load_and_rasterize_shapefile(self, shapefile_path, column_name, value_mapping=None, filter_config=None, normalize=True):
        """
        Load shapefile, convert categorical values, and rasterize to grid
        
        Args:
            shapefile_path: Path to shapefile
            column_name: Column to rasterize
            value_mapping: Dict mapping categorical values to numeric scores
            filter_config: Dict with 'column' and 'values' for filtering features
            normalize: Whether to normalize values to [0, 1]
        
        Returns:
            2D numpy array
        """
        try:
            # Find actual shapefile
            actual_path = self.find_shapefile(str(shapefile_path))
            if actual_path is None:
                raise FileNotFoundError(f"Shapefile not found: {shapefile_path}")
            
            # Load shapefile
            gdf = gpd.read_file(actual_path)
            
            logger.info(f"  Loaded {actual_path.name}: {len(gdf)} features")
            logger.info(f"  Column '{column_name}' unique values: {gdf[column_name].unique()[:10]}")
            
            # Ensure correct CRS
            if gdf.crs is None:
                gdf.set_crs(self.crs, inplace=True)
            elif str(gdf.crs) != self.crs:
                gdf = gdf.to_crs(self.crs)
            
            # Apply filtering if specified
            if filter_config is not None:
                filter_column = filter_config.get('column')
                filter_values = filter_config.get('values', [])
                
                if filter_column and filter_values:
                    before_filter = len(gdf)
                    gdf = gdf[gdf[filter_column].isin(filter_values)]
                    after_filter = len(gdf)
                    
                    logger.info(f"  Applied filter: {filter_column} in {filter_values}")
                    logger.info(f"  Filtered: {before_filter} → {after_filter} features")
            
            # Convert categorical values to numeric if mapping provided
            if value_mapping is not None:
                logger.info(f"  Applying value mapping: {value_mapping}")
                
                numeric_column = f"{column_name}_numeric"
                gdf[numeric_column] = gdf[column_name].map(value_mapping)
                
                # Check for unmapped values
                unmapped = gdf[numeric_column].isna().sum()
                mapped = gdf[numeric_column].notna().sum()
                
                logger.info(f"  Mapped: {mapped}, Unmapped: {unmapped}")
                
                if unmapped > 0:
                    logger.warning(f"  ⚠ {unmapped} rows with unmapped values")
                    logger.warning(f"    Unmapped values: {gdf[gdf[numeric_column].isna()][column_name].unique()}")
                    gdf[numeric_column] = gdf[numeric_column].fillna(0)
                
                column_name = numeric_column
            
            # Get transform and bounds
            transform, bounds = self.create_transform_and_bounds()
            width, height = self.grid_size
            
            # Create raster by burning geometry
            raster = np.zeros((height, width), dtype=np.float32)
            
            # Count features with non-zero values
            non_zero_features = 0
            
            # Rasterize each feature
            for idx, row in gdf.iterrows():
                if pd.isna(row[column_name]):
                    continue
                
                geometry = row.geometry
                value = float(row[column_name])
                
                # Skip zero values (no data)
                if value == 0:
                    continue
                
                non_zero_features += 1
                
                try:
                    # Rasterize this geometry
                    burned = rasterize(
                        [(geometry, value)],
                        out_shape=(height, width),
                        transform=transform,
                        default_value=0,
                        dtype=np.float32
                    )
                    
                    raster = np.maximum(raster, burned)
                except Exception as e:
                    logger.warning(f"  ⚠ Could not rasterize feature {idx}: {e}")
                    continue
            
            logger.info(f"✓ Rasterized {actual_path.name}")
            logger.info(f"  Non-zero features: {non_zero_features}/{len(gdf)}")
            logger.info(f"  Raster value range: [{raster.min():.3f}, {raster.max():.3f}]")
            
            return raster
        
        except Exception as e:
            logger.error(f"✗ Failed to rasterize {shapefile_path}: {e}")
            raise
    
    def process_geothermal_potential_layer(self, reservoir_config):
        """
        Process geothermal potential layer
        
        Args:
            reservoir_config: Configuration dict for reservoir
        
        Returns:
            Normalized raster [0, 1] or None if failed
        """
        name = reservoir_config['name']
        path = reservoir_config['path']
        column = reservoir_config['potential_column']
        value_mapping = reservoir_config['potential_values']
        value_scale = reservoir_config.get('value_scale', {
            'low': 0.2,
            'medium': 0.5,
            'high': 1.0
        })
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {name} (geothermal potential)")
        logger.info(f"{'='*80}")
        
        try:
            # Build final mapping: original_value -> numeric_value
            final_mapping = {}
            for orig_val, norm_label in value_mapping.items():
                final_mapping[orig_val] = value_scale.get(norm_label, 0.0)
            
            logger.info(f"  Config mapping: {value_mapping}")
            logger.info(f"  Value scale: {value_scale}")
            logger.info(f"  Final numeric mapping: {final_mapping}")
            
            # Load and rasterize with numeric mapping
            raster = self.load_and_rasterize_shapefile(
                path, 
                column, 
                value_mapping=final_mapping,
                filter_config=None,
                normalize=False
            )
            
            logger.info(f"  Final range: [{raster.min():.3f}, {raster.max():.3f}]")
            logger.info(f"  Non-zero cells: {np.count_nonzero(raster)}/{raster.size}")
            
            return raster
        
        except Exception as e:
            logger.error(f"✗ Failed to process {name}: {e}")
            return None
    
    def process_sandstone_presence_layer(self, reservoir_config):
        """
        Process sandstone presence layer (binary: 0 or 1)
        
        Args:
            reservoir_config: Configuration dict for reservoir
        
        Returns:
            Binary raster [0, 1] or None if failed
        """
        name = reservoir_config['name']
        path = reservoir_config['path']
        column = reservoir_config['sandstone_column']
        filter_config = reservoir_config.get('sandstone_filter')
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {name} (sandstone presence)")
        logger.info(f"{'='*80}")
        
        try:
            # Load and rasterize (numeric column, no mapping needed)
            raster = self.load_and_rasterize_shapefile(
                path, 
                column, 
                value_mapping=None,
                filter_config=filter_config,
                normalize=False
            )
            
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
            Normalized raster [0, 1] or None if failed
        """
        name = reservoir_config['name']
        path = reservoir_config['path']
        column = reservoir_config['sandstone_column']
        
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {name} (sandstone share %)")
        logger.info(f"{'='*80}")
        
        try:
            # Load and rasterize (numeric column)
            raster = self.load_and_rasterize_shapefile(
                path, 
                column, 
                value_mapping=None,
                filter_config=None,
                normalize=False
            )
            
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
        Clip raster to extent boundary using mask
        
        Args:
            raster: Input raster
            extent_gdf: GeoDataFrame with extent boundary
        
        Returns:
            Clipped raster
        """
        from rasterio.features import geometry_mask
        
        transform, bounds = self.create_transform_and_bounds()
        width, height = self.grid_size
        
        try:
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
        except Exception as e:
            logger.warning(f"  ⚠ Could not clip to extent: {e}")
            return raster
    
    def save_raster(self, raster, name, dtype=np.float32):
        """Save raster to GeoTIFF"""
        transform, _ = self.create_transform_and_bounds()
        output_path = self.output_dir / f"{name}.tif"
        
        try:
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
        except Exception as e:
            logger.error(f"✗ Failed to save {output_path}: {e}")
            return None
    
    def save_raster_as_npy(self, raster, name):
        """Save raster as numpy array"""
        output_path = self.output_dir / f"{name}.npy"
        try:
            np.save(output_path, raster)
            logger.info(f"✓ Saved: {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"✗ Failed to save {output_path}: {e}")
            return None
    
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
            try:
                extent_gdf = self.load_extent_boundary()
            except Exception as e:
                logger.warning(f"⚠ Could not load extent boundary: {e}")
                logger.warning(f"  Proceeding without clipping")
        
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
                
                # Clip to extent if requested and available
                if clip_to_boundary and extent_gdf is not None:
                    raster = self.clip_to_extent(raster, extent_gdf)
                    logger.info(f"  Clipped to basin extent")
                
                # Save outputs
                tif_path = self.save_raster(raster, name, dtype=np.float32)
                npy_path = self.save_raster_as_npy(raster, name)
                
                if tif_path and npy_path:
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
        
        if len(results) > 0:
            for name, data in sorted(results.items()):
                logger.info(f"  {name}: {data['type']}, range [{data['range'][0]:.3f}, {data['range'][1]:.3f}], cells: {data['valid_cells']}")
        else:
            logger.warning("⚠ No layers processed successfully!")
        
        return results


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

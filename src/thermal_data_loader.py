    def _export_geotiff(self, data: np.ndarray, filename: str):
        """Export temperature grid as GeoTIFF with proper georeferencing"""
        output_path = self.output_dir / filename
        
        if self.basin_bounds is None:
            logger.warning("  ⚠ Basin bounds not set, using identity transform")
            transform = Affine.identity()
        else:
            # Map grid pixels to real coordinates
            x_min, x_max, y_min, y_max = self.basin_bounds
            
            pixel_width = (x_max - x_min) / data.shape[1]
            pixel_height = (y_max - y_min) / data.shape[0]
            
            # Bottom-left corner (y_min) with positive pixel height
            transform = Affine.translation(x_min, y_min) * Affine.scale(pixel_width, pixel_height)
            
            logger.info(f"  Georeferencing: {x_min:.0f}-{x_max:.0f} / {y_min:.0f}-{y_max:.0f}")
        
        with rasterio.open(
            output_path, 'w',
            driver='GTiff',
            height=data.shape[0],
            width=data.shape[1],
            count=1,
            dtype=rasterio.float32,
            crs=self.crs,
            transform=transform
        ) as dst:
            dst.write(np.flipud(data.astype(rasterio.float32)), 1)
        
        logger.info(f"  ✓ Saved: {output_path}")

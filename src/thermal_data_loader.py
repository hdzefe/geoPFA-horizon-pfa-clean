    def interpolate_temperature_grid_hybrid(self, mean_depth_surface, borehole_gdf):
        """
        Interpolate temperature using HYBRID approach with spatial filtering
        Only uses GeoTIS points within North German Basin (prevents Molasse/Rhine contamination)
        
        ✅ FILTER: Only use boreholes with mean_depth < -100m (subsurface)
        Excludes near-surface/positive depths that create RBF artifacts
        """
        logger.info(f"    Interpolating temperature (HYBRID: GeoTIS spatially-filtered + gradient fallback)...")
        
        # ✅ FILTER: Keep only subsurface boreholes (mean_depth < -100m)
        borehole_gdf_subsurface = borehole_gdf[borehole_gdf['mean_depth'] < -100].copy()
        
        if len(borehole_gdf_subsurface) < 3:
            logger.warning(f"      ⚠ Less than 3 subsurface boreholes (mean_depth < -100m)")
            logger.info(f"      Using all boreholes (may have RBF artifacts from near-surface data)")
            borehole_gdf_subsurface = borehole_gdf.copy()
        else:
            logger.info(f"      ✓ Filtered to {len(borehole_gdf_subsurface)}/{len(borehole_gdf)} subsurface boreholes (mean_depth < -100m)")
        
        left = self.extent['left']
        right = self.extent['right']
        bottom = self.extent['bottom']
        top = self.extent['top']
        
        x_grid = np.linspace(left, right, self.grid_size[0])
        y_grid = np.linspace(bottom, top, self.grid_size[1])
        X_mesh, Y_mesh = np.meshgrid(x_grid, y_grid)
        
        temperature_at_boreholes = np.full(len(borehole_gdf_subsurface), np.nan, dtype=np.float32)
        geotis_available = np.zeros(len(borehole_gdf_subsurface), dtype=bool)
        
        # Query GeoTIS at borehole locations (spatially filtered)
        if self.geoTIS_data:
            logger.info(f"      Querying GeoTIS at {len(borehole_gdf_subsurface)} borehole locations (subsurface only, spatially filtered to basin)...")
            
            for idx, (i, row) in enumerate(borehole_gdf_subsurface.iterrows()):
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
            logger.info(f"      GeoTIS temperatures found (basin-filtered, subsurface): {geotis_count}/{len(borehole_gdf_subsurface)} boreholes")
        
        # Fallback: use heat flow gradient where GeoTIS is missing
        gradient_temps = self.calculate_temperature_from_depth(borehole_gdf_subsurface['mean_depth'].values)
        
        # Use GeoTIS where available, gradient elsewhere
        temp_blended = np.where(
            geotis_available,
            temperature_at_boreholes,
            gradient_temps
        )
        
        borehole_gdf_subsurface['temperature_blended'] = temp_blended
        
        # RBF interpolate blended temperatures
        logger.info(f"      RBF interpolating blended temperatures across basin (subsurface only)...")
        
        x_pts = borehole_gdf_subsurface['X'].values
        y_pts = borehole_gdf_subsurface['Y'].values
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
        logger.info(f"      Data sources: {geotis_count} GeoTIS (spatially filtered, subsurface), {len(borehole_gdf_subsurface) - geotis_count} gradient-derived")
        
        return temperature_grid.astype(np.float32), geotis_available

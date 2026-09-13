    def interpolate_temperature_grid_hybrid(self, horizon_name, mean_depths_df):
        """
        Interpolate temperature using HYBRID approach with pre-calculated mean depths and X,Y from CSV
        
        Args:
            horizon_name: Horizon name (het1, het2, sin1, sin2, pli1, pli2)
            mean_depths_df: DataFrame with pre-calculated mean depths, X, Y coordinates
        
        Returns:
            Tuple (temperature_grid, geotis_count)
        """
        logger.info(f"    Interpolating temperature (HYBRID: GeoTIS + gradient fallback)...")
        
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
        df_copy[depth_col] = -np.abs(df_copy[depth_col])  # Ensure negative values
        
        # Filter: exclude 0 values and invalid records
        merged_valid = df_copy[
            (df_copy[depth_col] != 0) & 
            (df_copy[depth_col].notna()) &
            (df_copy[depth_col] < 0)  # Negative depths only (below surface)
        ].copy()
        
        logger.info(f"      ✓ Filtered to {len(merged_valid)}/{len(df_copy)} boreholes with valid mean depths (excluding 0 values)")
        
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
                
                except Exception as e:
                    continue
            
            geotis_count = np.sum(geotis_available)
            logger.info(f"      GeoTIS temperatures found: {geotis_count}/{len(merged_valid)} boreholes")
        
        # Fallback: use heat flow gradient where GeoTIS is missing
        gradient_temps = self.calculate_temperature_from_depth(merged_valid[depth_col].values)
        
        # Use GeoTIS where available, gradient elsewhere
        temp_blended = np.where(
            geotis_available,
            temperature_at_boreholes,
            gradient_temps
        )
        
        merged_valid['temperature_blended'] = temp_blended
        
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

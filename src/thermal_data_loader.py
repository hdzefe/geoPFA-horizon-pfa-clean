    def interpolate_temperature_at_depth(self, z_horizon: np.ndarray) -> np.ndarray:
        """Query temperature at specific depths, prioritizing borehole data"""
        T_grid = np.full(z_horizon.shape, np.nan, dtype=np.float32)
        
        # Get grid bounds
        x_min, x_max, y_min, y_max = self.basin_bounds
        pixel_width = (x_max - x_min) / z_horizon.shape[1]
        pixel_height = (y_max - y_min) / z_horizon.shape[0]
        
        # Iterate over grid cells
        for i in range(z_horizon.shape[0]):
            for j in range(z_horizon.shape[1]):
                grid_x = x_min + j * pixel_width
                grid_y = y_min + i * pixel_height
                grid_depth = z_horizon[i, j]
                
                if np.isnan(grid_depth):
                    continue
                
                # **PRIORITY 1: Check if borehole exists at this grid location**
                borehole_T = self._get_temperature_from_borehole(grid_x, grid_y, grid_depth)
                if not np.isnan(borehole_T):
                    T_grid[i, j] = borehole_T
                    continue
                
                # **PRIORITY 2: Fallback to GeoTIS nearest-neighbor**
                geoTIS_T = self._interpolate_sparse_at_xy_geoTIS(grid_x, grid_y, grid_depth)
                if not np.isnan(geoTIS_T):
                    T_grid[i, j] = geoTIS_T
        
        return T_grid
    
    def _get_temperature_from_borehole(self, x: float, y: float, depth: float, radius: float = 500) -> float:
        """
        Get temperature from borehole if one exists near this location
        Uses geothermal gradient: T = T_surface + (depth / 1000) * gradient
        """
        if self.boreholes is None or self.heat_flow_data is None:
            return np.nan
        
        # Find boreholes within radius
        dist = np.sqrt((self.boreholes['x'].values - x)**2 + (self.boreholes['y'].values - y)**2)
        nearby_mask = dist < radius
        
        if not np.any(nearby_mask):
            return np.nan
        
        # Get nearest borehole
        nearest_idx = np.argmin(dist[nearby_mask])
        actual_idx = np.where(nearby_mask)[0][nearest_idx]
        borehole = self.boreholes.iloc[actual_idx]
        
        # Surface temperature (approx 10°C at surface)
        T_surface = 10.0
        
        # Use calculated geothermal gradient
        T = T_surface + (depth / 1000.0) * self.gradient_degC_per_km
        
        return float(T)
    
    def _interpolate_sparse_at_xy_geoTIS(self, x: float, y: float, depth: float, radius: float = 50000) -> float:
        """Find nearest GeoTIS temperature point within radius at specific depth"""
        # Get closest GeoTIS depth level
        available_depths = sorted(self.geoTIS_data.keys())
        
        if not available_depths:
            return np.nan
        
        # Find nearest depth
        depth_diffs = [abs(d - depth) for d in available_depths]
        nearest_depth_idx = np.argmin(depth_diffs)
        query_depth = available_depths[nearest_depth_idx]
        
        xyz = self.geoTIS_data[query_depth]
        if xyz is None or len(xyz) == 0:
            return np.nan
        
        X = xyz[:, 0]
        Y = xyz[:, 1]
        T = xyz[:, 2]
        
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        
        # Find nearest point within radius
        valid = dist < radius
        if not np.any(valid):
            return np.nan
        
        nearest_idx = np.argmin(dist[valid])
        actual_idx = np.where(valid)[0][nearest_idx]
        
        return float(T[actual_idx])

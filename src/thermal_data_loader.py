    def interpolate_temperature_at_depth(self, z_horizon: np.ndarray) -> np.ndarray:
        """Interpolate temperature at horizon depths using GeoTIS data"""
        T_grid = np.full(z_horizon.shape, np.nan, dtype=np.float32)
        
        # Iterate over grid cells
        for i in range(z_horizon.shape[0]):
            for j in range(z_horizon.shape[1]):
                grid_depth = z_horizon[i, j]
                
                if np.isnan(grid_depth):
                    continue
                
                # Get grid coordinates
                x_min, x_max, y_min, y_max = self.basin_bounds
                pixel_width = (x_max - x_min) / 2000
                pixel_height = (y_max - y_min) / 2000
                
                grid_x = x_min + j * pixel_width
                grid_y = y_min + i * pixel_height
                
                # Get GeoTIS temperature at this depth
                T_grid[i, j] = self._interpolate_sparse_at_xy(grid_x, grid_y, grid_depth)
        
        return T_grid
    
    def _interpolate_sparse_at_xy(self, x: float, y: float, depth: float) -> float:
        """Query GeoTIS temperature at (x,y) and depth"""
        # Find closest depth level in GeoTIS
        available_depths = sorted(self.geoTIS_data.keys())
        if not available_depths:
            return np.nan
        
        depth_diffs = [abs(d - depth) for d in available_depths]
        nearest_depth = available_depths[np.argmin(depth_diffs)]
        
        xyz = self.geoTIS_data[nearest_depth]
        if xyz is None or len(xyz) == 0:
            return np.nan
        
        # Nearest neighbor within 50km radius
        X = xyz[:, 0]
        Y = xyz[:, 1]
        T = xyz[:, 2]
        
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        valid = dist < 50000
        
        if not np.any(valid):
            return np.nan
        
        nearest_idx = np.argmin(dist[valid])
        actual_idx = np.where(valid)[0][nearest_idx]
        
        return float(T[actual_idx])

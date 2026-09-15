    def load_surface_from_ts_files(self, ts_files: list) -> np.ndarray:
        """Load GOCAD TS surface files and interpolate to grid"""
        logger.info(f"    Loading {len(ts_files)} TS files...")
        
        points = []
        values = []
        
        for ts_file in ts_files:
            try:
                with open(ts_file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                
                # Parse TS file - look for VRTX (vertex) lines
                lines = content.split('\n')
                for line in lines:
                    if line.startswith('VRTX'):
                        parts = line.split()
                        if len(parts) >= 5:
                            try:
                                x = float(parts[2])
                                y = float(parts[3])
                                z = float(parts[4])
                                points.append([x, y])
                                values.append(z)
                            except (ValueError, IndexError):
                                pass
            except Exception as e:
                logger.warning(f"    ⚠ Error reading {Path(ts_file).name}: {e}")
        
        if not points:
            logger.warning(f"    ⚠ No points found in TS files")
            return np.full_like(self.xv, np.nan)
        
        points = np.array(points)
        values = np.array(values)
        
        logger.info(f"    ✓ Extracted {len(points)} points from {len(ts_files)} surfaces")
        
        # DECIMATE: Keep only 10k evenly distributed points for efficiency
        if len(points) > 10000:
            sample_indices = np.random.choice(len(points), 10000, replace=False)
            points_decimated = points[sample_indices]
            values_decimated = values[sample_indices]
            logger.info(f"    → Decimated to {len(points_decimated)} points for efficiency")
        else:
            points_decimated = points
            values_decimated = values
        
        # Use griddata instead of RBF (much faster, less memory)
        from scipy.interpolate import griddata
        try:
            z_grid = griddata(points_decimated, values_decimated, (self.xv, self.yv), method='linear', fill_value=np.nan)
            
            valid = ~np.isnan(z_grid)
            if np.any(valid):
                logger.info(f"    ✓ Interpolated to grid: {np.nanmin(z_grid):.1f}m to {np.nanmax(z_grid):.1f}m")
            
            return z_grid
        except Exception as e:
            logger.warning(f"    ⚠ Interpolation failed: {e}")
            return np.full_like(self.xv, np.nan)
    
    def _interpolate_sparse_at_xy(self, xyz: np.ndarray, x: float, y: float, radius: float = 5000) -> float:
        """Nearest neighbor interpolate temperature at (x,y) from sparse (X,Y,T) points"""
        # Extract points within radius
        X = xyz[:, 0]
        Y = xyz[:, 1]
        T = xyz[:, 2]
        
        dist = np.sqrt((X - x)**2 + (Y - y)**2)
        mask = dist < radius
        
        if not np.any(mask):
            return np.nan
        
        # Find nearest point
        valid_dist = dist[mask]
        valid_idx = np.where(mask)[0]
        nearest_local_idx = np.argmin(valid_dist)
        nearest_idx = valid_idx[nearest_local_idx]
        
        return float(T[nearest_idx])

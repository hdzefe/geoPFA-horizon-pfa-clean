    def _interpolate_sparse_at_xy(self, xyz: np.ndarray, x: float, y: float, radius: float = 5000) -> float:
        """Find nearest temperature point within radius"""
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

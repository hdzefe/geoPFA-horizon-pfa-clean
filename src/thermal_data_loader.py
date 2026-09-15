    def _get_temperature_from_borehole(self, x: float, y: float, depth: float, radius: float = 500) -> float:
        """
        Get temperature from borehole if one exists near this location
        Uses geothermal gradient: T = T_surface + (depth / 1000) * gradient
        """
        if self.boreholes_gdf is None or self.heat_flow_stats is None:
            return np.nan
        
        # Find boreholes within radius
        dist = np.sqrt((self.boreholes_gdf['x'].values - x)**2 + (self.boreholes_gdf['y'].values - y)**2)
        nearby_mask = dist < radius
        
        if not np.any(nearby_mask):
            return np.nan
        
        # Surface temperature (approx 10°C at surface)
        T_surface = 10.0
        
        # Use calculated geothermal gradient
        T = T_surface + (depth / 1000.0) * self.gradient_degC_per_km
        
        return float(T)

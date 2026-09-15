    def process_horizon(self, horizon_name: str, top_surface_dir: str, base_surface_dir: str, fraction: float):
        """Process a single horizon"""
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing: {horizon_name.upper()}")
        logger.info(f"{'='*80}")
        
        # Load surfaces
        top_files = sorted(glob.glob(str(self.surfaces_dir / top_surface_dir / "*.ts")))
        base_files = sorted(glob.glob(str(self.surfaces_dir / base_surface_dir / "*.ts")))
        
        if not top_files or not base_files:
            logger.warning(f"  ⚠ Surface files not found")
            logger.warning(f"    Top: {self.surfaces_dir / top_surface_dir}")
            logger.warning(f"    Base: {self.surfaces_dir / base_surface_dir}")
            return
        
        logger.info(f"1. Loading surfaces...")
        z_top = self.load_surface_from_ts_files(top_files)
        z_base = self.load_surface_from_ts_files(base_files)
        
        # Interpolate horizon depth using fraction
        z_horizon = z_top + (z_base - z_top) * fraction
        
        valid_z = ~np.isnan(z_horizon)
        if not np.any(valid_z):
            logger.warning(f"  ⚠ No valid depth data")
            return
        
        logger.info(f"  ✓ Horizon depth range: {np.nanmin(z_horizon):.1f}m to {np.nanmax(z_horizon):.1f}m")
        
        # Query GeoTIS at horizon depth
        logger.info(f"2. Querying GeoTIS temperatures...")
        T_horizon = self.interpolate_temperature_at_depth(z_horizon)
        
        valid_T = ~np.isnan(T_horizon)
        if not np.any(valid_T):
            logger.warning(f"  ⚠ No valid temperature data")
            return
        
        logger.info(f"  ✓ Temperature range (raw): {np.nanmin(T_horizon):.1f}°C to {np.nanmax(T_horizon):.1f}°C")
        
        # CLIP to realistic range: 3-200°C
        T_clipped = np.clip(T_horizon, 3.0, 200.0)
        
        logger.info(f"  ✓ Temperature range (clipped): {np.nanmin(T_clipped):.1f}°C to {np.nanmax(T_clipped):.1f}°C")
        
        # Export
        logger.info(f"3. Exporting temperature evidence...")
        self._export_geotiff(T_clipped, f"{horizon_name}_temperature_evidence.tif")
        
        logger.info(f"✓ {horizon_name.upper()} completed successfully!")
        logger.info(f"  Final temperature range: {np.nanmin(T_clipped):.1f}°C to {np.nanmax(T_clipped):.1f}°C")

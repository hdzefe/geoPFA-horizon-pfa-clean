    def load_heat_flow_database(self):
        """Load GFZ German Heat Flow Database and calculate regional gradient"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GFZ GERMAN HEAT FLOW DATABASE 2022")
        logger.info("="*80)
        
        if not self.heat_flow_file.exists():
            logger.warning(f"⚠ Heat flow file not found: {self.heat_flow_file}")
            logger.warning(f"  Using default gradient: 23.1 °C/km")
            self.geothermal_gradient = 23.1
            return
        
        try:
            # Load Excel - Skip first 3 rows, row 4 becomes header
            df = pd.read_excel(self.heat_flow_file, sheet_name='data', skiprows=3, dtype=str)
            
            logger.info(f"✓ Loaded: {len(df):,} records from heat flow database")
            
            # Convert columns with European decimal format (comma = decimal separator)
            for col in ['lat', 'lng', 'q']:
                if col in df.columns:
                    df[col] = df[col].str.replace(',', '.', regex=False)
                    df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Filter valid data
            df_clean = df.dropna(subset=['lat', 'lng', 'q']).copy()
            df_clean = df_clean[(df_clean['q'] > 0) & (df_clean['q'] < 300)]
            
            logger.info(f"✓ Valid records: {len(df_clean):,} / {len(df):,}")
            
            logger.info(f"\n  Heat flow data (lat/lng) extent:")
            logger.info(f"    Latitude:  {df_clean['lat'].min():.4f}° to {df_clean['lat'].max():.4f}°")
            logger.info(f"    Longitude: {df_clean['lng'].min():.4f}° to {df_clean['lng'].max():.4f}°")
            
            # North German Basin extent in lat/lng (WGS84)
            # Approximate bounds: covers ~133,938 km²
            basin_lat_min, basin_lat_max = 50.8, 54.5
            basin_lng_min, basin_lng_max = 6.0, 14.0
            
            logger.info(f"\n  Basin extent (WGS84 lat/lng):")
            logger.info(f"    Latitude:  {basin_lat_min:.1f}° to {basin_lat_max:.1f}°")
            logger.info(f"    Longitude: {basin_lng_min:.1f}° to {basin_lng_max:.1f}°")
            
            # Filter to basin extent using lat/lng directly
            basin_mask = (df_clean['lat'] >= basin_lat_min) & (df_clean['lat'] <= basin_lat_max) & \
                        (df_clean['lng'] >= basin_lng_min) & (df_clean['lng'] <= basin_lng_max)
            
            df_basin = df_clean[basin_mask].copy()
            
            logger.info(f"\n✓ Measurements in North German Basin: {len(df_basin):,} / {len(df_clean):,}")
            
            if len(df_basin) > 0:
                q_basin = df_basin['q']
                mean_hf = q_basin.mean()
                
                logger.info(f"\n  ✓✓✓ BASIN HEAT FLOW STATISTICS ✓✓✓")
                logger.info(f"    Mean: {mean_hf:.1f} mW/m²")
                logger.info(f"    Median: {q_basin.median():.1f} mW/m²")
                logger.info(f"    Std Dev: {q_basin.std():.1f} mW/m²")
                logger.info(f"    Range: {q_basin.min():.1f} to {q_basin.max():.1f} mW/m²")
                
                # Calculate gradient from heat flow: dT/dz = q / λ (in °C/km)
                # q is in mW/m² = W/m² / 1000
                # λ in W/(m·K)
                self.geothermal_gradient = (mean_hf / 1000.0) / self.thermal_conductivity * 1000.0
                
                logger.info(f"\n  Calculated Geothermal Gradient (λ = {self.thermal_conductivity} W/(m·K)):")
                logger.info(f"    {self.geothermal_gradient:.2f} °C/km")
                logger.info(f"    (Based on basin-measured heat flow: {mean_hf:.1f} mW/m²)")
                
            else:
                # Use Germany-wide average if no basin data
                q_all = df_clean['q']
                mean_hf = q_all.mean()
                
                logger.warning(f"⚠ No heat flow measurements found in basin extent")
                logger.info(f"\n  Using Germany-wide average:")
                logger.info(f"    Mean: {mean_hf:.1f} mW/m²")
                logger.info(f"    Total measurements: {len(q_all)}")
                
                self.geothermal_gradient = (mean_hf / 1000.0) / self.thermal_conductivity * 1000.0
                
                logger.info(f"\n  Calculated Geothermal Gradient:")
                logger.info(f"    {self.geothermal_gradient:.2f} °C/km (Germany-wide)")
        
        except Exception as e:
            logger.warning(f"⚠ Error loading heat flow database: {e}")
            import traceback
            logger.warning(traceback.format_exc())
            logger.warning(f"  Using default gradient: 23.1 °C/km")
            self.geothermal_gradient = 23.1

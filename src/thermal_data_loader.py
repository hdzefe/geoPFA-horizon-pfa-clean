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
            # Read all columns as strings first to handle European decimal format
            df = pd.read_excel(self.heat_flow_file, sheet_name='data', skiprows=3, dtype=str)
            
            logger.info(f"✓ Loaded: {len(df):,} records from heat flow database")
            
            # Convert columns with European decimal format (comma = decimal separator)
            # lat, lng, q all use commas as decimal separators
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
            
            # North German Basin extent (Gauss-Krueger Zone 3)
            basin_xmin, basin_xmax = 3342763, 3921379
            basin_ymin, basin_ymax = 5650524, 6103144
            
            logger.info(f"\n  Basin extent (GK Zone 3):")
            logger.info(f"    X: {basin_xmin:,} to {basin_xmax:,}")
            logger.info(f"    Y: {basin_ymin:,} to {basin_ymax:,}")
            
            # Convert lat/lng to GK coordinates for filtering
            from pyproj import Transformer
            transformer = Transformer.from_crs("EPSG:4326", "EPSG:31467")
            
            # Test with first point
            test_lng = df_clean['lng'].iloc[0]
            test_lat = df_clean['lat'].iloc[0]
            logger.info(f"\n  TEST POINT (first measurement):")
            logger.info(f"    Input (lng, lat): {test_lng:.4f}, {test_lat:.4f}")
            
            test_x, test_y = transformer.transform(test_lng, test_lat)
            logger.info(f"    Transformed (X, Y): {test_x:,.0f}, {test_y:,.0f}")
            logger.info(f"    Expected range: X ~3.3-3.9M, Y ~5.6-6.1M")
            
            # Transform all points
            x_gk, y_gk = transformer.transform(df_clean['lng'].values, df_clean['lat'].values)
            
            logger.info(f"\n  Heat flow data extent (converted to GK coordinates):")
            logger.info(f"    X: {x_gk.min():,.0f} to {x_gk.max():,.0f}")
            logger.info(f"    Y: {y_gk.min():,.0f} to {y_gk.max():,.0f}")
            
            # Filter to basin extent
            basin_mask = (x_gk >= basin_xmin) & (x_gk <= basin_xmax) & \
                        (y_gk >= basin_ymin) & (y_gk <= basin_ymax)
            
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
                logger.warning(f"  Coordinate transformation may be incorrect")
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

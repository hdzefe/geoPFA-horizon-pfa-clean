    def load_geotis_temperatures(self):
        """Load GeoTIS temperature data from .data files (CSV format: X;Y;T;STDEV)"""
        logger.info("\n" + "="*80)
        logger.info("LOADING GEOTIS TEMPERATURE DATA (BASIN FILTERED)")
        logger.info("="*80)
        
        data_files = sorted(glob.glob(str(self.geotis_dir / "*.data")))
        
        if not data_files:
            logger.error(f"❌ No .data files found in {self.geotis_dir}")
            return False
        
        logger.info(f"✓ Found {len(data_files)} temperature depth files")
        
        try:
            for data_file in data_files:
                filename = Path(data_file).stem
                # Extract depth from filename (format: T2022_LIAG_AGEMAR_-5000)
                depth_str = filename.split('_')[-1]
                
                try:
                    depth_m = int(depth_str)
                except ValueError:
                    logger.warning(f"  ⚠ Could not parse depth from {filename}")
                    continue
                
                # Load CSV data file (X;Y;T;STDEV format)
                try:
                    df = pd.read_csv(data_file, delimiter=';', dtype={'X': float, 'Y': float, 'T': float})
                    
                    if df.empty or 'T' not in df.columns:
                        logger.warning(f"  ⚠ No temperature data in {filename}")
                        continue
                    
                    # Store sparse data as-is (X, Y, T coordinates)
                    # Will interpolate to grid during processing
                    self.T_stack[depth_m] = df[['X', 'Y', 'T']].values
                    self.levels_m_nhn.append(depth_m)
                    
                    T_vals = df['T'].values
                    valid_count = np.isfinite(T_vals).sum()
                    logger.info(f"  ✓ {filename}: depth={depth_m:+6d}m, "
                               f"points={len(df)}, "
                               f"T=[{np.nanmin(T_vals):6.1f}, {np.nanmax(T_vals):6.1f}]°C")
                
                except Exception as e:
                    logger.warning(f"  ⚠ Error reading {filename}: {e}")
                    continue
            
            self.levels_m_nhn.sort()
            
            if len(self.levels_m_nhn) == 0:
                logger.error(f"❌ No valid depth levels loaded")
                return False
            
            logger.info(f"\n✓ Loaded {len(self.levels_m_nhn)} depth levels")
            logger.info(f"  Depth range: {self.levels_m_nhn[0]:+6d}m to {self.levels_m_nhn[-1]:+6d}m")
            
            return True
        
        except Exception as e:
            logger.error(f"❌ Error loading GeoTIS data: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False
    
    def initialize_grid(self):
        """Initialize processing grid from borehole bounds"""
        logger.info("\n" + "="*80)
        logger.info("INITIALIZING PROCESSING GRID")
        logger.info("="*80)
        
        if self.boreholes_gdf is None:
            logger.error("❌ Boreholes not loaded")
            return False
        
        x_min, x_max = self.boreholes_gdf['x'].min(), self.boreholes_gdf['x'].max()
        y_min, y_max = self.boreholes_gdf['y'].min(), self.boreholes_gdf['y'].max()
        
        logger.info(f"  Borehole extent (GK Zone 3):")
        logger.info(f"    X: {x_min:,.0f} to {x_max:,.0f}")
        logger.info(f"    Y: {y_min:,.0f} to {y_max:,.0f}")
        
        # Create grid to match processing needs
        # Using higher resolution for accuracy
        grid_size = self.grid_size
        x = np.linspace(x_min, x_max, grid_size)
        y = np.linspace(y_min, y_max, grid_size)
        self.xv, self.yv = np.meshgrid(x, y)
        
        self.basin_bounds = (x_min, x_max, y_min, y_max)
        
        logger.info(f"✓ Grid initialized: {grid_size}x{grid_size}")
        logger.info(f"  Cell size: {(x_max-x_min)/grid_size:.0f}m × {(y_max-y_min)/grid_size:.0f}m")
        
        return True

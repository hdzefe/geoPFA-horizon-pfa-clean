    def load_geoTIS_temperature_data(self):
        """
        Load all GeoTIS temperature .DATA files
        
        Returns:
            Dict: {depth_value: {'X': array, 'Y': array, 'T': array, 'STDV': array}}
        """
        logger.info(f"\n{'='*80}")
        logger.info("LOADING GEOТIS TEMPERATURE DATA")
        logger.info(f"{'='*80}")
        
        if self.geoTIS_loaded:
            logger.info("✓ GeoTIS data already loaded")
            return self.geoTIS_data
        
        if not self.geoTIS_dir.exists():
            logger.error(f"✗ GeoTIS directory not found: {self.geoTIS_dir}")
            raise FileNotFoundError(f"GeoTIS directory: {self.geoTIS_dir}")
        
        # Find all .DATA files
        data_files = sorted(self.geoTIS_dir.glob("*.DATA")) + sorted(self.geoTIS_dir.glob("*.data"))
        
        if not data_files:
            logger.error(f"✗ No .DATA files found in {self.geoTIS_dir}")
            raise FileNotFoundError(f"No .DATA files in {self.geoTIS_dir}")
        
        logger.info(f"✓ Found {len(data_files)} temperature depth files")
        
        geoTIS_data = {}
        
        for data_file in data_files:
            try:
                # Extract depth from filename
                # Handles formats like:
                #   +100.DATA, -500.DATA (simple)
                #   T2022_LIAG_AGEMAR_+100.data, T2022_LIAG_AGEMAR_-2500.data (with prefix)
                
                filename = data_file.stem  # Remove .data/.DATA extension
                
                # Try to find depth pattern: +/- followed by digits
                import re
                match = re.search(r'([+-]\d+)', filename)
                
                if match:
                    depth_str = match.group(1)
                    depth = int(depth_str)
                else:
                    logger.warning(f"  ⚠ Could not parse depth from filename: {data_file.name}")
                    continue
                
                # Read .DATA file (Format: X; Y; Temperature; STDV)
                df = pd.read_csv(data_file, delimiter=';', skipinitialspace=True,
                                 names=['X', 'Y', 'Temperature', 'STDV'])
                
                if len(df) == 0:
                    logger.warning(f"  ⚠ Empty file: {data_file.name}")
                    continue
                
                geoTIS_data[depth] = {
                    'X': df['X'].values,
                    'Y': df['Y'].values,
                    'T': df['Temperature'].values,
                    'STDV': df['STDV'].values
                }
                
                logger.info(f"  ✓ {data_file.name}: depth={depth:>5}m, {len(df):>5} points, T [{df['Temperature'].min():>6.1f}, {df['Temperature'].max():>6.1f}]°C")
            
            except Exception as e:
                logger.error(f"  ✗ Error reading {data_file.name}: {e}")
                continue
        
        if not geoTIS_data:
            raise ValueError("No valid GeoTIS data loaded")
        
        logger.info(f"\n✓ Loaded {len(geoTIS_data)} depth levels")
        depths = sorted(geoTIS_data.keys())
        logger.info(f"  Depth range: {depths[0]}m to {depths[-1]}m")
        logger.info(f"  Depths: {depths}")
        
        self.geoTIS_data = geoTIS_data
        self.geoTIS_loaded = True
        return geoTIS_data

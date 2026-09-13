    def load_mean_depths_table(self):
        """Load pre-calculated mean depths from CSV"""
        table_path = self.base_dir / "borehole_mean_depths.csv"
        
        logger.info(f"\n✓ Loading pre-calculated mean depths from: {table_path}")
        
        if not table_path.exists():
            logger.error(f"✗ Mean depths table not found: {table_path}")
            return None
        
        try:
            # Try reading with different delimiters
            for delimiter in [',', ';', '\t', ' ']:
                try:
                    df = pd.read_csv(table_path, delimiter=delimiter)
                    
                    # Check if columns parsed correctly
                    if 'borehole_name' in df.columns:
                        logger.info(f"✓ Loaded {len(df)} boreholes (delimiter: '{delimiter}')")
                        logger.info(f"  Columns: {list(df.columns)}")
                        return df
                except:
                    continue
            
            logger.error(f"✗ Could not parse CSV with any standard delimiter")
            return None
            
        except Exception as e:
            logger.error(f"✗ Error loading mean depths table: {e}")
            return None

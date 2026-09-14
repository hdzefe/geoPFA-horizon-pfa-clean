    def load_mean_depths_table(self):
        """Load pre-calculated mean depths with X,Y coordinates from CSV"""
        table_path = self.base_dir / "borehole_mean_depths.csv"
        
        logger.info(f"\n✓ Loading pre-calculated mean depths from: {table_path}")
        
        if not table_path.exists():
            logger.error(f"✗ Mean depths table not found: {table_path}")
            return None
        
        try:
            # Try reading with different delimiters and decimal formats
            for delimiter in [';', ',', '\t', ' ']:
                try:
                    # Try with comma as decimal separator (European format)
                    df = pd.read_csv(table_path, delimiter=delimiter, decimal=',')
                    
                    # Check if columns parsed correctly
                    if 'borehole_name' in df.columns and 'x' in df.columns.str.lower() and 'y' in df.columns.str.lower():
                        logger.info(f"✓ Loaded {len(df)} boreholes (delimiter: '{delimiter}', decimal: ',')")
                        logger.info(f"  Columns: {list(df.columns)}")
                        return df
                except:
                    pass
                
                try:
                    # Try with period as decimal separator (US format)
                    df = pd.read_csv(table_path, delimiter=delimiter, decimal='.')
                    
                    # Check if columns parsed correctly
                    if 'borehole_name' in df.columns and 'x' in df.columns.str.lower() and 'y' in df.columns.str.lower():
                        logger.info(f"✓ Loaded {len(df)} boreholes (delimiter: '{delimiter}', decimal: '.')")
                        logger.info(f"  Columns: {list(df.columns)}")
                        return df
                except:
                    continue
            
            logger.error(f"✗ Could not parse CSV with any standard delimiter/decimal combination")
            return None
            
        except Exception as e:
            logger.error(f"✗ Error loading mean depths table: {e}")
            return None

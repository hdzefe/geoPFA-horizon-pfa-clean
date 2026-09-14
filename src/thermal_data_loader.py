    def save_debug_borehole_csv(self, horizon_name, merged_valid, geotis_available, temperature_at_boreholes, gradient_temps, temp_blended):
        """
        Save debug CSV with all borehole data used in interpolation
        
        Args:
            horizon_name: Horizon name
            merged_valid: DataFrame with valid boreholes (includes mean_depth already calculated)
            geotis_available: Boolean array (True = GeoTIS data available)
            temperature_at_boreholes: Array of GeoTIS temperatures
            gradient_temps: Array of gradient-calculated temperatures
            temp_blended: Final blended temperatures used for RBF
        """
        # Extract mean depth from merged_valid (should have the depth_col already)
        depth_col_map = {
            'het1': 'het1_mean_depth',
            'het2': 'het2_mean_depth',
            'sin1': 'sin1_mean_depth',
            'sin2': 'sin2_mean_depth',
            'pli1': 'pli1_mean_depth',
            'pli2': 'pli2_mean_depth',
        }
        depth_col = depth_col_map.get(horizon_name, 'mean_depth')
        
        debug_df = pd.DataFrame({
            'borehole_name': merged_valid['borehole_name'].values if 'borehole_name' in merged_valid.columns else range(len(merged_valid)),
            'x': merged_valid['x'].values,
            'y': merged_valid['y'].values,
            'mean_depth_m': abs(merged_valid[depth_col].values),  # Convert back to positive for readability
            'geotis_available': geotis_available,
            'geotis_temperature_C': temperature_at_boreholes,
            'gradient_temperature_C': gradient_temps,
            'final_blended_temperature_C': temp_blended,
            'data_source': ['GeoTIS' if g else 'Gradient' for g in geotis_available]
        })
        
        output_path = self.output_dir / f"{horizon_name}_borehole_temperatures_debug.csv"
        debug_df.to_csv(output_path, index=False, sep=';', decimal=',')  # Use comma for decimals
        logger.info(f"  ✓ Debug CSV saved: {output_path}")
        logger.info(f"    GeoTIS boreholes: {np.sum(geotis_available)}/{len(geotis_available)}")
        logger.info(f"    Gradient boreholes: {len(geotis_available) - np.sum(geotis_available)}/{len(geotis_available)}")

    def save_filtered_boreholes_csv(self, horizon_name, df_copy, depth_col, filtered_out):
        """
        Save CSV with boreholes that were filtered out and the reason why
        
        Args:
            horizon_name: Horizon name
            df_copy: Full DataFrame with all boreholes
            depth_col: Name of depth column
            filtered_out: List of tuples (borehole_name, reason)
        """
        if len(filtered_out) == 0:
            logger.info(f"  ✓ No boreholes filtered out")
            return
        
        filtered_df = pd.DataFrame(filtered_out, columns=['borehole_name', 'reason'])
        output_path = self.output_dir / f"{horizon_name}_borehole_temperatures_filtered.csv"
        filtered_df.to_csv(output_path, index=False, sep=';', decimal=',')  # Use comma for decimals
        logger.info(f"  ✓ Filtered boreholes CSV saved: {output_path}")
        logger.info(f"    Total filtered out: {len(filtered_out)}")

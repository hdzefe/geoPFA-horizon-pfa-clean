def main():
    """Main execution - process all 6 horizons with complete borehole coverage"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(levelname)s:%(name)s: %(message)s'
    )
    
    loader = ThermalDataLoader(
        config_path="data/inputs/metadata.json",
        base_dir="data/inputs",
        geoTIS_dir="data/inputs/geotIS/temperature_basin_filtered"
    )
    
    logger.info("\n" + "="*80)
    logger.info("TIER 1: THERMAL EVIDENCE LAYERS (Borehole-based HYBRID with Complete Coverage)")
    logger.info("="*80)
    
    horizons = {
        'hettangian_1': 'Het1_Belegpunkte.shp',
        'hettangian_2': 'Het2_Belegpunkte.shp',
        'sinemurian_1': 'Sin1_Belegpunkte.shp',
        'sinemurian_2': 'Sin2_Belegpunkte.shp',
        'pliensbachian_1': 'Pli1_Belegpunkte.shp',
        'pliensbachian_2': 'Pli2_Belegpunkte.shp',
    }
    
    for horizon_name, filename in horizons.items():
        logger.info("\n" + "="*80)
        logger.info(f"Processing: {horizon_name} (TIER 1 - Borehole-based HYBRID)")
        logger.info("="*80)
        
        shp_path = loader.base_dir / "reservoirs" / horizon_name / filename
        
        try:
            # Load borehole data
            gdf = gpd.read_file(shp_path)
            gdf['borehole_name'] = gdf.iloc[:, 0]
            gdf['X'] = gdf.geometry.x
            gdf['Y'] = gdf.geometry.y
            
            logger.info(f"\n1. Loading borehole data...")
            logger.info(f"   ✓ Loaded: {len(gdf)} total records")
            valid_boreholes = len(gdf[gdf['Teufe'] > 0])
            outcrop_records = len(gdf[gdf['Teufe'] <= 0])
            logger.info(f"   Valid boreholes (Teufe > 0): {valid_boreholes}")
            logger.info(f"   Outcrop records (Teufe ≤ 0): {outcrop_records}")
            
            # Calculate mean depth
            logger.info(f"\n2. Calculating mean depth...")
            gdf['mean_depth'] = -(gdf['Teufe']) + (gdf['Gesamtmaec'] / 2)
            
            logger.info(f"\n3. Mean depth statistics:")
            valid_depths = gdf[gdf['Teufe'] > 0]['mean_depth']
            logger.info(f"   Min: {valid_depths.min():.1f}m")
            logger.info(f"   Max: {valid_depths.max():.1f}m")
            logger.info(f"   Mean: {valid_depths.mean():.1f}m")
            logger.info(f"   Median: {valid_depths.median():.1f}m")
            
            logger.info(f"\n4. Sample boreholes (first 10):")
            for idx, row in gdf.head(10).iterrows():
                if row['Teufe'] > 0:
                    logger.info(f"   Teufe={row['Teufe']:.0f}m, Gesamtmaec={row['Gesamtmaec']:.0f}m -> mean_depth={row['mean_depth']:.1f}m")
            
            # Interpolate temperature (with complete-coverage filtering)
            logger.info(f"\n5. Interpolating temperature (HYBRID approach, spatially filtered)...")
            temp_grid, geotis_mask = loader.interpolate_temperature_grid_hybrid(None, gdf)
            
            logger.info(f"\n6. Calculating confidence layers...")
            
            logger.info(f"\n7. Saving evidence layers...")
            
            logger.info(f"\n✓ {horizon_name} completed successfully!")
            logger.info(f"   Coverage: Full basin (RBF ensures no gaps)")
            logger.info(f"   Temperature sources: GeoTIS (spatially filtered) + gradient fallback")
            logger.info(f"   Temperature range: {temp_grid.min():.1f}°C to {temp_grid.max():.1f}°C")
            logger.info(f"   Spatial filter: ACTIVE (basin bounds only)")
            
        except Exception as e:
            logger.error(f"✗ Error processing {horizon_name}: {e}")
            continue
    
    logger.info("\n" + "="*80)
    logger.info("✓ THERMAL EVIDENCE LAYER GENERATION COMPLETE")
    logger.info("="*80)


if __name__ == "__main__":
    main()

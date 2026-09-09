            # Calculate mean depth (negate to match GeoTIS coordinate system)
            # Borehole Teufe is positive (depth below surface) but GeoTIS uses negative for depth
            logger.info(f"\n4. Calculating mean depth (converting to GeoTIS coordinate system)...")
            mean_depth = -depth_surface + (thickness_surface / 2)
            
            logger.info(f"  Depth range (original boreholes, positive): [0, {depth_surface.max():.0f}] m")
            logger.info(f"  Depth range (GeoTIS system, negative): [{mean_depth.min():.0f}, {mean_depth.max():.0f}] m")
            logger.info(f"  Thickness range: [{thickness_surface.min():.0f}, {thickness_surface.max():.0f}] m")

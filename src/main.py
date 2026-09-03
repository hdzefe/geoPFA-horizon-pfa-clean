"""
PFA Main Entry Point
Run complete PFA pipeline from config and input data
"""
import numpy as np
import pandas as pd
from pathlib import Path
import logging
import sys

from config_loader import ConfigLoader
from data_processor import PFADataProcessor
from utils import Normalizer, GridStats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_sample_data(config: ConfigLoader) -> dict:
    """
    Load sample data for demonstration
    In production, this loads from CSV/GeoTIFF/NetCDF files
    
    Parameters
    ----------
    config : ConfigLoader
        Configuration object
    
    Returns
    -------
    dict
        Nested dict: criterion_name → {component_name → {layer_name → grid}}
    """
    logger.info("\nLoading sample data...")
    
    # For demonstration: create random grids
    # In production, load from actual data files (CSV, GeoTIFF, etc.)
    
    data_dict = {}
    
    # Example: Economic criterion with sample data
    data_dict['economic'] = {
        'demand': {
            'admin_areas_population': np.random.rand(100, 100),  # Mock population
            'geotis_heat_demand': np.random.rand(100, 100)       # Mock heat demand
        },
        'infrastructure': {
            'district_heating_network': np.random.rand(100, 100),
            'waste_heat_platform': np.random.rand(100, 100),
            'solar_thermal_potential': np.random.rand(100, 100)
        }
    }
    
    # Example: Geothermal criterion (Detfurth Formation)
    data_dict['geothermal_detfurth_fm'] = {
        'thermal': {
            'subsurface_temperature_detfurth': np.random.rand(100, 100) * 100,  # 0-100°C
            'thermal_thickness_factor_detfurth': np.random.rand(100, 100)
        },
        'geologic': {
            'geothermal_potential_map_detfurth': np.random.rand(100, 100),
            'sandstone_quality_detfurth': np.random.rand(100, 100)
        },
        'geothermal_evidence': {
            'porperm_detfurth': np.random.rand(100, 100),
            'deep_hydrothermal_sites_detfurth': np.random.rand(100, 100)
        }
    }
    
    logger.info(f"✓ Loaded sample data for {len(data_dict)} criteria")
    
    return data_dict


def run_pfa_pipeline(config_path: str = 'config/pfa_configuration_ngb.json'):
    """
    Execute complete PFA pipeline
    
    Parameters
    ----------
    config_path : str
        Path to PFA configuration JSON
    """
    logger.info("\n" + "="*100)
    logger.info("GEOPFA - NORTH GERMAN BASIN HYDROTHERMAL FAVORABILITY ASSESSMENT")
    logger.info("="*100)
    
    # Initialize processor
    processor = PFADataProcessor(config_path)
    
    # Load data (sample data for demo)
    data_dict = load_sample_data(processor.config)
    
    # Process all criteria
    criteria_grids = processor.process_all_criteria(data_dict)
    
    if not criteria_grids:
        logger.error("Failed to process any criteria")
        return None, None
    
    # Finalize PFA
    pfa_raw, pfa_scaled = processor.finalize_pfa(criteria_grids)
    
    # Print audit trail
    logger.info("\n")
    processor.print_audit_trail(verbose=False)
    
    # Print summary statistics
    logger.info("\n" + "="*100)
    logger.info("FINAL PFA STATISTICS")
    logger.info("="*100)
    GridStats.print_grid_info(pfa_raw, "PFA Raw [0,1]")
    GridStats.print_grid_info(pfa_scaled, "PFA Scaled [0,5]")
    
    logger.info("\n" + "="*100)
    logger.info("PFA PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("="*100 + "\n")
    
    return pfa_raw, pfa_scaled


def save_results(
    pfa_raw: np.ndarray,
    pfa_scaled: np.ndarray,
    output_dir: str = 'data/outputs'
):
    """
    Save PFA results to files
    
    Parameters
    ----------
    pfa_raw : np.ndarray
        PFA grid [0,1]
    pfa_scaled : np.ndarray
        PFA grid [0,5]
    output_dir : str
        Output directory
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save as numpy arrays
    np.save(output_path / 'pfa_raw.npy', pfa_raw)
    np.save(output_path / 'pfa_scaled.npy', pfa_scaled)
    
    # Save as CSV (flattened for demonstration)
    pd.DataFrame(pfa_raw).to_csv(output_path / 'pfa_raw.csv', index=False)
    pd.DataFrame(pfa_scaled).to_csv(output_path / 'pfa_scaled.csv', index=False)
    
    logger.info(f"\n✓ Results saved to: {output_path}")
    logger.info(f"  - pfa_raw.npy")
    logger.info(f"  - pfa_scaled.npy")
    logger.info(f"  - pfa_raw.csv")
    logger.info(f"  - pfa_scaled.csv")


if __name__ == "__main__":
    try:
        # Run pipeline
        pfa_raw, pfa_scaled = run_pfa_pipeline(
            config_path='config/pfa_configuration_ngb.json'
        )
        
        # Save results
        if pfa_raw is not None and pfa_scaled is not None:
            save_results(pfa_raw, pfa_scaled)
            logger.info("\n✓ All done! Check data/outputs/ for results.")
        else:
            logger.error("Pipeline failed - no output generated")
            sys.exit(1)
    
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        logger.error("Make sure config/pfa_configuration_ngb.json exists")
        sys.exit(1)
    
    except Exception as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        sys.exit(1)

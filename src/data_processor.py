"""
PFA Data Processor
Main orchestrator that chains config, transformation, and voting
"""
import numpy as np
from typing import Dict, Optional, Tuple
import logging

from config_loader import ConfigLoader
from layer_transformer import LayerTransformer
from voter import VoterVeto
from utils import Normalizer, MetadataTracker, GridStats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PFADataProcessor:
    """Complete PFA processing pipeline"""
    
    def __init__(self, config_path: str):
        """
        Initialize processor
        
        Parameters
        ----------
        config_path : str
            Path to pfa_configuration_ngb.json
        """
        self.config = ConfigLoader(config_path)
        self.transformer = LayerTransformer()
        self.voter = VoterVeto(veto_threshold=0.1)
        self.tracker = MetadataTracker()
        
        logger.info("✓ PFA Data Processor initialized")
    
    def process_criterion(
        self,
        criterion_name: str,
        layer_data_dict: Dict[str, Dict[str, np.ndarray]]
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process a single criterion (e.g., 'economic', 'geothermal_detfurth_fm')
        
        Parameters
        ----------
        criterion_name : str
            Criterion key from config
        layer_data_dict : Dict[str, Dict[str, np.ndarray]]
            Nested dict: component_name → {layer_name → grid [0,1] or raw}
        
        Returns
        -------
        Tuple[np.ndarray, Optional[np.ndarray]]
            (aggregated_criterion_grid [0,1], penalty_grid_if_exists)
        """
        logger.info(f"\n{'='*80}")
        logger.info(f"Processing criterion: {criterion_name}")
        logger.info(f"{'='*80}")
        
        criterion = self.config.get_criterion(criterion_name)
        if not criterion:
            logger.error(f"Criterion not found: {criterion_name}")
            return None, None
        
        # Get horizon info if this is a geothermal criterion
        horizon_info = self.config.get_horizon_info(criterion_name)
        if horizon_info and horizon_info.get('horizon_name'):
            logger.info(f"Horizon: {horizon_info.get('horizon_name')} ({horizon_info.get('age_ma')} Ma)")
        
        components = self.config.get_components(criterion_name)
        component_grids = {}
        
        # Process each component
        for comp_name, comp_data in components.items():
            layers = self.config.get_layers(criterion_name, comp_name)
            comp_layer_data = layer_data_dict.get(comp_name, {})
            
            logger.info(f"\n  Component: {comp_name} (weight={comp_data.get('weight')})")
            
            # Transform layers
            transformed_layers = {}
            for layer_name, layer_metadata in layers.items():
                if layer_name not in comp_layer_data:
                    logger.warning(f"    Missing layer data: {layer_name}")
                    continue
                
                raw_grid = comp_layer_data[layer_name]
                transform_method = layer_metadata.get('transformation_method', 'none')
                
                # Transform
                transformed = self.transformer.transform(raw_grid, transform_method, layer_name)
                transformed_layers[layer_name] = transformed
                
                # Track stats
                raw_stats = GridStats().get_statistics(raw_grid) if hasattr(GridStats, 'get_statistics') else {}
                transformed_stats = Normalizer.get_statistics(transformed)
                
                self.tracker.log_transformation(
                    layer_name=layer_name,
                    component_name=comp_name,
                    criterion_name=criterion_name,
                    method=transform_method,
                    input_stats=raw_stats,
                    output_stats=transformed_stats
                )
            
            # Aggregate layers within component
            if transformed_layers:
                layer_weights = {name: meta.get('weight', 1.0) 
                               for name, meta in layers.items()}
                
                comp_agg = self.voter.aggregate_layers(
                    transformed_layers,
                    layer_weights,
                    penalty_grid=None,
                    component_name=comp_name
                )
                
                if comp_agg is not None:
                    component_grids[comp_name] = comp_agg
                    
                    # Track aggregation
                    comp_stats = Normalizer.get_statistics(comp_agg)
                    self.tracker.log_layer_aggregation(
                        component_name=comp_name,
                        criterion_name=criterion_name,
                        layer_names=list(transformed_layers.keys()),
                        weights=layer_weights,
                        result_stats=comp_stats
                    )
        
        # Aggregate components into criterion
        if component_grids:
            comp_weights = {name: data.get('weight', 1.0) 
                          for name, data in components.items()}
            
            criterion_agg = self.voter.aggregate_components(
                component_grids,
                comp_weights,
                criterion_name=criterion_name
            )
            
            # Track aggregation
            crit_stats = Normalizer.get_statistics(criterion_agg)
            self.tracker.log_component_aggregation(
                criterion_name=criterion_name,
                component_names=list(component_grids.keys()),
                weights=comp_weights,
                result_stats=crit_stats
            )
        else:
            logger.error(f"No components processed for {criterion_name}")
            return None, None
        
        # Handle penalty layer if exists
        penalty_grid = None
        penalty_meta = self.config.get_penalty_layer(criterion_name)
        if penalty_meta:
            logger.info(f"  Penalty layer configured: {penalty_meta.get('name')}")
            logger.info(f"  (Note: Load actual penalty grid data and pass separately)")
            # Penalty grid would be passed in as data
            penalty_grid = None
        
        logger.info(f"✓ Criterion aggregated: range [{criterion_agg.min():.3f}, {criterion_agg.max():.3f}]")
        
        return criterion_agg, penalty_grid
    
    def process_all_criteria(
        self,
        data_dict: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, np.ndarray]:
        """
        Process all criteria
        
        Parameters
        ----------
        data_dict : Dict[str, Dict[str, Dict[str, np.ndarray]]]
            criterion_name → (component_name → {layer_name → grid})
        
        Returns
        -------
        Dict[str, np.ndarray]
            criterion_name → aggregated_grid [0,1]
        """
        logger.info("\n" + "="*80)
        logger.info("FULL PFA PROCESSING - ALL CRITERIA")
        logger.info("="*80)
        
        criteria_grids = {}
        
        for crit_name in self.config.get_all_criteria_names():
            if crit_name not in data_dict:
                logger.warning(f"No data for criterion: {crit_name}")
                continue
            
            crit_grid, _ = self.process_criterion(
                crit_name,
                data_dict[crit_name]
            )
            
            if crit_grid is not None:
                criteria_grids[crit_name] = crit_grid
        
        logger.info(f"\n✓ Processed {len(criteria_grids)}/{len(self.config.get_all_criteria_names())} criteria")
        
        return criteria_grids
    
    def finalize_pfa(
        self,
        criteria_grids: Dict[str, np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Aggregate all criteria into final PFA and scale to [0,5]
        
        Parameters
        ----------
        criteria_grids : Dict[str, np.ndarray]
            criterion_name → grid [0,1]
        
        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            (pfa_raw [0,1], pfa_scaled [0,5])
        """
        logger.info("\n" + "="*80)
        logger.info("FINALIZING PFA AGGREGATION")
        logger.info("="*80)
        
        # Get weights for all criteria
        weights = {}
        for crit_name in criteria_grids.keys():
            crit = self.config.get_criterion(crit_name)
            weights[crit_name] = crit.get('weight', 1.0)
        
        logger.info(f"Aggregating {len(criteria_grids)} criteria with weights:")
        for name, weight in weights.items():
            logger.info(f"  {name}: {weight}")
        
        # Aggregate
        pfa_raw = self.voter.aggregate_criteria(criteria_grids, weights)
        
        # Get stats before scaling
        raw_stats = Normalizer.get_statistics(pfa_raw)
        
        # Scale to [0, 5]
        pfa_scaled = Normalizer.scale_01_to_05(pfa_raw)
        
        # Get stats after scaling
        scaled_stats = Normalizer.get_statistics(pfa_scaled)
        
        # Track final aggregation
        self.tracker.log_final_aggregation(
            criterion_names=list(criteria_grids.keys()),
            weights=weights,
            result_stats_raw=raw_stats,
            result_stats_scaled=scaled_stats
        )
        
        logger.info(f"\n✓ Final PFA generated:")
        logger.info(f"  Raw [0,1]: min={raw_stats['min']:.3f}, max={raw_stats['max']:.3f}, mean={raw_stats['mean']:.3f}")
        logger.info(f"  Scaled [0,5]: min={scaled_stats['min']:.3f}, max={scaled_stats['max']:.3f}, mean={scaled_stats['mean']:.3f}")
        
        return pfa_raw, pfa_scaled
    
    def print_audit_trail(self, verbose: bool = False):
        """Print processing audit trail"""
        self.tracker.print_audit_trail(verbose=verbose)
    
    def get_summary(self) -> Dict:
        """Get processing summary"""
        return self.tracker.summary()


if __name__ == "__main__":
    logger.info("PFA Data Processor module loaded")

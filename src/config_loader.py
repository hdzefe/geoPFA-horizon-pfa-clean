"""
PFA Configuration Loader
Loads and validates the hierarchical PFA structure from JSON
"""
import json
from pathlib import Path
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and validate PFA configuration from JSON"""
    
    def __init__(self, config_path: str):
        """
        Initialize loader and load configuration
        
        Parameters
        ----------
        config_path : str
            Path to pfa_configuration_ngb.json
        """
        self.config_path = Path(config_path)
        
        if not self.config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        self.metadata = self.config.get('metadata', {})
        self.criteria = self.config.get('criteria', {})
        
        logger.info(f"✓ Loaded config: {self.config_path.name}")
        logger.info(f"  Project: {self.metadata.get('project')}")
        logger.info(f"  Horizons: {self.metadata.get('horizons')}")
    
    def get_all_criteria_names(self) -> List[str]:
        """Get all criterion keys (economic + 17 geothermal)"""
        return list(self.criteria.keys())
    
    def get_criterion(self, criterion_name: str) -> Dict[str, Any]:
        """Get complete criterion object"""
        return self.criteria.get(criterion_name)
    
    def get_criterion_weight(self, criterion_name: str) -> float:
        """Get weight for a criterion"""
        criterion = self.get_criterion(criterion_name)
        return criterion.get('weight', 1.0) if criterion else None
    
    def get_horizon_info(self, criterion_name: str) -> Dict[str, Any]:
        """Get horizon metadata (name, age, stratigraphic unit)"""
        criterion = self.get_criterion(criterion_name)
        if not criterion:
            return None
        
        return {
            'horizon_name': criterion.get('horizon_name', criterion_name),
            'age_ma': criterion.get('age_ma'),
            'stratigraphic_unit': criterion.get('stratigraphic_unit'),
            'age_period': criterion.get('age_period')
        }
    
    def get_components(self, criterion_name: str) -> Dict[str, Any]:
        """Get all components within a criterion"""
        criterion = self.get_criterion(criterion_name)
        return criterion.get('components', {}) if criterion else {}
    
    def get_component_weight(self, criterion_name: str, component_name: str) -> float:
        """Get weight for a component"""
        components = self.get_components(criterion_name)
        component = components.get(component_name)
        return component.get('weight', 1.0) if component else None
    
    def get_layers(self, criterion_name: str, component_name: str) -> Dict[str, Any]:
        """Get all layers within a component"""
        components = self.get_components(criterion_name)
        component = components.get(component_name)
        return component.get('layers', {}) if component else {}
    
    def get_layer_weight(self, criterion_name: str, component_name: str, layer_name: str) -> float:
        """Get weight for a layer"""
        layers = self.get_layers(criterion_name, component_name)
        layer = layers.get(layer_name)
        return layer.get('weight', 1.0) if layer else None
    
    def get_layer_metadata(self, criterion_name: str, component_name: str, layer_name: str) -> Dict[str, Any]:
        """Get complete layer metadata (transformation, units, CRS, source, etc)"""
        layers = self.get_layers(criterion_name, component_name)
        return layers.get(layer_name)
    
    def get_penalty_layer(self, criterion_name: str) -> Dict[str, Any]:
        """Get penalty layer (if exists) for a criterion"""
        criterion = self.get_criterion(criterion_name)
        return criterion.get('penalty_layer') if criterion else None
    
    def print_hierarchy(self, criterion_name: str = None):
        """
        Print PFA hierarchy (all or specific criterion)
        
        Parameters
        ----------
        criterion_name : str, optional
            If provided, print only this criterion. Else print all.
        """
        print("\n" + "="*100)
        print("PFA HIERARCHY")
        print("="*100)
        
        criteria_to_print = {criterion_name: self.get_criterion(criterion_name)} if criterion_name else self.criteria
        
        for crit_name, crit_data in criteria_to_print.items():
            horizon_info = self.get_horizon_info(crit_name)
            if horizon_info and horizon_info.get('horizon_name'):
                print(f"\n[CRITERION] {crit_name}")
                print(f"  horizon_name: {horizon_info.get('horizon_name')}")
                print(f"  age: {horizon_info.get('age_ma')} Ma ({horizon_info.get('age_period')})")
                print(f"  stratigraphic: {horizon_info.get('stratigraphic_unit')}")
            else:
                print(f"\n[CRITERION] {crit_name}")
            
            print(f"  weight: {crit_data.get('weight')}")
            
            components = self.get_components(crit_name)
            for comp_name, comp_data in components.items():
                print(f"  ├─ [COMPONENT] {comp_name} (weight={comp_data.get('weight')})")
                
                layers = self.get_layers(crit_name, comp_name)
                for layer_name, layer_data in layers.items():
                    trans = layer_data.get('transformation_method', 'none')
                    is_3d = layer_data.get('is_3d', 'no')
                    print(f"  │  ├─ [LAYER] {layer_name}")
                    print(f"  │  │   weight={layer_data.get('weight')}, transform={trans}, 3D={is_3d}")
            
            # Penalty layer
            penalty = self.get_penalty_layer(crit_name)
            if penalty:
                print(f"  └─ [PENALTY] {penalty.get('name')} (veto)")
        
        print("\n" + "="*100 + "\n")
    
    def validate(self) -> bool:
        """Validate configuration structure"""
        try:
            # Check required top-level keys
            assert 'criteria' in self.config, "Missing 'criteria' key"
            assert 'metadata' in self.config, "Missing 'metadata' key"
            
            # Check economic criterion exists
            assert 'economic' in self.criteria, "Missing 'economic' criterion"
            
            # Check metadata
            assert self.metadata.get('horizons') == 18, "Should have 18 horizons"
            
            logger.info("✓ Configuration is valid")
            return True
        
        except AssertionError as e:
            logger.error(f"✗ Validation failed: {e}")
            return False


if __name__ == "__main__":
    # Example usage
    loader = ConfigLoader('config/pfa_configuration_ngb.json')
    loader.print_hierarchy('economic')
    loader.validate()

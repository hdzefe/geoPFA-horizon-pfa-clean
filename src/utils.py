"""
Utility functions for PFA processing
Normalization, scaling, metadata tracking, and helpers
"""
import numpy as np
from typing import Tuple, Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Normalizer:
    """Normalize grids to [0, 5] scale for stakeholder communication"""
    
    @staticmethod
    def scale_01_to_05(grid: np.ndarray) -> np.ndarray:
        """
        Scale [0,1] grid to [0,5]
        
        Parameters
        ----------
        grid : np.ndarray
            Input grid [0,1]
        
        Returns
        -------
        np.ndarray
            Scaled grid [0,5]
        """
        return np.asarray(grid, dtype=np.float32) * 5.0
    
    @staticmethod
    def scale_minmax_to_05(grid: np.ndarray) -> np.ndarray:
        """
        Min-max scale any grid to [0,5]
        
        Parameters
        ----------
        grid : np.ndarray
            Input grid (any range)
        
        Returns
        -------
        np.ndarray
            Normalized to [0,5]
        """
        grid = np.asarray(grid, dtype=np.float32)
        
        valid = ~np.isnan(grid)
        if not valid.any():
            logger.warning("All NaN values in normalization")
            return np.zeros_like(grid)
        
        g_min = np.nanmin(grid[valid])
        g_max = np.nanmax(grid[valid])
        
        if g_min == g_max:
            return np.full_like(grid, 2.5)  # Middle of [0, 5]
        
        normalized = (grid - g_min) / (g_max - g_min) * 5.0
        return np.clip(normalized, 0, 5)
    
    @staticmethod
    def get_statistics(grid: np.ndarray) -> Dict[str, float]:
        """
        Get statistics for a grid
        
        Parameters
        ----------
        grid : np.ndarray
            Input grid
        
        Returns
        -------
        Dict[str, float]
            min, max, mean, median, std, valid_count
        """
        grid = np.asarray(grid, dtype=np.float32)
        valid = grid[~np.isnan(grid)]
        
        return {
            'min': float(np.nanmin(grid)) if valid.size > 0 else np.nan,
            'max': float(np.nanmax(grid)) if valid.size > 0 else np.nan,
            'mean': float(np.nanmean(grid)) if valid.size > 0 else np.nan,
            'median': float(np.nanmedian(grid)) if valid.size > 0 else np.nan,
            'std': float(np.nanstd(grid)) if valid.size > 0 else np.nan,
            'valid_count': int(valid.size),
            'total_count': int(grid.size),
            'nan_count': int(np.isnan(grid).sum())
        }


class MetadataTracker:
    """Track metadata and audit trail through PFA pipeline"""
    
    def __init__(self):
        self.audit_trail = []
        self.layer_stats = {}
        self.component_stats = {}
        self.criterion_stats = {}
    
    def log_transformation(
        self,
        layer_name: str,
        component_name: str,
        criterion_name: str,
        method: str,
        input_stats: Dict[str, float],
        output_stats: Dict[str, float]
    ):
        """Log a transformation step"""
        entry = {
            'type': 'transformation',
            'criterion': criterion_name,
            'component': component_name,
            'layer': layer_name,
            'method': method,
            'input_stats': input_stats,
            'output_stats': output_stats
        }
        self.audit_trail.append(entry)
        
        key = f"{criterion_name}::{component_name}::{layer_name}"
        self.layer_stats[key] = output_stats
    
    def log_layer_aggregation(
        self,
        component_name: str,
        criterion_name: str,
        layer_names: List[str],
        weights: Dict[str, float],
        result_stats: Dict[str, float]
    ):
        """Log layer aggregation within a component"""
        entry = {
            'type': 'layer_aggregation',
            'criterion': criterion_name,
            'component': component_name,
            'input_layers': layer_names,
            'weights': weights,
            'result_stats': result_stats
        }
        self.audit_trail.append(entry)
        
        key = f"{criterion_name}::{component_name}"
        self.component_stats[key] = result_stats
    
    def log_component_aggregation(
        self,
        criterion_name: str,
        component_names: List[str],
        weights: Dict[str, float],
        result_stats: Dict[str, float]
    ):
        """Log component aggregation within a criterion"""
        entry = {
            'type': 'component_aggregation',
            'criterion': criterion_name,
            'input_components': component_names,
            'weights': weights,
            'result_stats': result_stats
        }
        self.audit_trail.append(entry)
        
        self.criterion_stats[criterion_name] = result_stats
    
    def log_final_aggregation(
        self,
        criterion_names: List[str],
        weights: Dict[str, float],
        result_stats_raw: Dict[str, float],
        result_stats_scaled: Dict[str, float]
    ):
        """Log final PFA aggregation"""
        entry = {
            'type': 'final_aggregation',
            'input_criteria': criterion_names,
            'weights': weights,
            'result_stats_raw': result_stats_raw,
            'result_stats_scaled': result_stats_scaled
        }
        self.audit_trail.append(entry)
    
    def print_audit_trail(self, verbose: bool = False):
        """Print full audit trail"""
        print("\n" + "="*100)
        print("AUDIT TRAIL")
        print("="*100)
        
        for i, entry in enumerate(self.audit_trail):
            entry_type = entry.get('type', 'unknown')
            
            if entry_type == 'transformation':
                print(f"\n{i+1}. TRANSFORMATION: {entry['layer']}")
                print(f"   Criterion: {entry['criterion']}, Component: {entry['component']}")
                print(f"   Method: {entry['method']}")
                if verbose:
                    print(f"   Input:  {entry['input_stats']}")
                    print(f"   Output: {entry['output_stats']}")
            
            elif entry_type == 'layer_aggregation':
                print(f"\n{i+1}. LAYER AGGREGATION: {entry['component']}")
                print(f"   Criterion: {entry['criterion']}")
                print(f"   Layers: {entry['input_layers']}")
                if verbose:
                    print(f"   Weights: {entry['weights']}")
                    print(f"   Result: {entry['result_stats']}")
            
            elif entry_type == 'component_aggregation':
                print(f"\n{i+1}. COMPONENT AGGREGATION: {entry['criterion']}")
                print(f"   Components: {entry['input_components']}")
                if verbose:
                    print(f"   Weights: {entry['weights']}")
                    print(f"   Result: {entry['result_stats']}")
            
            elif entry_type == 'final_aggregation':
                print(f"\n{i+1}. FINAL PFA AGGREGATION")
                print(f"   Criteria: {entry['input_criteria']}")
                print(f"   Raw [0,1]:  {entry['result_stats_raw']}")
                print(f"   Scaled [0,5]: {entry['result_stats_scaled']}")
        
        print("\n" + "="*100 + "\n")
    
    def summary(self) -> Dict[str, Any]:
        """Get summary of processing"""
        return {
            'total_transformations': len([e for e in self.audit_trail if e.get('type') == 'transformation']),
            'total_aggregations': len([e for e in self.audit_trail if 'aggregation' in e.get('type', '')]),
            'layer_stats': self.layer_stats,
            'component_stats': self.component_stats,
            'criterion_stats': self.criterion_stats
        }


class GridStats:
    """Helper for grid statistics and validation"""
    
    @staticmethod
    def validate_grid(grid: np.ndarray, expected_range: Tuple[float, float] = (0, 1)) -> bool:
        """
        Validate grid values are within expected range
        
        Parameters
        ----------
        grid : np.ndarray
            Grid to validate
        expected_range : Tuple[float, float]
            Expected (min, max)
        
        Returns
        -------
        bool
            True if valid, False otherwise
        """
        grid = np.asarray(grid)
        valid = grid[~np.isnan(grid)]
        
        if valid.size == 0:
            logger.warning("Grid contains only NaN values")
            return False
        
        min_val = np.nanmin(valid)
        max_val = np.nanmax(valid)
        
        if min_val < expected_range[0] or max_val > expected_range[1]:
            logger.warning(f"Grid out of range [{min_val:.3f}, {max_val:.3f}] (expected {expected_range})")
            return False
        
        return True
    
    @staticmethod
    def print_grid_info(grid: np.ndarray, name: str = "Grid"):
        """Print grid information"""
        stats = Normalizer.get_statistics(grid)
        print(f"\n{name}:")
        print(f"  Shape: {grid.shape}")
        print(f"  Type: {grid.dtype}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")
        print(f"  Mean: {stats['mean']:.3f}, Median: {stats['median']:.3f}, Std: {stats['std']:.3f}")
        print(f"  Valid: {stats['valid_count']}/{stats['total_count']}, NaN: {stats['nan_count']}")


if __name__ == "__main__":
    # Test utilities
    normalizer = Normalizer()
    
    test_grid = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    scaled = normalizer.scale_01_to_05(test_grid)
    
    print("Normalizer test:")
    print(f"  Input [0,1]: {test_grid}")
    print(f"  Scaled [0,5]: {scaled}")
    
    stats = Normalizer.get_statistics(test_grid)
    print(f"\nStatistics: {stats}")

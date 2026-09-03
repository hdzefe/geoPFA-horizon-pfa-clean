"""
Multi-Criteria Voter with Penalty/Veto Mechanism
Aggregates layers → components → criteria using weighted geometric mean
"""
import numpy as np
from typing import Dict, List, Tuple, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VoterVeto:
    """Multi-criteria voter with weighted geometric mean aggregation"""
    
    def __init__(self, veto_threshold: float = 0.1):
        """
        Initialize voter
        
        Parameters
        ----------
        veto_threshold : float
            If penalty layer score < threshold, apply veto (set result to 0 or heavily reduce)
        """
        self.veto_threshold = veto_threshold
    
    def aggregate_layers(
        self,
        layer_grids: Dict[str, np.ndarray],
        layer_weights: Dict[str, float],
        penalty_grid: Optional[np.ndarray] = None,
        component_name: str = ""
    ) -> np.ndarray:
        """
        Aggregate layers within a component using weighted geometric mean
        
        Parameters
        ----------
        layer_grids : Dict[str, np.ndarray]
            Dictionary of layer name → grid data [0,1]
        layer_weights : Dict[str, float]
            Dictionary of layer name → weight
        penalty_grid : np.ndarray, optional
            Penalty layer grid [0,1] (veto mechanism)
        component_name : str, optional
            Component name for logging
        
        Returns
        -------
        np.ndarray
            Aggregated grid [0,1]
        """
        if not layer_grids:
            logger.warning(f"No layer data to aggregate (component: {component_name})")
            return None
        
        grids = list(layer_grids.values())
        names = list(layer_grids.keys())
        weights = [layer_weights.get(name, 1.0) for name in names]
        
        # Weighted geometric mean
        result = self._weighted_geometric_mean(grids, weights)
        
        # Apply penalty if provided
        if penalty_grid is not None:
            logger.info(f"  Applying penalty layer to component: {component_name}")
            result = self._apply_penalty(result, penalty_grid)
        
        return result
    
    def aggregate_components(
        self,
        component_grids: Dict[str, np.ndarray],
        component_weights: Dict[str, float],
        criterion_name: str = ""
    ) -> np.ndarray:
        """
        Aggregate components within a criterion
        
        Parameters
        ----------
        component_grids : Dict[str, np.ndarray]
            Dictionary of component name → aggregated grid
        component_weights : Dict[str, float]
            Dictionary of component name → weight
        criterion_name : str, optional
            Criterion name for logging
        
        Returns
        -------
        np.ndarray
            Aggregated criterion grid [0,1]
        """
        if not component_grids:
            logger.warning(f"No component data to aggregate (criterion: {criterion_name})")
            return None
        
        grids = list(component_grids.values())
        names = list(component_grids.keys())
        weights = [component_weights.get(name, 1.0) for name in names]
        
        logger.info(f"  Aggregating {len(names)} components: {names}")
        
        result = self._weighted_geometric_mean(grids, weights)
        
        return result
    
    def aggregate_criteria(
        self,
        criterion_grids: Dict[str, np.ndarray],
        criterion_weights: Dict[str, float]
    ) -> np.ndarray:
        """
        Aggregate all criteria into final PFA
        
        Parameters
        ----------
        criterion_grids : Dict[str, np.ndarray]
            Dictionary of criterion name → aggregated grid
        criterion_weights : Dict[str, float]
            Dictionary of criterion name → weight
        
        Returns
        -------
        np.ndarray
            Final PFA grid [0,1]
        """
        if not criterion_grids:
            logger.error("No criterion data to aggregate")
            return None
        
        grids = list(criterion_grids.values())
        names = list(criterion_grids.keys())
        weights = [criterion_weights.get(name, 1.0) for name in names]
        
        logger.info(f"Aggregating {len(names)} criteria into final PFA")
        
        result = self._weighted_geometric_mean(grids, weights)
        
        return result
    
    @staticmethod
    def _weighted_geometric_mean(
        grids: List[np.ndarray],
        weights: List[float]
    ) -> np.ndarray:
        """
        Compute weighted geometric mean: prod(grid_i ^ weight_i)
        
        For aggregating [0,1] favorability scores
        
        Parameters
        ----------
        grids : List[np.ndarray]
            List of input grids [0,1]
        weights : List[float]
            List of weights (will be normalized to sum=1)
        
        Returns
        -------
        np.ndarray
            Aggregated grid [0,1]
        """
        # Stack grids
        stacked = np.stack(grids, axis=0)  # (n_grids, height, width) or (n_grids,)
        
        # Normalize weights to sum=1
        weights = np.array(weights, dtype=np.float32)
        weights = weights / np.sum(weights)
        
        # Weighted geometric mean using log space
        # prod(x_i ^ w_i) = exp(sum(w_i * log(x_i)))
        # Avoid log(0) by using maximum
        log_grids = np.log(np.maximum(stacked, 1e-10))
        
        # Reshape weights for broadcasting
        if stacked.ndim == 1:
            # 1D case
            weighted_log = log_grids * weights
        else:
            # Multi-dimensional case
            weighted_log = log_grids * weights[:, np.newaxis, np.newaxis]
        
        result = np.exp(np.sum(weighted_log, axis=0))
        
        # Clip to [0, 1]
        result = np.clip(result, 0, 1)
        
        return result
    
    @staticmethod
    def _apply_penalty(
        result: np.ndarray,
        penalty_grid: np.ndarray
    ) -> np.ndarray:
        """
        Apply penalty layer (veto mechanism)
        
        Penalty grid [0,1]:
        - penalty < 0.1 (close to salt) → veto (set result to 0)
        - Otherwise, reduce result proportionally: result = result * penalty
        
        Parameters
        ----------
        result : np.ndarray
            Main grid [0,1]
        penalty_grid : np.ndarray
            Penalty grid [0,1] (higher = better, lower = penalized)
        
        Returns
        -------
        np.ndarray
            Penalized grid [0,1]
        """
        penalty_grid = np.asarray(penalty_grid, dtype=np.float32)
        result = np.asarray(result, dtype=np.float32)
        
        # Veto: if penalty < threshold, set result to 0
        vetoed = penalty_grid < 0.1
        result[vetoed] = 0.0
        
        # Soft penalty: multiply result by penalty factor
        result = result * penalty_grid
        
        return np.clip(result, 0, 1)


if __name__ == "__main__":
    # Test voter
    voter = VoterVeto()
    
    # Create test grids
    grid1 = np.array([[0.2, 0.5], [0.8, 0.3]], dtype=np.float32)
    grid2 = np.array([[0.6, 0.7], [0.4, 0.9]], dtype=np.float32)
    
    result = voter.aggregate_layers(
        {'layer1': grid1, 'layer2': grid2},
        {'layer1': 0.6, 'layer2': 0.4},
        component_name="test_component"
    )
    
    print(f"\nResult shape: {result.shape}")
    print(f"Result range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"Result mean: {result.mean():.3f}")

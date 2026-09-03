"""
Layer Transformation Engine
Handles: none, log10, inverse, inverse_penalty transformations
Normalizes all outputs to [0, 1]
"""
import numpy as np
from typing import Callable, Dict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class LayerTransformer:
    """Apply transformations to layer data"""
    
    def __init__(self):
        """Initialize transformation registry"""
        self.transformations = {
            'none': self._transform_none,
            'log10': self._transform_log10,
            'inverse': self._transform_inverse,
            'inverse_penalty': self._transform_inverse_penalty,
        }
    
    def transform(self, data: np.ndarray, method: str, layer_name: str = "") -> np.ndarray:
        """
        Apply transformation to data and normalize to [0, 1]
        
        Parameters
        ----------
        data : np.ndarray
            Input array (assumed to be in raw units)
        method : str
            Transformation method: 'none', 'log10', 'inverse', 'inverse_penalty'
        layer_name : str, optional
            Layer name for logging
        
        Returns
        -------
        np.ndarray
            Transformed and normalized to [0, 1]
        """
        if method not in self.transformations:
            logger.warning(f"Unknown transformation: {method} (layer: {layer_name}), using 'none'")
            method = 'none'
        
        # Apply transformation
        transformed = self.transformations[method](data)
        
        # Normalize to [0, 1]
        normalized = self._normalize_to_01(transformed)
        
        logger.info(f"  Transform: {layer_name} → {method} → [{normalized.min():.3f}, {normalized.max():.3f}]")
        
        return normalized
    
    @staticmethod
    def _transform_none(data: np.ndarray) -> np.ndarray:
        """No transformation, use as-is"""
        return np.asarray(data, dtype=np.float32)
    
    @staticmethod
    def _transform_log10(data: np.ndarray) -> np.ndarray:
        """
        Log10 transformation (for porosity-permeability data)
        log10(x) where x > 0
        """
        data = np.asarray(data, dtype=np.float32)
        # Avoid log(0) and log(negative)
        data = np.maximum(data, 1e-10)
        return np.log10(data)
    
    @staticmethod
    def _transform_inverse(data: np.ndarray) -> np.ndarray:
        """
        Inverse transformation: 1/x (for distance-based layers)
        Close distance = high value
        """
        data = np.asarray(data, dtype=np.float32)
        # Avoid division by zero
        data = np.maximum(data, 1e-10)
        return 1.0 / data
    
    @staticmethod
    def _transform_inverse_penalty(data: np.ndarray) -> np.ndarray:
        """
        Inverse penalty: 1/x (for salt structures)
        Closer to salt = higher penalty value (after normalization, becomes lower favorability)
        """
        data = np.asarray(data, dtype=np.float32)
        # Avoid division by zero
        data = np.maximum(data, 1e-10)
        return 1.0 / data
    
    @staticmethod
    def _normalize_to_01(data: np.ndarray) -> np.ndarray:
        """
        Min-max normalize to [0, 1]
        
        Parameters
        ----------
        data : np.ndarray
            Input array
        
        Returns
        -------
        np.ndarray
            Normalized array in [0, 1]
        """
        data = np.asarray(data, dtype=np.float32)
        
        # Handle NaN values
        valid = ~np.isnan(data)
        if not valid.any():
            logger.warning("All NaN values, returning zeros")
            return np.zeros_like(data)
        
        data_min = np.nanmin(data[valid])
        data_max = np.nanmax(data[valid])
        
        if data_min == data_max:
            # All same value - set to 0.5 (middle)
            return np.full_like(data, 0.5)
        
        # Min-max normalization
        normalized = (data - data_min) / (data_max - data_min)
        normalized = np.clip(normalized, 0, 1)
        
        return normalized


if __name__ == "__main__":
    # Test transformations
    transformer = LayerTransformer()
    
    test_data = np.array([1, 2, 3, 4, 5], dtype=np.float32)
    
    print("\nTesting transformations:")
    for method in ['none', 'log10', 'inverse', 'inverse_penalty']:
        result = transformer.transform(test_data, method, f"test_{method}")
        print(f"  {method}: min={result.min():.3f}, max={result.max():.3f}")

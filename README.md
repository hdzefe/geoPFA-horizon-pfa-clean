````markdown
# GeoPFA - North German Basin Hydrothermal Favorability Assessment

Multi-criteria spatial modeling framework for prospectivity and favorability analysis (PFA) of geothermal resources across 18 geological horizons in the North German Basin.

## 🎯 Overview

**GeoPFA** implements a hierarchical, weighted multi-criteria decision analysis (MCDA) framework to assess geothermal energy potential. The system processes:

- **18 geological horizons** (economic + 17 geothermal)
- **4 main criteria**: Economic, Thermal, Geologic, Evidence-based
- **5-15 components per criterion** (demand, infrastructure, thermal properties, etc.)
- **40+ spatial layers** (population, heat demand, temperature, porosity, etc.)

### Key Features

✅ **Hierarchical Aggregation**: Layers → Components → Criteria → Final PFA  
✅ **Weighted Geometric Mean**: Accounts for compensatory effect  
✅ **Penalty/Veto Mechanism**: Salt structures and exclusion zones  
✅ **4 Transformation Methods**: none, log10, inverse, inverse_penalty  
✅ **Normalization to [0,5]**: Stakeholder-friendly favorability scale  
✅ **Audit Trail**: Full processing history and metadata tracking  

## 📁 Project Structure

```
geoPFA-horizon-pfa-clean/
├── config/
│   └── pfa_configuration_ngb.json       # Hierarchical config (18 horizons, weights, layers)
├── src/
│   ├── main.py                          # Entry point, pipeline orchestration
│   ├── config_loader.py                 # Load & validate configuration
│   ├── layer_transformer.py             # Apply transformations & normalize
│   ├── voter.py                         # Multi-criteria aggregation
│   ├── data_processor.py                # Main orchestrator
│   └── utils.py                         # Normalization, tracking, statistics
├── data/
│   ├── inputs/                          # CSV/GeoTIFF/NetCDF input layers
│   └── outputs/                         # PFA grids (numpy arrays, CSV, GeoTIFF)
├── tests/
│   └── test_*.py                        # Unit & integration tests
├── requirements.txt                     # Python dependencies
└── README.md                            # This file
```

## 🔧 Installation

### Prerequisites
- Python 3.8+
- pip or conda

### Setup

```bash
# Clone repository
git clone https://github.com/hdzefe/geoPFA-horizon-pfa-clean.git
cd geoPFA-horizon-pfa-clean

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Quick Start

### 1. Run Complete Pipeline

```bash
cd src
python main.py
```

This:
- Loads configuration from `config/pfa_configuration_ngb.json`
- Processes all 18 horizons
- Aggregates layers → components → criteria → final PFA
- Saves results to `data/outputs/`

### 2. Process Single Criterion

```python
from config_loader import ConfigLoader
from data_processor import PFADataProcessor
import numpy as np

# Initialize
processor = PFADataProcessor('config/pfa_configuration_ngb.json')

# Prepare sample data
layer_data = {
    'component_1': {
        'layer_1': np.random.rand(100, 100),
        'layer_2': np.random.rand(100, 100),
    }
}

# Process
criterion_grid, penalty_grid = processor.process_criterion(
    criterion_name='economic',
    layer_data_dict=layer_data
)

print(f"Result range: [{criterion_grid.min():.3f}, {criterion_grid.max():.3f}]")
```

### 3. Custom Processing

```python
from config_loader import ConfigLoader
from layer_transformer import LayerTransformer
from voter import VoterVeto

config = ConfigLoader('config/pfa_configuration_ngb.json')
transformer = LayerTransformer()
voter = VoterVeto()

# Transform layers
grid_1_transformed = transformer.transform(raw_grid_1, 'log10', 'porosity')
grid_2_transformed = transformer.transform(raw_grid_2, 'none', 'permeability')

# Aggregate
result = voter.aggregate_layers(
    {'porosity': grid_1_transformed, 'permeability': grid_2_transformed},
    {'porosity': 0.6, 'permeability': 0.4},
    component_name='thermal'
)
```

## 📊 Configuration Structure

The PFA hierarchy is defined in `config/pfa_configuration_ngb.json`:

```json
{
  "metadata": {
    "project": "North German Basin Geothermal PFA",
    "horizons": 18,
    "version": "1.0"
  },
  "criteria": {
    "economic": {
      "weight": 0.25,
      "components": {
        "demand": {
          "weight": 0.6,
          "layers": {
            "admin_areas_population": {
              "weight": 0.5,
              "transformation_method": "log10",
              "units": "persons",
              "source": "ZENSUS_2011"
            },
            "geotis_heat_demand": {
              "weight": 0.5,
              "transformation_method": "none",
              "units": "MWh/a"
            }
          }
        }
      }
    },
    "geothermal_detfurth_fm": {
      "weight": 1.0,
      "horizon_name": "Detfurth Formation",
      "age_ma": 200.5,
      "age_period": "Jurassic",
      "stratigraphic_unit": "Lower Jurassic",
      "components": { ... },
      "penalty_layer": {
        "name": "salt_diapirs",
        "description": "Distance to active salt structures"
      }
    }
  }
}
```

## 🔄 Processing Pipeline

### Transformation Methods

| Method | Use Case | Formula | Output |
|--------|----------|---------|--------|
| `none` | Direct favorability scores | x | [0,1] |
| `log10` | Porosity, permeability data | log₁₀(x) | [0,1] |
| `inverse` | Distance-based layers | 1/x | [0,1] |
| `inverse_penalty` | Salt structures, penalties | 1/x | [0,1] |

### Aggregation: Weighted Geometric Mean

**At all levels** (layers → components → criteria):

```
Result = ∏ᵢ (gridᵢ)^(wᵢ)  where Σwᵢ = 1
       = exp(Σ wᵢ · ln(gridᵢ))
```

**Advantages:**
- Compensatory: high values can offset moderate values
- Non-linear: emphasizes consistent favorability
- Normalized output: always in [0,1]

### Penalty/Veto Mechanism

If penalty layer exists (e.g., salt structures):
- **Hard veto**: penalty < 0.1 → set result to 0
- **Soft penalty**: result = result × penalty

## 📈 Output

### Generated Files

```
data/outputs/
├── pfa_raw.npy              # Raw PFA grid [0,1]
├── pfa_scaled.npy           # Scaled PFA grid [0,5]
├── pfa_raw.csv              # CSV export (raw)
└── pfa_scaled.csv           # CSV export (scaled)
```

### Favorability Scale [0,5]

| Score | Interpretation |
|-------|-----------------|
| 0.0   | Highly unfavorable (vetoed) |
| 1.0   | Very low favorability |
| 2.5   | Moderate favorability |
| 4.0   | High favorability |
| 5.0   | Extremely favorable (best sites) |

### Audit Trail

Full processing history accessible via:

```python
processor.print_audit_trail(verbose=True)
processor.get_summary()
```

## 🧪 Testing

```bash
# Run unit tests
pytest tests/

# Run with verbose output
pytest tests/ -v

# Coverage report
pytest tests/ --cov=src
```

## 📚 Module Documentation

### `config_loader.py`
- **ConfigLoader**: Load, validate, and query PFA configuration
- Methods: `get_criterion()`, `get_components()`, `get_layers()`, `print_hierarchy()`

### `layer_transformer.py`
- **LayerTransformer**: Apply transformations to individual layers
- Methods: `transform()`, `_transform_log10()`, `_transform_inverse()`, `_normalize_to_01()`

### `voter.py`
- **VoterVeto**: Multi-criteria aggregation with penalty mechanism
- Methods: `aggregate_layers()`, `aggregate_components()`, `aggregate_criteria()`, `_weighted_geometric_mean()`, `_apply_penalty()`

### `data_processor.py`
- **PFADataProcessor**: Orchestrate complete pipeline
- Methods: `process_criterion()`, `process_all_criteria()`, `finalize_pfa()`, `print_audit_trail()`

### `utils.py`
- **Normalizer**: Scale grids to [0,1] or [0,5], compute statistics
- **MetadataTracker**: Track audit trail and metadata through pipeline
- **GridStats**: Validate and report grid information

## 🔗 Input Data Format

### Expected Structure

```
data/inputs/
├── economic/
│   ├── admin_areas_population.tif
│   ├── geotis_heat_demand.tif
│   └── ...
├── detfurth_fm/
│   ├── subsurface_temperature.tif
│   ├── porosity.tif
│   └── ...
└── ...
```

### Supported Formats
- **Raster**: GeoTIFF, NetCDF (via rasterio)
- **Tabular**: CSV, Parquet (via pandas)
- **In-memory**: NumPy arrays, GeoPandas GeoDataFrames

### Loading Example

```python
import rasterio
import numpy as np

# Load GeoTIFF
with rasterio.open('data/inputs/economic/admin_areas_population.tif') as src:
    grid = src.read(1)  # NumPy array

# Use in processor
layer_data = {'demand': {'admin_areas_population': grid}}
```

## 🎓 Methodology

GeoPFA implements a **Hierarchical Multi-Criteria Evaluation (MCE)** framework:

1. **Layer Transformation**: Normalize diverse data to [0,1] using method-specific transformations
2. **Spatial Aggregation**: Combine layers via weighted geometric mean at each hierarchy level
3. **Multi-Horizon Modeling**: Process all 18 horizons independently
4. **Final Integration**: Aggregate criteria into unified favorability grid
5. **Scale Conversion**: Normalize to [0,5] for stakeholder communication

### References

- Yalcin & Reis (2014): GIS-based landslide susceptibility mapping
- Malczewski (1999): GIS and multicriteria decision analysis
- Saaty (1980): Analytical Hierarchy Process (AHP)

## ⚙️ Configuration Parameters

### Key Weights (Adjustable)

Edit `config/pfa_configuration_ngb.json` to adjust:

```json
"criteria": {
  "economic": {
    "weight": 0.25  // ← Adjust criterion importance
  }
}
```

### Penalty Threshold

Adjust in `src/data_processor.py`:

```python
voter = VoterVeto(veto_threshold=0.1)  # ← Change veto threshold
```

## 🐛 Troubleshooting

### Import Errors

```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall
```

### Config Not Found

```
FileNotFoundError: Config file not found: config/pfa_configuration_ngb.json
```

**Solution**: Ensure config file exists in correct location (relative to `src/main.py`)

### NaN Values in Output

Check input data for missing/invalid values:

```python
import numpy as np
grid = np.load('data/inputs/layer.npy')
print(f"NaN count: {np.isnan(grid).sum()}")
print(f"Valid range: [{np.nanmin(grid)}, {np.nanmax(grid)}]")
```

## 📝 License

[Add your license here]

## 👥 Contributors

- **hdzefe** - Lead developer

## 📧 Contact

For questions or collaboration: [Add contact info]

## 🔍 Citation

If you use GeoPFA in research, please cite:

```bibtex
@software{geopfa2024,
  title={GeoPFA: Multi-Criteria Geothermal Favorability Assessment},
  author={hdzefe},
  year={2024},
  url={https://github.com/hdzefe/geoPFA-horizon-pfa-clean}
}
```

---

**Last Updated**: 2024  
**Status**: Active Development
````

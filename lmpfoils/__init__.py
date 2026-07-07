"""lmpfoils: reusable analysis code for LiMgPO4 optically stimulated luminescence
foil-detector proton dosimetry.

Ported from the exploratory ``ifj_lmp_foils`` research repository into a
tested, importable package. Provides:

- :mod:`lmpfoils.paths` -- repository-root discovery and standard data
  directory paths (used by the ``notebooks/`` sequence)
- :mod:`lmpfoils.geometry` -- circle geometry and image masking utilities
- :mod:`lmpfoils.image_io` -- TIFF reading, median filtering, background
  subtraction, metadata timestamp parsing
- :mod:`lmpfoils.detector` -- automatic detector (foil) circle finding
- :mod:`lmpfoils.alignment` -- angular alignment of detector circles
- :mod:`lmpfoils.preprocessing` -- raw-TIFF-to-interim preprocessing
  (constant-background removal, detector-circle finding, angular alignment)
- :mod:`lmpfoils.dataset` -- building analysis dataframes from raw
  micro-manager image folders
- :mod:`lmpfoils.pipeline` -- background subtraction, flat-field
  correction, re-centering and de-rotation of per-foil images
- :mod:`lmpfoils.comparison` -- comparing results to reference curves
  (percent-error helpers)
- :mod:`lmpfoils.reference_data` -- reading the bundled Markus-chamber /
  PHITS Monte Carlo reference tables (``data/reference/article-plots.h5``)
- :mod:`lmpfoils.fitting` -- relative-efficiency curve fitting (requires the
  optional ``lmfit`` dependency)
- :mod:`lmpfoils.plotting` -- diagnostic and article-style plotting helpers
"""

__version__ = "0.1.0"

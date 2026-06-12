# DocMaster public folder

This folder is intentionally no longer the primary UI.

DocMaster now runs as a SLAIHub-native PyQt desktop module through:

```python
from documaster.documaster import DocumasterWindow
```

The old browser frontend can be kept only as an optional cloud/local preview. The main GUI is implemented in `documaster/documaster.py` and uses the same service layer as the optional Flask AI routes.

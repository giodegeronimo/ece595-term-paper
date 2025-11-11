# Experiments

Interactive notebooks live here, but the canonical implementation has moved to `src/navit_rf`.
Import helpers via:

```python
import navit_rf as nrf
```

After `pip install -e .`, notebooks can reuse the same `ViTVelocity`, NaViT packing collates, and trainer utilities as the CLI scripts. Keep generated checkpoints under `experiments/navit_rf/outputs/` (gitignored).

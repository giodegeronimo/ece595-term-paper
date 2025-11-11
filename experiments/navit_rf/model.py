"""
Legacy compatibility shim.

The canonical implementation now lives under src/navit_rf/model.py. Importing from
this module keeps older notebooks working while the project uses a src/ layout.
"""

from navit_rf.model import *  # noqa: F401,F403

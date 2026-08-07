# RedditDaily-Bot — Source Package
# All pipeline components are importable from src.*

import PIL.Image
if not hasattr(PIL.Image, "ANTIALIAS"):
    PIL.Image.ANTIALIAS = PIL.Image.Resampling.LANCZOS

__version__ = "1.0.0"

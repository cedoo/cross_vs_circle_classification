import numpy as np
from pathlib import Path

pathuj = Path("data/")
print(list(pathuj.glob('*/*/*.bmp')))

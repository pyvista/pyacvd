from importlib.metadata import PackageNotFoundError, version

from pyacvd.clustering import Clustering

try:
    __version__ = version("pyacvd")
except PackageNotFoundError:
    __version__ = "unknown"


__all__ = ["Clustering", "__version__"]

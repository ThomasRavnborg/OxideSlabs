from pathlib import Path


__version__ = '1.0.1'
__all__ = ('datapath',)


def datapath() -> Path:
    """Path to the directory containing the setup files."""
    return Path(__file__).parent / 'setups'

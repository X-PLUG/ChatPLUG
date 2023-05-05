"""
https://www.python.org/dev/peps/pep-0440
"""

__version_major__ = 0
__version_minor__ = 5
__version_micro__ = 0
__modifier__ = ''

__version__ = '.'.join(
    map(str,
        [__version_major__,
         __version_minor__,
         __version_micro__])
    ) + __modifier__

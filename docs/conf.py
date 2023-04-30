import os
import sys

sys.path.insert(0, os.path.abspath("."))

project = 'ChatPLUG'
copyright = '2017-2023, XPLUG Team'
author = 'XPLUG Team'
extensions = [
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages"
]

myst_enable_extensions = [
    "colon_fence",
]
myst_url_schemes = ["http", "https", ]

source_suffix = ['.rst', '.md']

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']

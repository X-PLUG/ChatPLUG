import os
import sys
sys.path.insert(0, os.path.abspath("."))

project = 'ChatPLUG'
copyright = '2023, XPLUG Team'
author = 'XPLUG Team'
extensions = [
    "sphinx.ext.intersphinx",
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx.ext.githubpages",
    "sphinx.ext.autosectionlabel",
]

myst_enable_extensions = [
    "colon_fence",
]
myst_url_schemes = ["http", "https", ]

# Make sure the target is unique
autosectionlabel_prefix_document = True

source_suffix = ['.rst', '.md']


templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']


html_theme = 'sphinx_rtd_theme'
html_static_path = ['_static']
html_css_files = ['css/custom.css']

def setup (app):
    app.add_css_file('css/custom.css')

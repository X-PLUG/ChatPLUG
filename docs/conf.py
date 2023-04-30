
project = 'ChatPLUG'
copyright = '2017-2023, XPLUG Team'
author = 'XPLUG Team'

extensions = [
    "myst_parser",
]
myst_enable_extensions = [
    "colon_fence",
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']

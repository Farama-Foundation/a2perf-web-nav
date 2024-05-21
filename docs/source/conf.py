# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import sys
sys.path.insert(0, os.path.abspath('../..'))
print('\n\n\n\n\n')
print(os.path.abspath('../..'))
print('\n\n\n\n\n')


project = 'web_nav_docs'
copyright = '2024, Korneel Van den Berghe'
author = 'Korneel Van den Berghe'
release = '17/05/2024'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.autodoc',
    'myst_parser',
]

templates_path = ['_templates']
exclude_patterns = []



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']
# sphinx-build -b html /home/korneel/farama_a2perf/A2Perf/a2perf/domains/web_navigation/docs/source /home/korneel/farama_a2perf/A2Perf/a2perf/domains/web_navigation/docs/build
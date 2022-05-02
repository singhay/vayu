# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.append(os.path.abspath('../'))

import vayu

# -- Project information -----------------------------------------------------

project = 'Vayu'
# copyright = '2020, Ayush Singh'
# author = 'Ayush Singh'
copyright = vayu.__copyright__
author = vayu.__author__

# The full version, including alpha/beta/rc tags
release = vayu.__version__


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.napoleon',
    'sphinx.ext.intersphinx',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'vayu/resources',
                    '.pytest_cache', '.eggs', 'vayu/*.pkl', 'vayu/*.bin', '*.json', '*.md']


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'alabaster'
pygments_style = 'sphinx'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
source_suffix = ['.rst', '.md']

# The master toctree document.
master_doc = 'index'

# -- Options for HTMLHelp output ---------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = project + '-doc'

autodoc_member_order = 'groupwise'
autoclass_content = 'both'
# the options are fixed and will be soon in release,
#  see https://github.com/sphinx-doc/sphinx/issues/5459
autodoc_default_options = {
    'members': None,
    'methods': None,
    # 'attributes': None,
    'special-members': '__call__',
    'exclude-members': '_abc_impl',
    'show-inheritance': True,
    'private-members': True,
    'noindex': True,
}

# -- Options for intersphinx extension ---------------------------------------

intersphinx_mapping = {
    'python': ('https://docs.python.org/3', None),
    'torch': ('https://pytorch.org/docs/stable/', None),
    'numpy': ('https://docs.scipy.org/doc/numpy/', None),
    'pytorch_lightning': ('https://pytorch-lightning.readthedocs.io/en/stable/', None),
    'gensim': ('https://radimrehurek.com/gensim/', None),
    'transformers': ('https://huggingface.co/transformers/', None)
}

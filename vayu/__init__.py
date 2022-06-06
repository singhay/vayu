"""Root package info."""

__name__ = 'vayu'
__version__ = '0.1.2'
__author__ = 'Ayush Singh'
__author_email__ = 'ayush.singh@evicore.com'
__service__ = 'CDR'
__license__ = 'Apache-2.0'
__copyright__ = 'Copyright (c) 2020, %s.' % __author__
__homepage__ = 'https://gitlab.qpidhealth.net/qaas/vayu'
__doc_homepage__ = 'http://10.205.63.29/docs/vayu/stable'
__docs__ = "Vayu is a framework built on top of pytorch for training binary classification models."
__long_docs__ = """
Vayu is a deep learning framework built keeping strictly modeling in mind. 
Support for named entity recognition (NER), language, knowledge graph models as well as custom tokenizer that 
supports a whitelist of tokens is planned for future releases which are now standalone scripts.
Documentation
-------------
- %s
""" % __doc_homepage__

import logging

_logger = logging.getLogger("vayu")
_logger.addHandler(logging.StreamHandler())
_logger.setLevel(logging.INFO)

# for compatibility with namespace packages
__import__('pkg_resources').declare_namespace(__name__)
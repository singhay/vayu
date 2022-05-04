from setuptools import setup, find_packages
import vayu

# Generated using pipreqs https://github.com/bndr/pipreqs
install_requires = [
    'numpy==1.18.1',
    'pandas==1.0.3',
    'scikit-learn==0.23.1',
    'torch==1.5.0',
    'transformers==3.0.2',
    'pytorch-nlp>=0.5.0',
    'pytorch-lightning==0.7.6',
    'iso8601',
    'gensim>=3.8.3',
    'hydra_core==0.11.3',
]
# Keep this in alphabetical sort order

tests_require = [
    'pyarrow;platform_system!="Windows"',
    'pytest==5.4.3',
    'pytest-cov==2.9.0'
]

setup(
    name=vayu.__name__,
    version=vayu.__version__,
    description=vayu.__docs__,
    long_description=vayu.__long_docs__,
    url=vayu.__homepage__,
    packages=find_packages(),
    install_requires=install_requires,
    setup_requires=['pytest-runner'],
    tests_require=tests_require,
    python_requires='>=3.6'
)

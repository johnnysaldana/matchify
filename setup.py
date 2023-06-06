from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name='matchify',
    version='1.3.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'tqdm',
        'textdistance',
        'jellyfish',
        'python-Levenshtein',
        'gensim',
        'nameparser',
        'phonenumbers',
        'usaddress',
        'python-dateutil',
        'jinja2',
        'click',
        'matplotlib',
        'faker',
        'boto3',
        'pyarrow',
        'sqlalchemy',
    ],
    extras_require={
        'deep': [
            'torch>=2.0',
            'transformers>=4.29',
            'sentence-transformers>=2.2',
            # Siamese fine-tuning needs these
            'datasets',
            'accelerate>=0.20',
        ],
        'dev': [
            'pytest>=7.0',
            'ruff>=0.0.260',
        ],
    },
    entry_points={
        'console_scripts': [
            'matchify=matchify.cli:cli',
        ],
    },
    author='Johnny Saldana',
    author_email='johnny.saldana99@gmail.com',
    description='Entity resolution / record linkage with five model implementations.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/johnnysaldana/matchify',
    keywords=['entity resolution', 'deduplication', 'record linkage', 'machine learning'],
    license='MIT',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    python_requires='>=3.8',
)

from setuptools import setup, find_packages

setup(
    name='matchify',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy',
        'scikit-learn',
        'boto3',
        'pyarrow'
    ],
    author='Your Name',
    author_email='your.email@example.com',
    description='A package for entity resolution using various models and strategies.',
    url='https://github.com/yourusername/matchify',
    keywords=['entity resolution', 'deduplication', 'machine learning'],
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
)

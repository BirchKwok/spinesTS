from setuptools import find_packages, setup

with open("README.md", "r") as fh:
  long_description = fh.read()

setup(
    name='spinesTS',
    version="0.3.13",
    description='spinesTS, a powerful timeseries toolsets.',
    keywords='machine learning',
    packages=find_packages(),
    long_description=long_description,
    long_description_content_type="text/markdown",
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    url='https://github.com/BirchKwok/spinesTS',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=[
        'scikit-learn>=1.0.2',
        'torch>=2.1.0',
        'scipy>=1.7.0',
        'numpy>=1.17.0',
        'pandas>=2.0.0',
        'tabulate>=0.8',
        'matplotlib>=3.5.1',
        'spinesUtils>=0.3.5'
    ],
    zip_safe=False,
    include_package_data=True
)

from setuptools import Extension, dist, find_packages, setup


setup(
    name='spinesTS',
    version="0.0.1",
    description='spinesTS, a powerful timeseries toolsets.',
    keywords='computer vision',
    packages=find_packages(),
    long_description='./README.md',
    classifiers=[
        'License :: OSI Approved :: Apache Software License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    url='https://github.com/BirchKwok/spinesTS',
    author='Birch Kwok',
    author_email='birchkwok@gmail.com',
    install_requires=[
        'scikit-learn>=1.0.2',
        'torch>=1.4',
        'scipy>=1.7.0',
        'numpy>=1.17.0',
        'pandas>=1.0.0',
        'tabulate>=0.8'
    ],
    zip_safe=False,
    include_package_data=True
)

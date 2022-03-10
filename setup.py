from setuptools import Extension, dist, find_packages, setup

setup(
    name='spinesTS',
    version="0.0.1a3",
    description='spinesTS, a timeseries toolsets.',
    keywords='computer vision',
    packages=find_packages(),
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
      'tensorflow>=2.5',
      'torch>=1.4',
      'scipy>=1.7.0',
      'numpy>=1.17.0',
      'pandas>=1.0.0'
    ],
    zip_safe=False,
    include_package_data=True
)

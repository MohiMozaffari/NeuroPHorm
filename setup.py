from setuptools import setup, find_packages

setup(
    name='neurophorm',
    version='0.1.0',
    author='Mohaddeseh Mozaffari',
    author_email='mohaddeseh.mozaffarii@gmail.com',
    description='A Python package for topological brain network analysis using persistent homology',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/mohimozaffari/neurophorm',
    packages=find_packages(),
    install_requires=[
        'numpy==1.23.5',
        'pandas==1.5.3',
        'scipy==1.10.1',
        'matplotlib==3.7.1',
        'seaborn==0.12.2',
        'scikit-learn==1.2.2',
        'gtda==0.3.1'
    ],
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
)

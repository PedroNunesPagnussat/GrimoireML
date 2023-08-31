from setuptools import setup, find_packages

setup(
    name='grimoireml',
    version='0.1.3',
    packages=find_packages(),
    install_requires=[
        'numpy>=1.18.0',  # Replace with the minimum version you require
        'scipy>=1.4.0'   # Replace with the minimum version you require
    ],
    author='Pedro',
    author_email='grimoireml1@gmail.com',
    description='A machine learning library',
)

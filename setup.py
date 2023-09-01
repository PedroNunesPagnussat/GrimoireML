from setuptools import setup, find_packages
import os

# Read the content of the README.md file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='grimoireml', 
    version='0.0.0.1',  
    author='Pedro Pagnussat',
    author_email='grimoireml1@gmail.com', 
    description='A machine learning library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/PedroNunesPagnussat/GrimoireML',  
    packages=find_packages(),
    classifiers=[
        'Development Status :: 3 - Alpha',  # Choose either "3 - Alpha", "4 - Beta", "5 - Production/Stable"
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
    ],
    install_requires=[
        'numpy==1.21.6', 
        'scipy==1.7.3',   
    ],
    python_requires='>=3.6',  

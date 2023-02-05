from setuptools import setup
from codecs import open
from os import path

here = path.abspath(path.dirname(__file__))

with open(path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='mmcq_numba',
    packages=['mmcq_numba'], 

    version='0.1.0', 

    license='Apache-2.0 license', 

    install_requires=['numba'], 

    author='hirokic5', 
    author_email='kanbac5@gmail.com', 

    url='https://github.com/hirokic5/mmcq_numba.git', 

    description='Analyze dominant colors in image with MMCQ algorithm',
    long_description=long_description, 
    long_description_content_type='text/markdown', 
    keywords='mmcq dominant-color', 
)
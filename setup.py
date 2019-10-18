from distutils.core import setup, Extension
import numpy

EXTENSIONS = [
    Extension('blocksmodule',
              ['blocksmodule.c'],
              include_dirs=[numpy.get_include()]),
]

setup(
    name='Map Dynamics',
    version='1.0',
    author='Geoffrey Woollard',
    # author_email='',
    # description='',
    # long_description='',
    # url='',
    # packages=PACKAGES,
    #package_dir=PACKAGE_DIR,
    # package_data=PACKAGE_DATA,
    ext_modules=EXTENSIONS,
)
from distutils.core import setup, Extension
from glob import glob


try:
    import numpy
except ImportError:
    sys.stderr.write('numpy is not installed, you can find it at: '
                     'http://www.numpy.org/\n')
    sys.exit()


# module1 = Extension('rtbtools',
#                     sources = ['rtbtools.c'])

# setup (name = 'map_dynamics',
#        version = '1.0',
#        description = 'description',
#        ext_modules = [module1])

EXTENSIONS = [
    Extension('blocksmodule',
              ['blocksmodule.c'],
              include_dirs=[numpy.get_include()]),
#     Extension('prody.dynamics.smtools',
#               glob(join('prody', 'dynamics', 'smtools.c')),
#               include_dirs=[numpy.get_include()]),
# #    Extension('prody.dynamics.saxstools',
# #              glob(join('prody', 'dynamics', 'saxstools.c')),
# #              include_dirs=[numpy.get_include()]),
#     Extension('prody.sequence.msatools',
#               [join('prody', 'sequence', 'msatools.c'),],
#               include_dirs=[numpy.get_include()]),
#     Extension('prody.sequence.msaio',
#               [join('prody', 'sequence', 'msaio.c'),],
#               include_dirs=[numpy.get_include()]),
#     Extension('prody.sequence.seqtools',
#               [join('prody', 'sequence', 'seqtools.c'),],
#               include_dirs=[numpy.get_include()]),
]

setup(
    name='ProDy',
    version='1.0',
    author='Geoffrey Woollard',
    # author_email='',
    # description='A Python Package for Protein Dynamics Analysis',
    # long_description=long_description,
    # url='http://www.csb.pitt.edu/ProDy',
    # packages=PACKAGES,
    #package_dir=PACKAGE_DIR,
    # package_data=PACKAGE_DATA,
    ext_modules=EXTENSIONS,
    # license='',
    # keywords=('protein, dynamics, elastic network model, '
    #           'Gaussian network model, anisotropic network model, '
    #           'essential dynamics analysis, principal component analysis, '
    #           'Protein Data Bank, PDB, GNM, ANM, SM, PCA'),
    # classifiers=[
    #              'Development Status :: 5 - Production/Stable',
    #              'Intended Audience :: Education',
    #              'Intended Audience :: Science/Research',
    #              'License :: OSI Approved :: MIT License',
    #              'Operating System :: MacOS',
    #              'Operating System :: Microsoft :: Windows',
    #              'Operating System :: POSIX',
    #              'Programming Language :: Python',
    #              'Programming Language :: Python :: 2',
    #              'Programming Language :: Python :: 3',
    #              'Topic :: Scientific/Engineering :: Bio-Informatics',
    #              'Topic :: Scientific/Engineering :: Chemistry',
    #             ],
    #scripts=SCRIPTS,
    # entry_points = {
    #     'console_scripts': SCRIPTS,
    # },
    # install_requires=['numpy>=1.10', 'biopython', 'pyparsing'],
    #provides=['ProDy ({0:s})'.format(__version__)]
)
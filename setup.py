from setuptools import setup

setup(
    name='pysut',
    version="0.1",
    packages=["pysut", "pysut.tests"],
    author="Stefan Pauliak",
    description="Python class for efficient handling of supply and use tables (SUTs)",
    author_email="stefan.pauliuk@ntnu.no",
    license=open('LICENSE').read(),
    install_requires=["numpy","scipy"],
    long_description=open('README.md').read(),
    url="https://github.com/stefanpauliuk/pySUT",
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',
        'Operating System :: MacOS :: MacOS X',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: POSIX',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Information Analysis',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],
)

pySUT
=====

Python class for efficient handling of supply and use tables (SUTs)

Created on Mon Jun 30 17:21:28 2014

@author: stefan pauliuk and Guillaume Majeau-Bettez, NTNU Trondheim, Norway <br>
with contributions from <br>
Konstantin Stadler, NTNU, Trondheim, Norway<br>
Chris Mutel, PSI, Villingen, CH <br>

<b>Dependencies:</b> <br>
numpy >= 1.9<br>
scipy >= 0.14<br>

<br>
<b>Tutorial:</b><br>
https://github.com/cmutel/pySUT/blob/master/Doc/pySUT_Documentation.ipynb
<b>Documenation of all methods and functions:</b><br>
http://htmlpreview.github.com/?https://github.com/stefanpauliuk/pySUT/blob/master/Doc/pysut.html
<br>

<b> Below, a quick installation guide and a link to the tutorial are provided:</b><br><br>

<b>a) Installation as package:</b> <br>
Pull package via git pull or download as .zip file and unpack. Choose a convenient location (Here: 'C:\MyPythonPackages\'). Then open a console, change to the directory ../pySUT-master/, and install the package from the command line: <br>
> python setup.py install 

This makes the package available to Python. At any other place in a system with the same python installation, pydsm is now ready to be imported simply by <br>
> import pysut

This setup also allows us to run the unit test: <br>

> import unittest

> import pysut

> import pysut.tests

> unittest.main(pysut.tests, verbosity=2)

Or, to run a specific test

> unittest.main(pydsm.tests.test_allocations_constructs, verbosity=2)

<br>
<b>b) Manual installation, by modifying the python path</b><br>
Pull package via git pull or download as .zip file and unpack. Choose a convenient location (Here: 'C:\MyPythonPackages\'). Then include in your code the following lines <br>
> import sys 

> sys.path.append('C:\\MyPythonPackages\\pySUT-master\\pydsm\\') 

> from pysut import SupplyUseTable



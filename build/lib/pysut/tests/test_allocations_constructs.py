# -*- coding: utf-8 -*-
"""
Created on Mon Aug 11 16:19:39 2014

@author: Stefan Pauliuk, NTNU
Guillaume Majeau-Bettez, NTNU Trondheim, Norway
"""
from __future__ import division
from .. import SupplyUseTable # remove and import the class manually if this unit test is run as standalone script
from .. import pysut # remove and import the class manually if this unit test is run as standalone script
import numpy as np
import numpy.testing as npt
import unittest

###############################################################################
class TestAllocationsConstructs(unittest.TestCase):
    """ Unit test class for allocations and constructs"""

    ## DEFINE SIMPLE TEST CASES
    def setUp(self):
        """
        We define simple Supply and Use Table systems to test all allocations
        and constructs.

        We first define a Supply and traceable Use Table system, with three
        products (i, j, k) and four industries (I, J1, J2, and K).

        Then, by aggregation we generate a Supply and untraceable Use table
        system (SuUT).

        To test commodity-technology construct (CTC) and the
        byproduct-technology construct (BTC), we also generate a square SUT
        (Va, Ua, Ga), with three products and three industries (I, J, K), by
        further aggregation.


        """

        # absolute tolerance for assertion tests (rounding error)
        self.atol = 1e-08

        # CASE 0
        #---------

        # Defining labels for industries, commodities and factors of prod.
        self.l_ind = np.array([['I', 'J1', 'J2', 'K']], dtype=object).T
        self.l_com = np.array([['i', 'j', 'k']], dtype=object).T
        self.l_ext = np.array([['CO2', 'CH4']], dtype=object).T

        # dimensions
        self.ind = len(self.l_ind)
        self.com = len(self.l_com)

        # labels for traceable flows
        self.l_tr = list()
        for i in self.l_ind:
            for j in self.l_com:
                self.l_tr.append(j + '_{' + i + '}')

        # Supply table
        self.V = np.array([[2, 0, 0, 0],
                           [1, 1, 3, 0],
                           [0, 0, 0, 11]], dtype=float)

        # Traceable use table
        self.Ut = np.array(np.zeros((4, 3, 4), float))
        self.Ut[3, 2, 0] = 4    # use of k from K by I
        self.Ut[3, 2, 1] = 0.75    # use of k from K by J1
        self.Ut[3, 2, 2] = 2    # use of k from K by J2
        self.Ut[0, 1, 3] = 0.25    # use of j from I by K
        self.Ut[2, 1, 3] = 0.5     # use of j from J2 by K

        # Untraceable use table
        self.Uu = np.array(sum(self.Ut, 0))
        """array([[ 0.  ,  0.  ,  0.  ,  0.  ],
                  [ 0.  ,  0.  ,  0.  ,  0.75],
                  [ 4.  ,  0.75,  2.  ,  0.  ]])"""


        # Use of factors of production by industries
        self.F = np.array([
            [10,    4,    15,    18],
            [0,    0,    1,    0]
            ], dtype=float)


        # Intensive properties used in partitioning
        self.PSI = np.array([
        #       I       J1      J2      K
            [0.1,     0.1,     0.1,     0.1],     # i
            [0.2,     0.2,     0.2,     0.2],     #j
            [0.3,     0.3,     0.3,     0.3],     # k
            ])

        # Alternate activity, used in AAA/AAC
        self.Gamma = np.array([
        #    i        j        k
            [1,       0,       0],       # I
            [0,       0,       0],       # J1
            [0,       1,       0],       # J2
            [0,       0,       1]        # K
            ])

        # Identifies primary product of each industry
        self.E_bar = np.array([
        #    I      J1      J2      K
            [1,       0,       0,       0],     #  i
            [0,       1,       1,       0],     #  j
            [0,       0,       0,       1],     #  k
            ])

        # Substitutability between products
        self.Xi = np.array([
            [1,	0,	0],
            [0,	0,	0],
            [0,	0.3,	1]
            ])

        # Square SUT
        self.Va = self.V.dot(self.E_bar.T)
        self.Ua = self.Uu.dot(self.E_bar.T)
        self.Fa = self.F.dot(self.E_bar.T)

        # Case 1: 3 regions, 2 industries, 2 products (square) 
        self.V_3r2i2p = np.array(
                      #I   J     I   J       I   J
                     [[9., 0.,   0., 0.,     0., 0.],    # i
                      [3., 3.,   0., 0.,     0., 0.],    # j
                                                         #
                      [0., 0.,   4., 4.,     0., 0.],    # i
                      [0., 0.,   0., 0.,     0., 0.],    # j  <-- no production
                                                         #
                      [0., 0.,   0., 0.,     2., 0.],    # i
                      [0., 0.,   0., 0.,     1., 2.]])   # j
                     #               ^
                     #               |
                     #               "Fake J", does not produce j, prim prod i

        self.E_bar_3r2i2p = np.array(
                         [[1, 0, 0, 0, 0, 0],
                          [0, 1, 0, 0, 0, 0],
                          [0, 0, 1, 1, 0, 0],
                          [0, 0, 0, 0, 0, 0],
                          [0, 0, 0, 0, 1, 0],
                          [0, 0, 0, 0, 0, 1]])

        # Case 2: 3 regions, 2 industries, 3 products:
        #--------------------------------------------

        # No production of j in Canada
        # No production of i or k in Norway
        # No production of j in US

        self.V_3r2i3p = np.array([
                         #   Ca   Ca     No   No     US   US
                         #   I    J      I    J      I    J
                         [   4.,  0.,    0.,  0.,    0.,  0.,  ],     #i  Ca
                         [   0.,  0.,    0.,  0.,    0.,  0.,  ],     #j  Ca
                         [   0.,  3.,    0.,  0.,    0.,  0.,  ],     #k  Ca
                         #
                         [   0.,  0.,    0.,  0.,    0.,  0.,  ],     #i  No
                         [   0.,  0.,    3.,  0.,    0.,  0.,  ],     #j  No
                         [   0.,  0.,    0.,  0.,    0.,  0.,  ],     #k  No
                         #
                         [   0.,  0.,    0.,  0.,    8.,  0.,  ],     #i  US
                         [   0.,  0.,    0.,  0.,    0.,  0.,  ],     #j  US
                         [   0.,  0.,    0.,  0.,    0.,  9.,  ]])    #k  US

        self.V_3r2i3p_coprod = self.V_3r2i3p.copy()
        self.V_3r2i3p_coprod[1,0] = 1 # I Ca coproduces j
        self.V_3r2i3p_coprod[-1,4] = 1 # I US coproduces k

        self.E_bar_3r2i3p = np.array([
                         #
                         #  J_No des not produce anything, no primary product
                         #
                         #   Ca   Ca     No   No     US   US
                         #   I    J      I    J      I    J
                         [   1,  0,    0,  0,    0,  0,  ],     #i  Ca
                         [   0,  0,    0,  0,    0,  0,  ],     #j  Ca
                         [   0,  1,    0,  0,    0,  0,  ],     #k  Ca
                         #
                         [   0,  0,    0,  0,    0,  0,  ],     #i  No
                         [   0,  0,    1,  0,    0,  0,  ],     #j  No
                         [   0,  0,    0,  0,    0,  0,  ],     #k  No
                         #
                         [   0,  0,    0,  0,    1,  0,  ],     #i  US
                         [   0,  0,    0,  0,    0,  0,  ],     #j  US
                         [   0,  0,    0,  0,    0,  1,  ]])    #k  US


        self.U_3r2i3p = np.array([
                         #   Ca  Ca    No  No     US   US
                         #   I   J     I   J      I    J
                         [  .0, .1,   .5,  0,    0,  0,  ],     #i  Ca
                         [   0,  0,    0,  0,    0,  0,  ],     #j  Ca
                         [  .2, .0,    0,  0,    0,  0,  ],     #k  Ca
                         #
                         [   0,  0,    0,  0,    0,  0,  ],     #i  No
                         [  .3,  0,   .0,  0,   .5,  0,  ],     #j  No
                         [   0,  0,    0,  0,    0,  0,  ],     #k  No
                         #
                         [   0,  0,    0,  0,   .0, .9,  ],     #i  US
                         [   0,  0,    0,  0,    0,  0,  ],     #j  US
                         [  .1,  0,   .4,  0,   .5, .0,  ]])    #k  US

        self.Z_3r2i3p= np.array(
                     [[ 0. ,  0. ,  0.1,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ],
                      [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                      [ 0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                      [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                      [ 0.3,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ],
                      [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                      [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.9],
                      [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                      [ 0.1,  0. ,  0. ,  0. ,  0.4,  0. ,  0.5,  0. ,  0. ]])
                      
        # Suppy table V und use table U, both square, with byproducts, for testing the psc/btc construct              
        self.V_Test_byprod = np.array([[30, 1, 1, 1, 0],
                                       [1, 5, 1, 0, 0],
                                       [2, 0, 3, 1, 0],
                                       [0, 0, 1, 2, 0],
                                       [0, 0, 0, 0, 8]])
        
        self.U_Test_byprod = np.array([[12, 4, 5, 1, 0],
                                       [1, 1, 1, 0, 0],
                                       [0, 1, 1, 1, 0],
                                       [1, 0, 0, 0, 0],
                                       [0, 1, 1, 1, 0]])                      

    def test_V_bar_tilde(self):

        # Initialize sut
        V = np.array([[1.4, 0, 0,  12],
                      [5.,  3, 6., 0],
                      [0,   0, 0,  0.1],
                      [0,   0, 0,  0]])
        E_bar = np.array([[1, 0, 0, 0],
                          [0, 1, 1, 0],
                          [0, 0, 0, 1],
                          [0, 0, 0, 0]])
        sut = SupplyUseTable(V=V, E_bar=E_bar)

        # Test V_bar: primary supply flows only
        V0 = np.array([[1.4, 0, 0,  0],
                       [0.,  3, 6., 0],
                       [0,   0, 0,  0.1],
                       [0,   0, 0,  0]])
        npt.assert_array_equal(V0, sut.V_bar)

        # Test V_tild: secondary supply flows only
        V0 = np.array([[0,   0, 0,  12],
                       [5.,  0, 0,  0],
                       [0,   0, 0,  0],
                       [0,   0, 0,  0]])
        npt.assert_array_equal(V0, sut.V_tild)




    def test_V_bar_tilde_assumeDiag(self):

        # Initialize sut
        V = np.array([[1.4, 0, 0,  12],
                      [5.,  3, 6., 0],
                      [0,   0, 0,  0.1],
                      [0,   0, 0,  0]])
        sut = SupplyUseTable(V=V)

        # Test V_bar: primary supply flows only
        V0 = np.array([[1.4, 0, 0,  0],
                       [0.,  3, 0., 0],
                       [0,   0, 0,  0],
                       [0,   0, 0,  0]])
        npt.assert_array_equal(V0, sut.V_bar)

        # Test V_tild: secondary supply flows only
        V0 = np.array([[0.0, 0, 0,  12],
                       [5.,  0, 6., 0],
                       [0,   0, 0,  0.1],
                       [0,   0, 0,  0]])
        npt.assert_array_equal(V0, sut.V_tild)




    def test_build_E_bar_simpleExclusiveProd(self):

        V = np.array([[1.4, 0, 0,  12],
                      [5.,  3, 6., 0],
                      [0,   0, 0,  0.1],
                      [0,   0, 0,  0]])
        sut = SupplyUseTable(V=V)
        sut.build_E_bar()
        E_bar0 = np.array([[1, 0, 0, 0],
                           [0, 1, 1, 0],
                           [0, 0, 0, 1],
                           [0, 0, 0, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)

        sut.build_E_bar(prefer_exclusive=False)
        E_bar0 = np.array([[1, 0, 0, 1],
                           [0, 1, 1, 0],
                           [0, 0, 0, 0],
                           [0, 0, 0, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)




    def test_build_E_bar_notsquare(self):

        V = np.array([[1.4, 0, 0,  12,  0],
                      [5.,  3, 6., 0,   0],
                      [0,   0, 0,  0.1, 0],
                      [0,   0, 0,  0,   0]])
        sut = SupplyUseTable(V=V)

        sut.build_E_bar()
        E_bar0 = np.array([[0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0],
                           [0, 0, 0, 1, 0],
                           [0, 0, 0, 0, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)

        sut.build_E_bar(prefer_exclusive=False)
        E_bar0 = np.array([[0, 0, 0, 1, 0],
                           [1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)

    def test_build_E_bar_multipleExclusive(self):
        """ The point of this test is that the function should select V[5,4]=.2
         as the primary product instead of V[0,4]=0.2 by default, but the other
         way round when prefer_exclusive=false
        """

        V = np.array([[1.4, 0, 0,  12,  0],
                      [5.,  3, 6., 0,   0],
                      [0,   0, 0,  0.1, 0],
                      [0,   0, 0,  0,   0],
                      [0,   0, 0,  0.2, 0]])
        sut = SupplyUseTable(V=V)

        # First test: default
        sut.build_E_bar()
        E_bar0 = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)

        # Second test: prefer exclusive
        sut.build_E_bar(prefer_exclusive=False)
        E_bar0 = np.array([[1, 0, 0, 1, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)


    def test_build_E_bar_dontpreferdiag(self):
        """ The point of this test is that the function should select V[5,4]=.2
         as the primary product instead of V[0,4]=0.2 by default, but the other
         way round when prefer_exclusive=false
        """

        V = np.array([[1.4, 0, 0,  12,  0],
                      [5.,  3, 6., 0,   0],
                      [0,   0, 0,  0.1, 0],
                      [0,   0, 0,  0,   0],
                      [0,   0, 0,  0.2, 0]])
        sut = SupplyUseTable(V=V)

        # First test: always pick the biggest
        sut.build_E_bar(prefer_diag=False, prefer_exclusive=False)
        E_bar0 = np.array([[0, 0, 0, 1, 0],
                           [1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)

        # Second test: pick the largest, but prefer exclusive products
        sut.build_E_bar(prefer_diag=False)
        E_bar0 = np.array([[0, 0, 0, 0, 0],
                           [1, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)


    def test_build_E_bar_negatives(self):

        # native value, should not alter results
        V = np.array([[1.4, 0, 0,  12,   0],
                      [5.,  3, 6., 0,    0],
                      [0,   0, 0,  0.1,  0],
                      [0,   0, 0,  0,    0],
                      [0,   0, 0,  -0.2, 0]])
        sut = SupplyUseTable(V=V)

        sut.build_E_bar()
        E_bar0 = np.array([[1, 0, 0, 0, 0],
                           [0, 1, 1, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 0, 0],
                           [0, 0, 0, 1, 0]])
        npt.assert_array_equal(E_bar0, sut.E_bar)

    def test_build_E_bar_3reg2ind3prod(self):

        # native value, should not alter results
        sut = SupplyUseTable(V=self.V_3r2i3p)

        sut.build_E_bar()
        npt.assert_array_equal(self.E_bar_3r2i3p, sut.E_bar)

        sut = SupplyUseTable(V=self.V_3r2i3p_coprod)
        sut.build_E_bar()
        npt.assert_array_equal(self.E_bar_3r2i3p, sut.E_bar)

    def test_generate_Xi_square3regions(self):

                        #i  j    i  j       i  j
        Xi0 = np.array([[1, 0,   0, 0,      0, 0],       #i
                        [0, 1,   0, 3/5,    0, 0],       #j
                                                         #
                        [0, 0,   1, 0,      0, 0],       #i
                        [0, 0,   0, 0,      0, 0],       #j
                                                         #
                        [0, 0,   0, 0,      1, 0],       #i
                        [0, 0,   0, 2/5,    0, 1]])      #j

        sut = SupplyUseTable(V=self.V_3r2i2p, E_bar=self.E_bar_3r2i2p, regions=3)
        sut.build_mr_Xi()


        npt.assert_allclose(Xi0, sut.Xi)

    def test_generate_Xi_3reg2ind3prod(self):


        # No production of j in Canada
        # No product of i or k in Norway
        # No product of j in US

        # j primarily produced only in Norway, any 2nd prod of j diplaces j_No
        # No prod i in Norway displaces world mix (1:2 Ca:USA)
        # No prod k in Norway, displaces world mix (1:3 Ca:USA)


        Xi0 = np.array([
                         #   Ca   Ca  Ca No  No No   US  US US
                         #   i   j  k    i      j  k       i   j  k
                         [   1,  0, 0,   1./3,  0, 0,      0,  0, 0 ],     #i  Ca
                         [   0,  0, 0,   0,     0, 0,      0,  0, 0 ],     #j  Ca
                         [   0,  0, 1,   0,     0, 1./4,   0,  0, 0 ],     #k  Ca
                         #
                         [   0,  0, 0,   0,     0, 0,      0,  0, 0 ],     #i  No
                         [   0,  1, 0,   0,     1, 0,      0,  1, 0 ],     #j  No
                         [   0,  0, 0,   0,     0, 0,      0,  0, 0 ],     #k  No
                         #
                         [   0,  0, 0,   2./3,  0, 0,      1,  0, 0 ],     #i  US
                         [   0,  0, 0,   0,     0, 0,      0,  0, 0 ],     #j  US
                         [   0,  0, 0,   0,     0, 3./4,   0,  0, 1 ]])    #k  US

        sut = SupplyUseTable(V=self.V_3r2i3p, E_bar=self.E_bar_3r2i3p, regions=3)
        sut.build_mr_Xi()

        npt.assert_allclose(Xi0, sut.Xi)

    def test_generate_Gamma_square3regions(self):

        Gamma0 = np.array(
                        #i  j    i    j      i  j
                       [[1, 0,   0,   0,     0, 0],     # I
                        [0, 1,   0,   3/5,   0, 0],     # J
                                                        #
                        [0, 0,   0.5, 0,     0, 0],     # I
                        [0, 0,   0.5, 0,     0, 0],     # J
                                                        #
                        [0, 0,   0,   0,     1, 0],     # I
                        [0, 0,   0,   2/5,   0, 1]])    # J

        sut = SupplyUseTable(V=self.V_3r2i2p, E_bar=self.E_bar_3r2i2p, regions=3)
        sut.build_mr_Gamma()

        npt.assert_allclose(Gamma0, sut.Gamma)

    def test_generate_Gamma_3reg2ind3prod(self):


        # Sole producer of j is I_No (not J_No), pick that one
        # i_No assume tech of I_Ca and I_US (1:2)
        # k_No assume tech of J_Ca and J_US (1:3)

        Gamma0 = np.array([
                         #   Ca   Ca  Ca No  No No      US  US US
                         #   i  j  k    i     j  k       i  j  k
                         [   1, 0, 0,   1./3, 0, 0,      0, 0, 0 ],    #I  Ca
                         [   0, 0, 1,   0,    0, 1./4,   0, 0, 0 ],    #J  Ca
                         #
                         [   0, 1, 0,   0,    1, 0,      0, 1, 0 ],    #I  No
                         [   0, 0, 0,   0,    0, 0,      0, 0, 0 ],    #J  No
                         #
                         [   0, 0, 0,   2./3, 0, 0,      1, 0, 0 ],    #I  US
                         [   0, 0, 0,   0,    0, 3./4,   0, 0, 1 ]])   #J  US

        sut = SupplyUseTable(V=self.V_3r2i3p, E_bar=self.E_bar_3r2i3p, regions=3)
        sut.build_mr_Gamma()
        npt.assert_allclose(Gamma0, sut.Gamma)



    def test_psc_agg(self):
        """ Tests Product Substition Construct on SuUT"""


        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.85      ,  0.6875    ,  0.        ]])

        S0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, E_bar=self.E_bar, Xi=self.Xi, F=self.F)
        A, __, __, S, __, __, Z, F_con = sut.psc_agg(keep_size=False)

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(np.empty(0), Z, atol=self.atol)
        npt.assert_allclose(np.empty(0), F_con, atol=self.atol)
        
    def test_psc_agg_byprod(self):
        """ Tests whether by-products are returned correctly for the Product Substition Construct on SuUT"""

        AmRef = np.array([[0.4,	0.8,	1.666666667,	0.5,	0],
                          [0.033333333,	0.2,	0.333333333,	0,	0],
                          [0,	0.2,	0.333333333,	0.5,	0],
                          [0.033333333,	0,	0,	0,	0],
                          [0,	0.2,	0.333333333,	0.5,	0]])
 
        AbRef = np.array([[0,	0.2,	0.333333333,	0.5,	0],
                          [0.033333333,	0,	0.333333333,	0,	0],
                          [0.066666667,	0,	0,	0.5,	0],
                          [0,	0,	0.333333333,	0,	0],
                          [0,	0,	0,	0,	0]])
        

        sut = SupplyUseTable(U=self.U_Test_byprod, V=self.V_Test_byprod)
        sut.build_E_bar() # is unit matrix with the given example
        sut.build_mr_Xi() # is unit matrix with the given example
        A, Am, Ab, __, __, __, Z, F_con = sut.psc_agg(keep_size=False)

        npt.assert_allclose(A, AmRef-AbRef, atol=self.atol)
        npt.assert_allclose(Am, AmRef, atol=self.atol)
        npt.assert_allclose(Ab, AbRef, atol=self.atol)

    def test_partition_coefficients(self):
        """ Test calculation of PA coeff. (PHI) from intensive properties (PSI)
        """

        PHI0 = np.array([[0.5,  0.5,  0. ],
                         [0. ,  1. ,  0. ],
                         [0. ,  1. ,  0. ],
                         [0. ,  0. ,  1. ]])
        sut = SupplyUseTable(PSI=self.PSI, V=self.V)
        sut._SupplyUseTable__pa_coeff()
        npt.assert_allclose(PHI0, sut.PHI)

    def test_pc_agg(self):
        """ Tests partition aggregation construct on SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.75],
                       [2.  ,  4.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.        ,  0.95      ,  0.        ]])

        F_con0 = np.array([[5.,  24.,  18.],
                           [0.,   1.,   0.]])

        S0 = np.array([[2.5       ,  4.8       ,  1.63636364],
                       [0.        ,  0.2       ,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, PSI=self.PSI, F=self.F)
        A, S, __, __, Z, F_con = sut.pc_agg(keep_size=False)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)


    def test_pc_agg_noflows(self):
        """ Tests partition aggregation construct on SuUT"""


        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.        ,  0.95      ,  0.        ]])


        S0 = np.array([[2.5       ,  4.8       ,  1.63636364],
                       [0.        ,  0.2       ,  0.        ]])

        nothing = np.empty(0)

        sut = SupplyUseTable(U=self.Uu, V=self.V, PSI=self.PSI, F=self.F)
        A, S, __, __, Z, F_con = sut.pc_agg(keep_size=False, return_flows=False)

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)


    def test_aac_agg(self):
        """ Tests Alternate Activity Construct on SuUT"""

        Z0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.75      ],
                       [3.33333333,  3.41666667,  0.        ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.66666667,  0.68333333,  0.        ]])

        S0 = np.array([[2.5       ,  4.8       ,  1.63636364],
                       [-0.16666667,  0.26666667,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, E_bar=self.E_bar, Gamma=self.Gamma,
                  F=self.F)
        A, S, __, __, Z, __ = sut.aac_agg(keep_size=False)

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(Z0, Z, atol=self.atol)


    def test_aac_agg_3reg2ind3prod_nocoprod(self):
        """ The point of this one is to test SuUT where some products are
        simply not produced, neither as primary nor as secondary flows"""

        sut = SupplyUseTable(U=self.U_3r2i3p,
                  V=self.V_3r2i3p,
                  E_bar=self.E_bar_3r2i3p,
                  regions=3)

        sut.build_mr_Gamma()
        A, S, nn_in, nn_out, Z, F_con = sut.aac_agg()
        npt.assert_allclose(self.Z_3r2i3p, Z, atol=self.atol)

    def test_aac_agg_3reg2ind3prod_coprod(self):
        """ The point of this one is to test SuUT where some products are
        simply not produced, neither as primary nor as secondary flows"""

        sut = SupplyUseTable(U=self.U_3r2i3p,
                  V=self.V_3r2i3p_coprod,
                  regions=3)
        sut.build_E_bar()
        sut.build_mr_Gamma()

        A, S, nn_in, nn_out, Z, F_con = sut.aac_agg()

        #  Ca_j has same tech as No_I
        #  US_k has the tech of US_J, which was deduced from (primary) US_i
        Z0 = np.array([[-1./6, 1./6, 0.1, 0., 0.5, 0.,  0. , 0., 0.],
                       [ 0.   , 0.   , 0. , 0., 0. , 0.,  0. , 0., 0.],
                       [ 0.2  , 0.   , 0. , 0., 0. , 0.,  0. , 0., 0.],
                       [ 0.   , 0.   , 0. , 0., 0. , 0.,  0. , 0., 0.],
                       [ 0.3  , 0.   , 0. , 0., 0. , 0.,  0.5, 0., 0.],
                       [ 0.   , 0.   , 0. , 0., 0. , 0.,  0. , 0., 0.],
                       [ 0.   , 0.   , 0. , 0., 0. , 0., -0.1, 0., 1.],
                       [ 0.   , 0.   , 0. , 0., 0. , 0.,  0. , 0., 0.],
                       [-1./30, 2./15, 0. , 0., 0.4, 0.,  0.5, 0., 0.]]
                     )

        npt.assert_allclose(Z0, Z, atol=self.atol)

    def test_psc_agg_3reg2ind3prod_coprod(self):
        """ The point of this one is to test SuUT where some products are
        simply not produced, neither as primary nor as secondary flows"""

        sut = SupplyUseTable(U=self.U_3r2i3p, V=self.V_3r2i3p_coprod, regions=3)
        sut.build_E_bar()
        sut.build_mr_Xi()

        A, __, __, S, nn_in, nn_out, Z, F = sut.psc_agg(return_flows=True)
        # Ca_j production (secondary to Ca_i) displaces No_j
        # Us_k production (secondary to US_i) displaces US_k
        Z0 = np.array([[ 0. ,  0. ,  0.1,  0. ,  0.5,  0. ,  0. ,  0. ,  0. ],
                       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                       [ 0.2,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                       [-0.7,  0. ,  0. ,  0. ,  0. ,  0. ,  0.5,  0. ,  0. ],
                       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0.9],
                       [ 0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ,  0. ],
                       [ 0.1,  0. ,  0. ,  0. ,  0.4,  0. , -0.5,  0. ,  0. ]])

        npt.assert_allclose(Z0, Z, atol=self.atol)

    def test_psc_agg_3reg2ind3prod_nocoprod(self):
        """ The point of this one is to test SuUT where some products are
        simply not produced, neither as primary nor as secondary flows"""

        sut = SupplyUseTable(U=self.U_3r2i3p,
                  V=self.V_3r2i3p,
                  E_bar=self.E_bar_3r2i3p,
                  regions=3)

        sut.build_mr_Xi()
        A, __, __, S, nn_in, nn_out, Z, F = sut.psc_agg(return_flows=True)
        npt.assert_allclose(self.Z_3r2i3p, Z, atol=self.atol)



    def test_lsc(self):
        """ Tests Lump Sum Construct on SuUT"""

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.33333333,  0.6875    ,  0.        ]])

        S0 = np.array([[3.33333333,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, E_bar=self.E_bar, F=self.F)
        A, S, __, __, Z, F_con = sut.lsc(keep_size=False)

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(Z, np.empty(0))
        npt.assert_allclose(F_con, np.empty(0))

    def test_lsc_with_absolue_flows(self):
        """ Tests Lump Sum Construct on SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [0.  ,  0.  ,  0.75],
                       [4.  ,  2.75,  0.  ]])

        F_con0 = np.array([[10.,  19.,  18.],
                           [0.,   1.,   0.]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, E_bar=self.E_bar, F=self.F)
        __, __ , __,__, Z, F_con = sut.lsc(keep_size=False,
                                           return_flows=True)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)

    def test_itc(self):
        """ Tests Industry Technology Construct on SuUT"""

        Z0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.75      ],
                       [2.66666667,  4.08333333,  0.        ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.33333333,  0.81666667,  0.        ]])


        F_con0 = np.array([[6.66666667,  22.33333333,  18.        ],
                           [0.        ,   1.        ,   0.        ]])



        S0 = np.array([[3.33333333,  4.46666667,  1.63636364],
                       [0.        ,  0.2       ,  0.        ]])


        sut = SupplyUseTable(U=self.Uu, V=self.V, F=self.F)
        A, S, nn_in, nn_out, Z, F_con = sut.itc(keep_size=False)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)

    def test_esc_nonsquare(self):
        """ Test European System Construct on non-square system """

        Z0 = np.array([[ 0.  ,  0.  ,  0.  ],
                       [ 0.  ,  0.  ,  0.75],
                       [ 4.  ,  2.75,  0.  ]])

        A0 = np.array([[ 0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.06818182],
                       [ 2.        ,  0.55      ,  0.        ]])

        F_con0 = np.array([[ 10.,  19.,  18.],
                           [  0.,   1.,   0.]])

        S0 = np.array([[ 5.        ,  3.8       ,  1.63636364],
                       [ 0.        ,  0.2       ,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, F=self.F, E_bar = self.E_bar)
        A, S, nn_in, nn_out, Z, F_con = sut.esc()

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)

    def test_esc_nonsquare_noflow(self):
        """ Test European System Construct on non-square system """


        A0 = np.array([[ 0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.06818182],
                       [ 2.        ,  0.55      ,  0.        ]])

        S0 = np.array([[ 5.        ,  3.8       ,  1.63636364],
                       [ 0.        ,  0.2       ,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, F=self.F, E_bar = self.E_bar)
        A, S, nn_in, nn_out, Z, F_con = sut.esc(return_flows=False)

        nothing = np.empty(0)

        npt.assert_allclose(nothing, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(nothing, F_con, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)

    def test_esc_square(self):
        """ Test European System Construct on square system, assume primary
        production on diagonal."""

        Z0 = np.array([[ 0.  ,  0.  ,  0.  ],
                       [ 0.  ,  0.  ,  0.75],
                       [ 4.  ,  2.75,  0.  ]])

        A0 = np.array([[ 0.        ,  0.        ,  0.        ],
                       [ 0.        ,  0.        ,  0.06818182],
                       [ 2.        ,  0.55      ,  0.        ]])

        F_con0 = np.array([[ 10.,  19.,  18.],
                       [  0.,   1.,   0.]])

        S0 = np.array([[ 5.        ,  3.8       ,  1.63636364],
                       [ 0.        ,  0.2       ,  0.        ]])

        sut = SupplyUseTable(U=self.Ua, V=self.Va, F=self.Fa)
        A, S, nn_in, nn_out, Z, F_con = sut.esc()

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)

    def test_btc_nonsquare(self):
        """ Tests Byproduct Technology Construct on non-square SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [-1.  ,  0.  ,  0.75],
                       [4.  ,  2.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [-0.5      ,  0.        ,  0.06818182],
                       [2.        ,  0.6875    ,  0.        ]])

        F_con0 = np.array([[10.,  19.,  18.],
                           [0.,   1.,   0.]])

        S0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Uu, V=self.V, F=self.F, E_bar=self.E_bar)
        A, S, nn_in, nn_out, Z, F_con = sut.btc(keep_size=False)

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)

    def test_btc_square(self):
        """Tests Byproduct Technology Construct on square SuUT"""

        Z0 = np.array([[0.  ,  0.  ,  0.  ],
                       [-1.  ,  0.  ,  0.75],
                       [4.  ,  2.75,  0.  ]])

        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [-0.5       ,  0.        ,  0.06818182],
                       [2.        ,  0.6875    ,  0.        ]])

        F_con0 = np.array([[10.,  19.,  18.],
                           [0.,   1.,   0.]])

        S0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Ua, V=self.Va, F=self.Fa)
        A, S, nn_in, nn_out, Z, F_con = sut.btc(keep_size=False)

        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)


    def test_btc_square_noflow(self):
        """Tests Byproduct Technology Construct on square SuUT"""


        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [-0.5       ,  0.        ,  0.06818182],
                       [2.        ,  0.6875    ,  0.        ]])

        S0 = np.array([[5.        ,  4.75      ,  1.63636364],
                       [0.        ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Ua, V=self.Va, F=self.Fa)
        A, S, nn_in, nn_out, Z, F_con = sut.btc(keep_size=False,
                                                return_flows=False)
        nothing = np.empty(0)
        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(nothing, Z, atol=self.atol)
        npt.assert_allclose(nothing, F_con, atol=self.atol)


    def test_ctc(self):
        """ Tests Commodity Technology Construct on square SuUT"""

        Z0 = np.array([[0.    ,  0.    ,  0.    ],
                       [0.    ,  0.    ,  0.75  ],
                       [3.3125,  3.4375,  0.    ]])


        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.65625   ,  0.6875    ,  0.        ]])


        F_con0 = np.array([[5.25,  23.75,  18.  ],
                           [-0.25,   1.25,   0.  ]])


        S0 = np.array([[2.625     ,  4.75      ,  1.63636364],
                       [-0.125     ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Ua, V=self.Va, F=self.Fa)
        A, S, nn_in, nn_out, Z, F_con = sut.ctc()

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(Z0, Z, atol=self.atol)
        npt.assert_allclose(F_con0, F_con, atol=self.atol)

    def test_ctc_noflow(self):
        """ Tests Commodity Technology Construct on square SuUT"""

        Z0 = np.array([[0.    ,  0.    ,  0.    ],
                       [0.    ,  0.    ,  0.75  ],
                       [3.3125,  3.4375,  0.    ]])


        A0 = np.array([[0.        ,  0.        ,  0.        ],
                       [0.        ,  0.        ,  0.06818182],
                       [1.65625   ,  0.6875    ,  0.        ]])


        F_con0 = np.array([[5.25,  23.75,  18.  ],
                           [-0.25,   1.25,   0.  ]])


        S0 = np.array([[2.625     ,  4.75      ,  1.63636364],
                       [-0.125     ,  0.25      ,  0.        ]])

        sut = SupplyUseTable(U=self.Ua, V=self.Va, F=self.Fa)
        A, S, nn_in, nn_out, Z, F_con = sut.ctc(return_flows=False)

        npt.assert_allclose(A0, A, atol=self.atol)
        npt.assert_allclose(S0, S, atol=self.atol)
        npt.assert_allclose(np.empty(0), Z, atol=self.atol)
        npt.assert_allclose(np.empty(0), F_con, atol=self.atol)

#    if __name__ == '__main__':
#        unittest.main()

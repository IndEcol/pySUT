# -*- coding: utf-8 -*-
"""
Functions and classes for efficient handling of supply and use tables (SUTs)

Created on Mon Jun 30 17:21:28 2014

@author: stefan pauliuk, NTNU Trondheim, Norway

standard abbreviation: SUT

dependencies:
    numpy
    scipy

How to load this class:
- add class folder to system path
sys.path.append(Package_Path)

import pysut
from pysut import SupplyUseTable

"""

import sys
import logging
import numpy as np

# check for correct version number
if sys.version_info.major < 3:
    logging.warn('This package requires Python 3.0 or higher.')


class SupplyUseTable(object):

    """ Class containing a complete supply and use table

    Attributes
    ----------
    V : product by industry supply table
    U : product by industry use table
    Y : final demand product by end-use category
    F : Extensions: type by industry
    FY: Extensions: type by end-use category
    TL: Trade link (for MRIO models)


    unit : Unit for each row of V,U, and y (product unit)
    version : string
        This can be used as a version tracking system.
    year : string or int
        Baseyear of the IOSystem
    name : string, optional
        Name of the SUT, default is 'SUT'


    """

    def __init__(self, V=None, U=None, Y=None, F=None, FY=None, TL=None,
                 unit=None, version=None, year=None, name='SUT'):
        """Basic initialisation and dimension check methods"""
        self.V = V  # mandatory
        self.U = U  # mandatory
        self.Y = Y  # optional
        self.F = F  # optional
        self.FY = FY  # optional
        self.TL = TL  # optional

        self.name = name  # optional
        self.unit = unit  # optional
        self.year = year  # optional
        self.version = version  # optional

    def return_version_info(self):
        return str('Class SUT. Version 0.1. Last change: September 24th, 2014.')

    def dimension_check(self):
        """ This method checks which variables are present and checks whether data types and dimensions match
        """
        # Compile a little report on the presence and dimensions of the elements in the SUT
        DimReport = str('<br><b> Checking dimensions of SUT structure</b><br>')
        if self.V is not None:
            DimReport += str('Supply table is present with ' + str(len(self.V)) +
                             ' rows (products) and ' + str(len(self.V[0])) + ' columns (industries).<br>')
        else:
            DimReport += str('Supply table is not present.<br>')
        if self.U is not None:
            DimReport += str('Use table is present with ' + str(len(self.U)) +
                             ' rows (products) and ' + str(len(self.U[0])) + ' columns (industries).<br>')
        else:
            DimReport += str('Use table is not present.<br>')
        if self.Y is not None:
            if len(self.Y.shape) == 1:  # if Y is a true vector
                DimReport += str('Final demand is present with ' + str(len(self.Y)) +
                                 ' rows (products) and 1 column (FD categories).<br>')
            else:
                DimReport += str('Final demand is present with ' + str(len(self.Y)) +
                                 ' rows (products) and ' + str(len(self.Y[0])) + ' columns (FD categories).<br>')
        else:
            DimReport += str('Final demand is not present.<br>')
        if self.F is not None:
            DimReport += str('Industry extensions are present with ' + str(len(self.F)) +
                             ' rows (stressors) and ' + str(len(self.F[0])) + ' columns (industries).<br>')
        else:
            DimReport += str('Industry extensions are not present.<br>')
        if self.FY is not None:
            DimReport += str('FD extensions are present with ' + str(len(self.FY)) +
                             ' rows (stressors) and ' + str(len(self.FY[0])) + ' columns (FD categories).<br>')
        else:
            DimReport += str('FD extensions are not present.<br>')
        if self.TL is not None:
            DimReport += str('Trade link is present with ' + str(len(self.TL)) +
                             ' rows (products) and ' + str(len(self.TL[0])) + ' columns (regions).<br>')
        else:
            DimReport += str('Trade link is not present.<br>')

        # for most operations, especially the constructs, U and V are required to
        # be present and have correct dimensions. We check for this:
        if self.U is not None:
            if self.V is not None:
                if len(self.V) == len(self.U):
                    if len(self.V[0]) == len(self.U[0]):
                        StatusFlag = 1  # V and U have proper dimensions
                    else:
                        StatusFlag = 0
                else:
                    StatusFlag = 0
            else:
                StatusFlag = 0
        else:
            StatusFlag = 0

        return DimReport, StatusFlag

    def compare_IndustrialUseAndSupply(self):
        """ This method computes total industrial supply and total industrial use, and compares the two
        ResultVector = U.e */ V.e
        """
        return self.U.sum(axis=1) / self.V.sum(axis=1)

    def supply_diag_check(self):
        """ to apply the BTC, we need to have a non-zero diagonal for each producing sector.
        Determine which sectors produce: """
        SupplySum_i = self.g_V()
        SupplySum_p = self.q_V()
        SupplyDiag = self.V.diagonal()
        SupplyDiag_Eval = np.zeros((self.V.shape[0], 7))
        for m in range(0, self.V.shape[0]):
            if SupplySum_p[m] != 0:
                if SupplySum_i[m] != 0:
                    if SupplyDiag[m] != 0:
                        SupplyDiag_Eval[m, 0] = 1  # Normal situation, OK
                    else:
                        SupplyDiag_Eval[m, 1] = 1  # No supply by apparent main producer, problem
                else:
                    # Product only produced by other sectors, this sector is empty
                    SupplyDiag_Eval[m, 2] = 1
            else:
                if SupplySum_i[m] != 0:
                    # product not produced, apparent main sector produces only other products
                    SupplyDiag_Eval[m, 3] = 1
                else:
                    SupplyDiag_Eval[m, 4] = 1  # Product not produced and main sector is empty
            SupplyDiag_Eval[m, 5] = SupplySum_p[m]
            SupplyDiag_Eval[m, 6] = SupplySum_i[m]

        return SupplyDiag_Eval

    """
    Basic computations, row sum, col sum, etc.
    """

    def g_V(self):
        """ Compute total industrial output g from supply table V."""
        return self.V.sum(axis=0)

    def q_V(self):
        """ Compute total product output g from supply table V."""
        return self.V.sum(axis=1)

    def return_diag_V(self):
        """ Returns the diagonal of the supply table in matrix form : V^              """
        if self.V.shape[0] != self.V.shape[1]:
            raise ValueError(
                'Error: Supply table is not square, there is no proper diagonal of that matrix.')
        else:
            Result_Array = np.zeros((self.V.shape[0], self.V.shape[0]))
            for m in range(0, self.V.shape[0]):
                Result_Array[m, m] = self.V[m, m]
            return Result_Array

    def return_offdiag_V(self):
        """   Returns the off-diagonal of the supply table in matrix form : V_offdiag              """
        if self.V.shape[0] != self.V.shape[1]:
            raise ValueError(
                'Error: Supply table is not square, there is no proper diagonal of that matrix.')
        else:
            Result_Array = self.V.copy()
            for m in range(0, self.V.shape[0]):
                Result_Array[m, m] = 0
            return Result_Array

    def market_balance(self):
        """ Returns the market balance of the SUT."""
        if self.Y is not None:
            if len(self.Y.shape) == 1:  # if Y is a true vector
                return self.V.sum(axis=1) - self.U.sum(axis=1) - self.Y
            else:  # if Y is an array
                return self.V.sum(axis=1) - self.U.sum(axis=1) - self.Y.sum(axis=1)
        else:
            raise ValueError(
                'Error: There is no final demand; the market balance cannot be computed.')

    """
    Aggregation, removal, and re-arrangement methods
    """

    def aggregate_rearrange_products(self, PA, PR):
        """ multiplies an aggregation matrix PA from the left to V, U, and Y, rearranges the rows in columns of V, U, and Y according to the sorting matrix PR
        Equations: X_aggregated = PA * X, where X = U, V, or Y (and also TL)
        X_new = PR * X_aggregated * PR', where X = U, V
        Y_new = PR * Y_aggregated (and also TL)
        """
        self.V = np.dot(PR, np.dot(np.dot(PA, self.V), PR.transpose()))
        self.U = np.dot(PR, np.dot(np.dot(PA, self.U), PR.transpose()))
        if self.Y is not None:
            self.Y = np.dot(PR, np.dot(PA, self.Y))
        if self.F is not None:
            self.F = np.dot(self.F, PR.transpose())
        # No changes apply to FY
        if self.TL is not None:
            self.TL = np.dot(PR, np.dot(PA, self.TL))

        return 'Products were aggregated. Products and industries were resorted successfully.'

    def aggregate_regions(self, AV):
        """ This method aggregates the supply and use table. The length of the vector AV sais how many regions there are in the model. The total number of products and industries must be a multiple of that number, else, an error is given.
        Then, the SUT is summed up according to the positions in AV. if AV[n] == x, then region n in the big SUT is aggregated into region x
        OBS: This method required the presence of U, V, Y, F, and FY. Only TL is optional.
        """
        # First, check whether the elements in AV are monotonically increasing, starting from 1:
        if (np.unique(AV) - np.arange(1, max(AV) + 1, 1)).sum() == 0:
            DR, StatusFlag = self.dimension_check()
            if StatusFlag == 1:  # Dimensions are OK, continue
                ProdsPerRegion = len(self.V) / len(AV)
                IndusPerRegion = len(self.V[0]) / len(AV)
                FDPerRegion = len(self.Y[0]) / len(AV)
                # if the number of products is a true multiple of the number of regions
                if int(ProdsPerRegion) == ProdsPerRegion:
                    # if the number of industries is a true multiple of the number of regions
                    if int(IndusPerRegion) == IndusPerRegion:
                        # if the number of final demand categories is a true multiple of the
                        # number of regions
                        if int(FDPerRegion) == FDPerRegion:
                            print('Everything has proper dimensions. Aggregating SUT.')
                            NewSupply = np.zeros(
                                (ProdsPerRegion * max(AV), IndusPerRegion * max(AV)))
                            NewUse = np.zeros((ProdsPerRegion * max(AV), IndusPerRegion * max(AV)))
                            NewF = np.zeros((len(self.F), IndusPerRegion * max(AV)))
                            NewY = np.zeros((ProdsPerRegion * max(AV), FDPerRegion * max(AV)))
                            NewFY = np.zeros((len(self.F), FDPerRegion * max(AV)))

                            SupplyIM = np.zeros((ProdsPerRegion * max(AV), len(self.V[0])))
                            UseIM = np.zeros((ProdsPerRegion * max(AV), len(self.U[0])))
                            YIM = np.zeros((ProdsPerRegion * max(AV), len(self.Y[0])))
                            for m in range(0, len(AV)):  # aggregate rows
                                SupplyIM[(AV[m] - 1) * ProdsPerRegion:(AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] = SupplyIM[(AV[m] - 1) * ProdsPerRegion:(
                                    AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] + self.V[m * ProdsPerRegion:m * ProdsPerRegion + ProdsPerRegion, :]
                                UseIM[(AV[m] - 1) * ProdsPerRegion:(AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] = UseIM[(AV[m] - 1) * ProdsPerRegion:(
                                    AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] + self.U[m * ProdsPerRegion:m * ProdsPerRegion + ProdsPerRegion, :]
                                YIM[(AV[m] - 1) * ProdsPerRegion:(AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] = YIM[(AV[m] - 1) * ProdsPerRegion:(
                                    AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] + self.Y[m * ProdsPerRegion:m * ProdsPerRegion + ProdsPerRegion, :]
                            for m in range(0, len(AV)):  # aggregate columns
                                NewSupply[:, (AV[m] - 1) * IndusPerRegion:(AV[m] - 1) * IndusPerRegion + IndusPerRegion] = NewSupply[:, (AV[m] - 1) * IndusPerRegion:(
                                    AV[m] - 1) * IndusPerRegion + IndusPerRegion] + SupplyIM[:, m * IndusPerRegion:m * IndusPerRegion + IndusPerRegion]
                                NewUse[:, (AV[m] - 1) * IndusPerRegion:(AV[m] - 1) * IndusPerRegion + IndusPerRegion] = NewUse[:, (AV[m] - 1) * IndusPerRegion:(
                                    AV[m] - 1) * IndusPerRegion + IndusPerRegion] + UseIM[:, m * IndusPerRegion:m * IndusPerRegion + IndusPerRegion]
                                NewY[:, (AV[m] - 1) * FDPerRegion:(AV[m] - 1) * FDPerRegion + FDPerRegion] = NewY[:, (AV[m] - 1) * FDPerRegion:(
                                    AV[m] - 1) * FDPerRegion + FDPerRegion] + YIM[:, m * FDPerRegion:m * FDPerRegion + FDPerRegion]
                                NewF[:, (AV[m] - 1) * IndusPerRegion:(AV[m] - 1) * IndusPerRegion + IndusPerRegion] = NewF[:, (AV[m] - 1) * IndusPerRegion:(
                                    AV[m] - 1) * IndusPerRegion + IndusPerRegion] + self.F[:, m * IndusPerRegion:m * IndusPerRegion + IndusPerRegion]
                                if self.FY is not None:  # if we have findal demand extensions
                                    NewFY[:, (AV[m] - 1) * FDPerRegion:(AV[m] - 1) * FDPerRegion + FDPerRegion] = NewFY[:, (AV[m] - 1) * FDPerRegion:(
                                        AV[m] - 1) * FDPerRegion + FDPerRegion] + self.FY[:, m * FDPerRegion:m * FDPerRegion + FDPerRegion]
                            # assign the new values to the object
                            self.V = NewSupply
                            self.U = NewUse
                            self.Y = NewY
                            self.F = NewF
                            self.FY = NewFY
                            if self.TL is not None:  # Special case: If a trade link is present:
                                NewTL = np.zeros((ProdsPerRegion * max(AV), max(AV)))
                                TL_IM = np.zeros((ProdsPerRegion * max(AV), len(AV)))
                                # First, aggregate the origin regions:
                                for m in range(0, len(AV)):
                                    TL_IM[(AV[m] - 1) * ProdsPerRegion:(AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] = TL_IM[(AV[m] - 1) * ProdsPerRegion:(
                                        AV[m] - 1) * ProdsPerRegion + ProdsPerRegion, :] + self.TL[m * ProdsPerRegion:m * ProdsPerRegion + ProdsPerRegion, :]
                                # Second, aggregate the destination regions:
                                for m in range(0, len(AV)):
                                    NewTL[
                                        :, int((AV[m] - 1))] = NewTL[:, int((AV[m] - 1))] + TL_IM[:, m]
                                self.TL = NewTL
                            ExitFlag = 1
                            ExitComment = 'Aggregation of regions went allright.'
                        else:
                            ExitFlag = 4
                            ExitComment = 'Total number of final demand categories is not a true multiple of the number of regions.'
                    else:
                        ExitFlag = 2
                        ExitComment = 'Total number of industries is not a true multiple of the number of regions.'
                else:
                    ExitFlag = 3
                    ExitComment = 'Total number of products is not a true multiple of the number of regions.'
            else:
                ExitFlag = 5
                ExitComment = 'Problem with the dimensions of the SUT.'
        else:
            ExitFlag = 0
            ExitComment = 'Problem with the sorting vector. It needs to contain all natural numbers from 1,2,3,... to its maximum value.'
        return ExitFlag, ExitComment

    def remove_products_industries(self, RPV, RIV):
        """ This method sets the products with the indices in the remove-product-vector RPV to zero.
        Likewise for the industries in the remove-industy-vector RIV
        """
        # First: remove products from U, V, and Y:
        for x in RPV:
            self.U[x, :] = 0
            self.V[x, :] = 0
            if self.Y is not None:
                self.Y[x, :] = 0
            # No changes to FY
            if self.TL is not None:
                # This might be problematic, since some methods require T to have at least
                # a 1 on the diagonal
                self.TL[x, :] = 0
        # Second: remove industries from U, V, and F:
        for x in RIV:
            self.U[:, x] = 0
            self.V[:, x] = 0
            self.F[:, x] = 0

        return 'Products and industries were removed successfully.'

    """
    Modify tables
    """

    def add_ones_to_diagonal(self):
        """ This method adds ones where there is a zero on the diagonal of V. This is needed for simple applications of the BTC."""
        if self.V.shape[0] != self.V.shape[1]:
            return 'Error: Supply table is not square, there is no proper diagonal of that matrix.'
        else:
            for m in range(0, self.V.shape[0]):
                if self.V[m, m] == 0:
                    self.V[m, m] = 1

    def clear_non_diag_supply(self):
        """ This method allows for simple application of the BTC. It removes all sectors that do not produce their respective main product."""
        if self.V.shape[0] != self.V.shape[1]:
            raise ValueError(
                'Error: Supply table is not square, there is no proper diagonal of that matrix.')
        else:
            for m in range(0, self.V.shape[0]):
                if self.V[m, m] == 0:
                    self.V[:, m] = 0
                    self.U[:, m] = 0

    """
    Constructs. Below, it is always assumed that U and V are present. For the industrial stressorts, F must be present as well.
    """

#    def Build_BTC_A_matrix_extended(self):
#        """ Builds the A-matrix of the extended BTC construct
#        return: Extended A matrix for BTC construct"""
#        self.A_BTC = np.concatenate((np.concatenate((np.dot(self.U,np.linalg.inv(self.return_diag_V())),-1 * np.eye(self.V.shape[0])),axis=1), np.concatenate((np.dot(self.return_offdiag_V(),np.linalg.inv(self.return_diag_V())),np.zeros((self.V.shape[0],self.V.shape[0]))),axis=1)),axis = 0)
#        return self.A_BTC
#
#    def Build_BTC_L_matrix_extended(self):
#        L_prov = np.linalg.inv(np.eye(2*self.V.shape[0])-self.Build_BTC_A_matrix_extended())
#        self.L_BTC = L_prov[:,0:self.V.shape[0]]
#        return self.L_BTC
    """ General: Determine L matrix"""

    def Build_L_matrix(self, A):
        return np.linalg.inv(np.eye(self.V.shape[0]) - A)

    """ byproduct technology construct (BTC)"""

    def Build_BTC_A_matrix(self, Xi=None):
        """ Builds the A-matrix of the normal BTC construct, using Xi as mapping matrix
        returns: A matrix for BTC construct
        A_BTC = (U - Xi * V_offdiag)V_diag_inv """
        if Xi == None:
            Xi = np.ones((self.V.shape))
        self.A_BTC = np.dot(
            (self.U - Xi * self.return_offdiag_V()), np.linalg.inv(self.return_diag_V()))
        return self.A_BTC

    def Build_BTC_Am_matrix(self):
        """ returns use part of BTC construct: Am = UV^-1. Used to re-construct the SUT from the BTC-IO model """
        return np.dot(self.U, np.linalg.inv(self.return_diag_V()))

    def Build_BTC_Ab_matrix(self):
        """ returns use part of BTC construct: Ab = VoffdiagV^-1. Used to re-construct the SUT from the BTC-IO model """
        return np.dot(self.return_offdiag_V(), np.linalg.inv(self.return_diag_V()))

    def Build_BTC_S(self):
        """Returns stressor coefficient matrix for the BTC construct."""
        self.S_BTC = np.dot(self.F, np.linalg.inv(self.return_diag_V()))
        return self.S_BTC

    """ Commodity technology construct (CTC)"""

    def Build_CTC_A_matrix_ixi(self):
        """ Builds the A-matrix of the CTC construct, industry-by-industry
        return: A matrix for CTC construct
        Equation taken from Miller and Blair (2009), chapter 5, Equation 5.26a
        A_CTC_ixi = g^ * V^-1 * U * g^-1"""
        if self.V.shape[0] != self.V.shape[1]:
            raise ValueError('Error: Supply table V is not square, no matrix inversion possible.')
        else:
            try:
                V_inv = np.linalg.inv(self.V)
                try:
                    g_inv = np.linalg.inv(np.diag(self.g_V()))
                    self.A_CTC_ixi = np.dot(
                        np.dot(np.diag(self.g_V()), V_inv), np.dot(self.U, g_inv))
                except:
                    raise ValueError(
                        'Error: Diagonal of total industry output g cannot be inverted. Singular matrix.')
            except:
                raise ValueError('Error: Supply table V is square, but no inverse exists.')
        return self.A_CTC_ixi

    def Build_CTC_A_matrix_cxc(self):
        """ Builds the A-matrix of the CTC construct, commodity-by-commodity
        return: A matrix for CTC construct
        Equation taken from Miller and Blair (2009), chapter 5, Equation 5.26
        A_CTC_cxc = U * V^-1"""
        if self.V.shape[0] != self.V.shape[1]:
            raise ValueError('Error: Supply table V is not square, no matrix inversion possible.')
        else:
            try:
                V_inv = np.linalg.inv(self.V)
                self.A_CTC_cxc = np.dot(self.U, V_inv)
            except:
                raise ValueError('Error: Supply table V is square, but no inverse exists.')
        return self.A_CTC_cxc

    def Build_CTC_cxc_S(self):
        """Returns stressor coefficient matrix for the CTC cxc construct. S = F V^-1"""
        self.S_CTC_cxc = np.dot(self.F, np.linalg.inv(self.V))
        return self.S_CTC_cxc

    """ Industry technology construct (ITC)"""

    def Build_ITC_A_matrix_ixi(self):
        """ Builds the A-matrix of the ITC construct, industry-by-industry
        return: A matrix for ITC construct
        Equation taken from Miller and Blair (2009), chapter 5, Equation 5.27a
        A_ITC_ixi = V'*q^-1  *  U * g^-1"""

        try:
            self.g_hat_inv = np.linalg.inv(np.diag(self.g_V()))
            self.q_hat_inv = np.linalg.inv(np.diag(self.q_V()))
            self.A_ITC_ixi = np.dot(
                np.dot(self.V.transpose(), self.q_hat_inv), np.dot(self.U, self.g_hat_inv))
        except:
            raise ValueError('Error: Singular matrix.')

        return self.A_ITC_ixi

    def Build_ITC_A_matrix_cxc(self):
        """ Builds the A-matrix of the ITC construct, commodity-by-commodity
        return: A matrix for ITC construct
        Equation taken from Miller and Blair (2009), chapter 5, Equation 5.27
        A_ITC_cxc = U * g^-1  *  V'*q^-1 """

        try:
            self.g_hat_inv = np.linalg.inv(np.diag(self.g_V()))
            self.q_hat_inv = np.linalg.inv(np.diag(self.q_V()))
            self.A_ITC_cxc = np.dot(
                np.dot(self.U, self.g_hat_inv), np.dot(self.V.transpose(), self.q_hat_inv))
        except:
            raise ValueError('Error: Singular matrix.')

        return self.A_ITC_cxc

    def Build_ITC_cxc_S(self):
        """Returns stressor coefficient matrix for the ITC cxc construct."""
        self.S_ITC_cxc = np.dot(np.dot(self.F, np.linalg.inv(np.diag(self.g_V()))), np.dot(
            self.V.transpose(), np.linalg.inv(np.diag(self.q_V()))))
        return self.S_ITC_cxc


"""
End of file
"""

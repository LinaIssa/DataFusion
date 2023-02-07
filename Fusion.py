from datetime import time

import numpy        as np
import scipy.sparse as sp
from typing import Union
from abc    import ABC, abstractmethod
from tools  import get_h_band, get_g_band, aliasing_adj, _centered
from Cube   import CubeHyperSpectral, CubeMultiSpectral


class Fusion(ABC):
    '''
    @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

    Fusion is an abstract class gathering all the procedures that are regardless to the regularisation. This includes
    the vectorial-based method of computing  the matrix A, B and C that are attached to the data. The conjugate
    gradient and the regularisation part of the matrix A is then defined specifically for each class inheriting the
    class Fusion. One must also define the fusion procedure inside each subclass.
    :param cubeMultiSpectral: A CubeMultiSpectral object that has been instantiating.
    The call method takes care of preprocessing the  multi-spectral image.
    :param cubeHyperSpectral: A cubeHyperSpectral object that has been instantiating. The call
    method takes care of preprocessing the hyper-spectral image.
    :param Lm: The spectral degradation operator on the
    multi-spectral image. Should be the same as the one given to the call of cubeMultiSpectral. The path to the
    Lm.fits file is defined in the configuration file.
    :param nc: a parameter defined in the configuration file
    :param nr: a parameter defined in the configuration file
    :param mu: a parameter that controls the regularisation
    strength. By default, mu=10

    '''

    def __init__(self, cubeMultiSpectral: CubeMultiSpectral, cubeHyperSpectral: CubeHyperSpectral, Lm: np.array,
                 nc: int, nr: int, mu: int = 10, **kwargs) -> None:

        self.Y_multi = cubeMultiSpectral.Ync
        self.Y_hyper = cubeHyperSpectral.Yns

        if self.Y_multi is None:
            raise ValueError('Y_multi needs to be processed by calling the class CubeMultiSpectral')
        if self.Y_hyper is None:
            raise ValueError('Y_hyper needs to be processed by calling the class CubeHyperSpectral')

        sig2_ms, sig2_hs = cubeMultiSpectral.sig2, cubeHyperSpectral.sig2

        if sig2_ms is None:
            raise ValueError(
                'By instantiating the class CubeMultiSpectral, the observed noise variance in the multi-spectral '
                'image will be computed.')
        if sig2_hs is None:
            raise ValueError(
                'By instantiating the class CubeHyperSpectral, the observed noise variance in the hyperspectral image '
                'will be computed.')

        sig2 = [sig2_ms, sig2_hs]

        self.sig2 = sig2
        self.fact_pad = cubeMultiSpectral.fact_pad
        self.d = cubeHyperSpectral.d
        self.V = cubeHyperSpectral.V
        self.Z = cubeHyperSpectral.Z
        self.lacp = cubeHyperSpectral.lacp

        self.nc, self.nr, self.mu = nc, nr, mu
        self.lm, self.lh = Lm.shape[0], Lm.shape[1]

        self.Lm = Lm
        self.sbnc = 1 / (sig2[0] * nr * nc * self.lm)
        self.sbns = 1 / (sig2[1] * (nr // self.d) * (nc // self.d) * self.lh)

    @abstractmethod
    def spatial_regularisation(self, D: np.array, Wd: np.array, Z: np.array) -> np.array:
        '''
        @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

        Constructs the spatial regularisation for the conjugate gradient. Should be overriden in each subclass that inherits
        from Fusion.
        :param D : the computed operator of finite differences from the spatial regularisation method.
        :param Wd: the computed matrix of weights from the spatial regularisation method
        :param Z : the spectral-reduced hyperspectral cube from the PCA projection carried out during the preprocessing.
        Z is stored as attribute of Fusion class for simplifying its retrieval.
        :return  : Areg, the part of the matrix A that contains the regularisation information
        '''
        return

    @abstractmethod
    def conjugate_gradient(self, *args, **kwargs):
        """
        @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

        Performs the conjugate gradient algorithm, the keystone in the fusion framework. To be adapted in each
        subclass that inherits from Fusion. The spatial regularisation method is computed each time Z is updated.
        :param args  : depends on the regularisation type but should include at least: A, D, Wd, B, C and Z
        :param kwargs:
        :return: Z the fusion product and obj the objective function
        """
        return

    @abstractmethod
    def __call__(self, *args, **kwargs):
        """
        @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

        Put the fusion framework into action by first writing the inverse problem as a linear system with vectorized
        matrix,that is solved by a conjugate gradient in a Fast Fourier domain. The formulation of the linear system
        being regularisation independent, it is carried out by the initialisation which stores the matrix in class
         attributes. The call method performs the conjugate gradient accordingly to the chosen regularisation.
        :param args:
        :param kwargs:
        :return: Z the product of the fusion
        :return: obj the objective function
        """
        return

    @staticmethod
    def _M(i: int,j: int, phv: float, nr: int, nc: int):
        """
        @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

        is called in the Anc function
        :param i:
        :param j:
        :param phv:
        :param nr:
        :param nc:
        :return:
        """
        res = np.zeros(nr * nc, dtype=np.complex)
        nf = phv.shape[0]
        for m in range(nf):
            res += np.conj(phv[m, j]) * phv[m, i]
        return res

    @staticmethod
    def _C(i, j, V, row, col, nr, nc, d):
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        is called in the Ans function
        :param j:
        :param V:
        :param row:
        :param col:
        :param nr:
        :param nc:
        :param d:
        :return:
        """
        res = np.zeros(nr * nc * d ** 2, dtype=np.complex)
        lh = len(V)
        for m in range(lh):
            g = get_g_band(m)
            gntng = d ** (-2) * np.conj(g[row]) * g[col]
            res += (V[m, i] * gntng * V[m, j])
        return res

    @staticmethod
    def _nTn_sparse(nr: int, nc: int, d: int):
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        is called in the Ans function
        """
        n1 = sp.identity(nc // d)
        n2_ = n1.copy()
        for i in range(d - 1):
            n2_ = sp.hstack((n2_, n1))
        n2 = n2_.copy()
        for i in range(d - 1):
            n2 = sp.vstack((n2, n2_))
        n3 = n2.copy()
        for i in range(nr // d - 1):
            n3 = sp.block_diag((n3, n2))
        n4_ = n3.copy()
        for i in range(d - 1):
            n4_ = sp.hstack((n4_, n3))
        n4 = n4_.copy()
        for i in range(d - 1):
            n4 = sp.vstack((n4, n4_))
        return n4

    @staticmethod
    def _FiniteDifferenceOperator(nr: int, nc: int) -> tuple:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        is called in the preprocess_spatial_regularisation method.
        """
        size = (nr, nc)
        Dx = np.zeros(size)
        Dy = np.zeros(size)

        Dx[0, 0] = 1
        Dx[0, 1] = -1
        Dy[0, 0] = 1
        Dy[1, 0] = -1

        Dx = np.fft.fft2(Dx)
        Dy = np.fft.fft2(Dy)
        return Dx, Dy

    @staticmethod
    def _PHV(lacp: int, Lm: np.array, V, nr: int, nc: int) -> np.array:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        is called in the _Anc method

        """

        nf = Lm.shape[0]
        lh = V.shape[0]
        res = np.zeros((nf, lacp, nr * nc), dtype=np.complex)
        print(' *** PHV computation ***')
        for m in range(nf):
            for i in range(lacp):
                sum_h = np.zeros(nr * nc, dtype=np.complex)
                for l in range(lh):
                    sum_h += get_h_band(l) * Lm[m, l] * V[l, i]
                res[m, i] = sum_h
        return res

    def _Anc(self) -> np.array:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        Computes the part of the matrix A related to the multi-spectral image. Called in MatrixA_data method
        """
        nc   = self.nc
        nr   = self.nr
        lacp = self.lacp

        t1 = time()
        row = np.arange(nr * nc)
        row = np.matlib.repmat(row, 1, lacp ** 2)[0]
        for i in range(lacp):
            row[i * nr * nc * lacp:(i + 1) * nr * nc * lacp] += i * nr * nc
        col = np.arange(nr * nc * lacp)
        col = np.matlib.repmat(col, 1, lacp)[0]

        phv = self._PHV(lacp, self.Lm, self.V, nr, nc)
        mat = np.reshape(np.reshape(np.arange(lacp ** 2), (lacp, lacp)).T, lacp ** 2)
        data = np.zeros((lacp ** 2, nr * nc), dtype=np.complex)
        for i in range(lacp):
            # print('i='+str(i))
            for j in range(lacp - i):
                # print('j='+str(j+i))
                temp = self._M(i, j + i, phv, nr, nc)
                if j == 0:
                    data[i * (lacp + 1)] = temp
                else:
                    index = j + i * (lacp + 1)
                    data[mat[index]] = np.conj(temp)
                    data[index] = temp
        data = np.reshape(data, (lacp ** 2 * nr * nc))
        anc = sp.coo_matrix((data, (row, col)), shape=(lacp * nr * nc, lacp * nr * nc), dtype=np.complex)
        t2 = time()
        print('Anc computation time : ' + str(t2 - t1) + 's.')

        return anc

    def _Ans(self, Lh: np.array) -> np.array:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        Computes the part of the matrix A related to the hyper-spectral image. Called in MatrixA_data method
        """
        nc = self.nc
        nr = self.nr
        d = self.d
        lacp = self.lacp

        t1 = time()
        ntn = _nTn_sparse(nr, nc, d)
        V = np.dot(np.diag(Lh), self.V)
        row = np.matlib.repmat(ntn.row, 1, lacp ** 2)[0]
        for i in range(lacp):
            row[i * nr * nc * lacp * d ** 2:(i + 1) * nr * nc * lacp * d ** 2] += i * nr * nc
        col = np.zeros(nr * nc * d ** 2 * lacp)
        for i in range(lacp):
            col[i * nr * nc * d ** 2:(i + 1) * nr * nc * d ** 2] = ntn.col + i * nr * nc
        col = np.matlib.repmat(col, 1, lacp)[0]
        mat = np.reshape(np.reshape(np.arange(lacp ** 2), (lacp, lacp)).T, lacp ** 2)
        data = np.zeros((lacp ** 2, nr * nc * d ** 2), dtype=np.complex)
        for i in range(lacp):
            # print('i='+str(i))
            for j in range(lacp - i):
                # print('j='+str(j+i))
                temp = self._C(i, j + i, V, ntn.row, ntn.col, nr, nc, d)
                if j == 0:
                    data[i * (lacp + 1)] = temp
                else:
                    index = j + i * (lacp + 1)
                    data[mat[index]] = np.conj(temp)
                    data[index] = temp
        data = np.reshape(data, np.prod(data.shape))
        t2 = time()
        print('Ans computation time : ' + str(t2 - t1) + 's.')
        ans = sp.coo_matrix((data, (row, col)), shape=(lacp * nr * nc, lacp * nr * nc), dtype=np.complex)

        return ans


    def preprocess_spatial_regularisation(self) -> Union[tuple, tuple]:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        Computing weights is part of the preprocessing for the spatial regularisation, regardless to the type of
        regularisation.
        :return: D operator of finite differences
        :return: Wd the weights matrix
        """
        V  = self.V  # the singular values matrix from the PCA decompostion performed on hyperspectral matrix
        Ym = self.Y_multi  # the preprocessed multi-spectral image
        Lm = self.Lm  # the spectral degradation operator on multi-spectral image
        nr, nc = self.nr, self.nc
        #############################################################
        #              Computing the operator of finite difference  #
        #############################################################

        D = _FiniteDifferenceOperator(nr, nc)

        #############################################################
        #              Computing Ym D                               #
        #############################################################
        Ymdx = np.fft.ifft2(Ym.reshape(Ym.shape[0], nr, nc) * D[0], axes=(1, 2))
        Ymdy = np.fft.ifft2(Ym.reshape(Ym.shape[0], nr, nc) * D[1], axes=(1, 2))

        #############################################################
        #              Trace Ym                                     #
        #############################################################

        trymx = np.linalg.norm(Ymdx, ord=2, axis=0)
        trymy = np.linalg.norm(Ymdy, ord=2, axis=0)

        #############################################################
        #              Trace Ym                                     #
        #############################################################

        trlm = np.trace(np.dot(Lm, Lm.T))
        sigchap2x = trymx / trlm
        sigchap2y = trymy / trlm

        #############################################################
        #              Compute Sigma                                #
        #############################################################

        vtv = np.diag(np.dot(V.T, V))
        Sigmazx = np.zeros((vtv.shape[0], sigchap2x.shape[0], sigchap2x.shape[1]))
        Sigmazy = np.zeros((vtv.shape[0], sigchap2y.shape[0], sigchap2y.shape[1]))
        for i in range(vtv.shape[0]):
            Sigmazx[i] = sigchap2x * vtv[i]
            Sigmazy[i] = sigchap2y * vtv[i]

        epsilon = 1e-2
        Wd = (0.5 * (1 / (Sigmazx + epsilon) + 1 / (Sigmazy + epsilon)),
              0.5 * (1 / (Sigmazx + epsilon) + 1 / (Sigmazy + epsilon)))
        Wd = (Wd - np.min(Wd)) / np.max(Wd - np.min(Wd))
        return D, Wd

    def MatrixA_data(self, Lh: np.array) -> np.array:
        """
        @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

        Computes the data-driven matrix A, ie without taking into account the regularization. This matrix needs to be
        computed once for all regardless of the chosen regularisation.

        :param Lh: the spectral degradation operator applied to the hyperspectral image. Should be the same as the one
        given to the call of cubeMultiSpectral. The path to the Lm.fits file is defined in the configuration file.
        :return: The matrix A attached to the data
        """
        ################################################
        #                Multi-spectral Part        #
        ################################################

        Anc = self._Anc()
        A_multi = self.sbnc * Anc
        ################################################
        #               Hyperspectral Part        #
        ################################################
        Ans = self._Ans(Lh)
        A_hyper = self.sbns * Ans
        #######################################
        #               Total Matrix A        #
        #######################################

        A_data = A_multi + A_hyper

        return A_data

    def MatrixB_data(self, Lh: np.array) -> np.array:
        """
        :param Lh: the spectral degradation operator applied to the hyperspectral image.
        Should be the same as the one given to the call of cubeMultiSpectral. The path to the Lm.fits file is defined
        in the configuration file.
        :return: The matrix B
        """
        Ym, Yh = self.Y_multi, self.Y_hyper
        Lm = self.Lm
        nr, nc = self.nr, self.nc
        d = self.d
        Ym = np.reshape(Ym, np.prod(Ym.shape))  # to be consistent with the output of set inputs
        Yh = np.reshape(Yh, np.prod(Yh.shape))  # to be consistent with the output of set inputs

        Ym = np.reshape(Ym, (len(Ym) // (nr * nc), nr * nc))
        Yh = np.reshape(Yh, (len(Yh) // (nr * nc // d ** 2), nr * nc // d ** 2))

        lh, lm = Yh.shape[0], Ym.shape[0]

        ############################################
        #               Multi-spectral Part         #
        ############################################

        bnc = np.dot(Lm.T, Ym)
        for l in range(lh):
            bnc[l] = get_h_band(l, mode='adj') * bnc[l]
        bnc = np.dot(self.V.T, bnc)
        ###########################################
        #               Hyperspectral Part        #
        ###########################################
        Yh_ = aliasing_adj(Yh, (Yh.shape[0], nr, nc))
        for l in range(lh):
            Yh_[l] *= get_g_band(l, mode='adj')
        bns = np.dot(np.dot(np.diag(Lh), self.V).T, Yh_)
        bm = np.reshape(-self.sbnc * bnc, np.prod(bnc.shape))
        bh = np.reshape(-self.sbns * bns, np.prod(bns.shape))

        #######################################
        #               Total Matrix B        #
        #######################################

        B_data = bm + bh

        return B_data

    def MatrixC_data(self) -> np.array:

        Ym, Yh = self.Y_multi, self.Y_hyper
        cm = 0.5 * self.sbnc * np.dot(np.conj(Ym).T, Ym)
        ch = 0.5 * self.sbns * np.dot(np.conj(Yh).T, Yh)
        C = cm + ch

        return C

    def postprocess(self, Zfusion: np.array) -> np.array:  # a mettre dans le module Cube.py ?

        Zfusion = np.reshape(Zfusion, (self.lacp, self.nr, self.nc))
        Zfusion = np.fft.ifft2(Zfusion, norm='ortho')
        Zfusion = np.real(
            _centered(Zfusion[:, :-2, :-2], (self.lacp, self.nr - 2 * self.fact_pad, self.nc - 2 * self.fact_pad)))

        return Zfusion

class Weighted_Sobolev_Reg(Fusion):
    def __init__(self, cubeMultiSpectral: CubeMultiSpectral, cubeHyperSpectral: CubeHyperSpectral, Lm: np.array,
                 Lh: np.array,
                 nc: int, nr: int) -> None:
        super().__init__(cubeMultiSpectral, cubeHyperSpectral, Lm, nc, nr, mu=10)
        # Linear System

        self.A = self.MatrixA_data(Lh)
        self.B = self.MatrixB_data(Lh)
        self.C = self.MatrixC_data()

    @abstractmethod
    def spatial_regularisation(self, D: np.array, Wd: np.array, Z: np.array) -> np.array:
        """
        Constructs the regularisation part of the matrix A. Needs D and Wd from the preprocessing
        :param D :  as computed by the spatial regularisation preprocessing
        :param Wd:  as computed by the spatial regularisation preprocessing
        :param Z :  the reduced hyperspectral image computed by the PCA. The value of Z chang
        :return: Areg, the part of the matrix A that contains the regularisation information
        """
        Z = np.reshape(Z, np.prod(Z.shape))
        Z = np.reshape(Z, (self.lacp, Wd[0].shape[1], Wd[0].shape[2]))
        gx = np.fft.fft2(np.real(np.fft.ifft2(Z * D[0])) * Wd[0] ** 2, norm='ortho') * np.conj(D[0])
        gy = np.fft.fft2(np.real(np.fft.ifft2(Z * D[1])) * Wd[1] ** 2, norm='ortho') * np.conj(D[1])
        Areg = 2 * self.mu * np.reshape(gx + gy, np.prod(gx.shape))
        return Areg

    @abstractmethod
    def conjugate_gradient(self, A, D, Wd, B, C, Z, save_it=False) -> Union[np.array, np.array]:
        """
        :param A: the vectorized data-driven A from MatrixA_data class method.
        :param D: the operator of finite differences from the spatial regularisation preprocess method
        :param Wd: the weights matrix retrieved from the spatial regularisation preprocess method
        :param B: the vectorized matrix B from MatrixB_data class method
        :param C: the vectorized matrix C from MatrixB_data class method
        :param Z: the spectral-reduced hyperspectral image stored as a class attribute.
        :param save_it: boolean, by default False
        :return: Z and obj
        """
        print('--- CONJUGATE GRADIENT ALGORITHM ---')
        t1 = time()
        nb_it = 0
        # z0 = Z.copy()
        # Initialize

        ########## Control procedure ##########
        print('NANs in Az, b and A_reg z : ' + str(np.sum(np.isnan(A.dot(Z)))) + ' ; ' + str(
            np.sum(np.isnan(B))) + ' ; ' + str(np.sum(np.isnan(self.spatial_regularisation(D, Wd, Z)))))
        #######################################

        r = A.dot(Z) + B + self.spatial_regularisation(D, Wd, Z)
        p = -r.copy()
        # Objective function
        obj = [0.5 * np.dot(np.conj(Z).T, A.dot(Z)) + np.dot(np.conj(B.T), Z) + C + 0.5 * np.dot(np.conj(Z).T,
                                                                                                 self.spatial_regularisation(
                                                                                                     D, Wd, Z))]
        print(str(nb_it) + ' -- Objective function value : ' + str(obj[nb_it]))
        ########## Control procedure ##########
        if np.isnan(obj):
            print('Objective value is NAN :')
            print('NANs in 0,5* zt A z : ' + str(np.sum(np.isnan(0.5 * np.dot(np.conj(Z).T, A.dot(Z))))))
            print('NANs in bt z : ' + str(np.sum(np.isnan(np.dot(np.conj(B.T), Z)))))
            print('NANs in 0,5* zt A_reg z : ' + str(
                np.sum(np.isnan(0.5 * np.dot(np.conj(Z).T, self.spatial_regularisation(D, Wd, Z))))))
        ######################################

        # Stopping criterion
        stop = obj[nb_it]
        # Iterations
        # while stop > 1e-8 and nb_it < 2000:
        while stop > 1e-4 and nb_it < 200:
            areg = self.spatial_regularisation(D, Wd, p)
            alpha = np.dot(np.conj(r).T, r) * (np.dot(np.conj(p).T, A.dot(p)) + np.dot(np.conj(p).T, areg)) ** -1
            Z = Z + alpha * p
            r_old = r.copy()
            r = r + alpha * (A.dot(p) + areg)
            beta = np.dot(np.conj(r).T, r) / np.dot(np.conj(r_old).T, r_old)
            p = -r + beta * p
            nb_it += 1
            obj.append(0.5 * np.dot(np.conj(Z).T, A.dot(Z)) + np.dot(np.conj(B.T), Z) + C + 0.5 * np.dot(np.conj(Z).T,
                                                                                                         self.spatial_regularisation(
                                                                                                             D, Wd, Z)))
            print(str(nb_it) + ' -- Objective function value : ' + str(obj[nb_it]))
            stop = (obj[-2] - obj[-1]) / obj[-2]
            # if save_it:
            #    hdu = fits.PrimaryHDU(self._postprocess(Z))
            #    hdu.writeto(SAVE + 'control/z_ah0_' + str(nb_it + 1) + 'mu' + str(self.mu) + '.fits', overwrite=True)

        t2 = time()
        print('Cg Computation time : ' + str(np.round((t2 - t1) / 60)) + 'min ' + str(np.round((t2 - t1) % 60)) + 's.')

        return Z, obj

    def __call__(self) -> np.array:
        # Linear System
        A = self.A
        B = self.B
        C = self.C
        Z = self.Z.copy()
        D, Wd = self.preprocess_spatial_regularisation()
        Zfusion, obj = self.conjugate_gradient(A, D, Wd, B, C, Z)
        Zfusion = self.postprocess(Zfusion)
        return Zfusion, obj

    # def __call__(self, regP =[], regKw = {},
    #                   gradP=[], gradKw= {}) -> np.array:
# @abstractmethod
# def __call__(self, regP=[], regKw={},
#                gradP=[], gradKw={}) -> np.array:
#         out = self.conjugate_gradient(*gradP, **gradKw)#

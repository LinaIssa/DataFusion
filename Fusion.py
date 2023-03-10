from time import time
import os.path
import pickle
import numpy as np
import scipy.sparse as sp

from abc import ABC, abstractmethod
from tools import get_h_band, get_g_band, aliasing_adj, _centered
from Cube import CubeHyperSpectral, CubeMultiSpectral
from astropy.io import fits


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
    :param PSF_MS : the path to the PSF file of NirCam defined in the configuration file
    :param PSF_HS : the path to the PSF file of NirSpec defined in the configuraiton file
    :param nc: a parameter defined in the configuration file
    :param nr: a parameter defined in the configuration file
    :param mu: a parameter that controls the regularisation
    strength. By default, mu=10

    '''

    def __init__(self, cubeMultiSpectral: CubeMultiSpectral, cubeHyperSpectral: CubeHyperSpectral, Lm: np.array,
                 PSF_MS: str, PSF_HS: str, nc: int, nr: int, mu: int, **kwargs) -> None:

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

        self.PSF_MS = PSF_MS
        self.PSF_HS = PSF_HS

    @abstractmethod
    def spatial_regularisation(self, D: np.ndarray, Wd: np.ndarray, Z: np.ndarray) -> np.array:
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
    def _M(i: int, j: int, phv: np.ndarray, nr: int, nc: int):
        """
        @author: Lina Issa, adapted from Claire Guilloteau's code FRHOMAGE

        is called in the Anc function
        """
        res = np.zeros(nr * nc, dtype=np.complex)
        nf = np.shape(phv)[0]
        for m in range(nf):
            res += np.conj(phv[m, j]) * phv[m, i]
        return res

    @staticmethod
    def _C(i, j, V, row, col, nr, nc, d, PSF_HS: str):
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        is called in the Ans function

        """
        res = np.zeros(nr * nc * d ** 2, dtype=complex)
        lh = len(V)
        for m in range(lh):
            g = get_g_band(PSF_HS, m)
            gntng = d ** (-2) * np.conj(g[row]) * g[col]
            res += (V[m, i] * gntng * V[m, j])
        return res

    @staticmethod
    def _nTn_sparse(nr: int, nc: int, d: int):
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        is called in the Ans function
        """
        print("Starting sparse computation : ")
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
        print("Sparse computation done !")

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
    def _PHV(lacp: int, Lm: np.ndarray, V, nr: int, nc: int, PSF_MS: str) -> np.array:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        computes the spectral and spatial degradation operator in the spectral reduced space induced by V.
        That is  Lm * M * V in which M corresponds to the NirCam's PSF, LM the spectral degradation.

        is called in the _Anc method

        """

        nf = Lm.shape[0]
        lh = V.shape[0]

        res = np.zeros((nf, lacp, nr * nc), dtype=complex)
        print(' *** PHV computation ***')
        for m in range(nf):
            for i in range(lacp):
                sum_h = np.zeros(nr * nc, dtype=complex)
                for l in range(lh):
                    PSF_spatial = get_h_band(PSF_MS, l)  # TODO PSF_image stored as a np.array in the initialisation
                    sum_h += PSF_spatial * Lm[m, l] * V[l, i]  # correction temporaire
                res[m, i] = sum_h
        print(' *** PHV computation done !! ***')
        return res

    def _Anc(self) -> np.array:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        Computes the part of the matrix A related to the multi-spectral image. Called in MatrixA_data method
        """
        nc = self.nc
        nr = self.nr
        lacp = self.lacp

        t1 = time()
        row = np.arange(nr * nc)
        row = np.tile(row, (1, lacp ** 2))[0]
        for i in range(lacp):
            row[i * nr * nc * lacp:(i + 1) * nr * nc * lacp] += i * nr * nc
        col = np.arange(nr * nc * lacp)
        col = np.tile(col, (1, lacp))[0]

        phv = self._PHV(lacp, self.Lm, self.V, nr, nc, self.PSF_MS)
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

    def _Ans(self, Lh: np.ndarray) -> np.array:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        Computes the part of the matrix A related to the hyper-spectral image. Called in MatrixA_data method
        """
        nc = self.nc
        nr = self.nr
        d = self.d
        lacp = self.lacp

        t1 = time()
        ntn = self._nTn_sparse(nr, nc, d)
        V = np.dot(np.diag(Lh), self.V)
        row = np.tile(ntn.row, (1, lacp ** 2))[0]
        for i in range(lacp):
            row[i * nr * nc * lacp * d ** 2:(i + 1) * nr * nc * lacp * d ** 2] += i * nr * nc
        col = np.zeros(nr * nc * d ** 2 * lacp)
        for i in range(lacp):
            col[i * nr * nc * d ** 2:(i + 1) * nr * nc * d ** 2] = ntn.col + i * nr * nc
        col = np.tile(col, (1, lacp))[0]
        mat = np.reshape(np.reshape(np.arange(lacp ** 2), (lacp, lacp)).T, lacp ** 2)
        data = np.zeros((lacp ** 2, nr * nc * d ** 2), dtype=complex)
        print("Starting spatial PSF computation for NirSpec image")
        for i in range(lacp):
            # print('i='+str(i))
            for j in range(lacp - i):
                # print('j='+str(j+i))
                temp = self._C(i, j + i, V, ntn.row, ntn.col, nr, nc, d, self.PSF_HS)
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

    def preprocess_spatial_regularisation(self) -> tuple[np.ndarray, np.ndarray]:
        """
        @author Lina Issa, adapted from Claire Guilloteau's FROMHAGE

        Computing weights is part of the preprocessing for the spatial regularisation, regardless to the type of
        regularisation.
        :return: D operator of finite differences
        :return: Wd the weights matrix
        """
        V = self.V  # the singular values matrix from the PCA decompostion performed on hyperspectral matrix
        Ym = self.Y_multi  # the preprocessed multi-spectral image
        Lm = self.Lm  # the spectral degradation operator on multi-spectral image
        nr, nc = self.nr, self.nc
        #############################################################
        #              Computing the operator of finite difference  #
        #############################################################

        D = self._FiniteDifferenceOperator(nr, nc)

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

    def MatrixA_data(self, Lh: np.ndarray) -> np.array:
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
        print("data-driven and PSF-dependant A computed")
        return A_data

    def MatrixB_data(self, Lh: np.ndarray) -> np.array:
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
            bnc[l] = get_h_band(self.PSF_MS, l, mode='adj') * bnc[l]
        bnc = np.dot(self.V.T, bnc)
        ###########################################
        #               Hyperspectral Part        #
        ###########################################
        Yh_ = aliasing_adj(Yh, (Yh.shape[0], nr, nc), self.d)
        for l in range(lh):
            Yh_[l] *= get_g_band(self.PSF_HS, l, mode='adj')
        bns = np.dot(np.dot(np.diag(Lh), self.V).T, Yh_)
        bm = np.reshape(-self.sbnc * bnc, np.prod(bnc.shape))
        bh = np.reshape(-self.sbns * bns, np.prod(bns.shape))

        #######################################
        #               Total Matrix B        #
        #######################################

        B_data = bm + bh

        print("data-driven and PSF-dependant B computed")

        return B_data

    def MatrixC_data(self) -> np.array:

        Ym, Yh = self.Y_multi, self.Y_hyper

        Ym = np.reshape(Ym, np.prod(Ym.shape))  # to be consistent with the output of set inputs
        Yh = np.reshape(Yh, np.prod(Yh.shape))  # to be consistent with the output of set inputs

        cm = 0.5 * self.sbnc * np.dot(np.conj(Ym).T, Ym)
        ch = 0.5 * self.sbns * np.dot(np.conj(Yh).T, Yh)

        C = cm + ch

        print("data-driven matrix C computed")

        return C

    def postprocess(self, Zfusion: np.ndarray) -> np.array:

        Zfusion = np.reshape(Zfusion, (self.lacp, self.nr, self.nc))
        Zfusion = np.fft.ifft2(Zfusion, norm='ortho')
        Zfusion = np.real(
            _centered(Zfusion[:, :-2, :-2], (self.lacp, self.nr - 2 * self.fact_pad, self.nc - 2 * self.fact_pad)))

        return Zfusion


class Weighted_Sobolev_Regularisation(Fusion):
    """
    This fusion class implements a Sobolev Regularisation method with weights as described in papers [1,2] and in Claire
    Guilloteau's thesis manuscrit.
    
    :param Lh        : spectral degradation operator onto the hyperspectral image
    :param outputdir : path to the output directory in which the results are stored
    :param first_run : boolean by default True. The first run computes the matrix A, B C and performs the spectral
     reduction and stores them so that they can be loaded for the other runs, assuming that the same images are fused
    """

    def __init__(self, cubeMultiSpectral: CubeMultiSpectral,
                 cubeHyperSpectral: CubeHyperSpectral,
                 Lm: np.ndarray, Lh: np.ndarray,
                 PSF_MS: str, PSF_HS: str,
                 nc: int, nr: int,
                 output_dir: str, mu: int = 10,
                 first_run: bool = True) -> None:

        super().__init__(cubeMultiSpectral, cubeHyperSpectral, Lm, PSF_MS, PSF_HS, nc, nr, mu)
        self.outputDir = output_dir
        # ***************************************************************************************************************
        #                                               Compute Linear System
        # ***************************************************************************************************************
        if first_run is True:

            print("Constructing the linear system : ")

            self.A = self.MatrixA_data(Lh)
            self.B = self.MatrixB_data(Lh)
            self.C = self.MatrixC_data()

            with open(os.path.join(output_dir, 'A.dat'), 'wb') as f:
                pickle.dump(self.A, f)
            with open(os.path.join(output_dir, 'B.dat'), 'wb') as f:
                pickle.dump(self.B, f)
            with open(os.path.join(output_dir, 'C.dat'), 'wb') as f:
                pickle.dump(self.C, f)

        else:

            print("Loading the pre-computed linear system")

            with open(os.path.join(output_dir, 'A.dat'), 'rb') as f:
                self.A = pickle.load(f)
            with open(os.path.join(output_dir, 'B.dat'), 'rb') as f:
                self.B = pickle.load(f)
            with open(os.path.join(output_dir, 'C.dat'), 'rb') as f:
                self.C = pickle.load(f)

    def spatial_regularisation(self, D: np.ndarray, Wd: np.ndarray, Z: np.ndarray) -> np.array:
        """
        Constructs the regularisation part of the matrix A. Needs D and Wd from the preprocessing
        :param D :  as computed by the spatial regularisation preprocessing
        :param Wd:  as computed by the spatial regularisation preprocessing
        :param Z :  the reduced hyperspectral image computed by the PCA. The value of Z chang
        :return: Areg, the part of the matrix A that contains the regularisation information
        """
        Z = np.reshape(
            Z,
            (self.lacp, Wd[0].shape[1], Wd[0].shape[2])
        )

        gx = np.fft.fft2(
            np.real(
                np.fft.ifft2(
                    Z * D[0]
                )
            ) * Wd[0] ** 2,
            norm='ortho') * np.conj(D[0])

        gy = np.fft.fft2(
            np.real(
                np.fft.ifft2(
                    Z * D[1]
                )
            ) * Wd[1] ** 2,
            norm='ortho') * np.conj(D[1])

        Areg = 2 * self.mu * np.reshape(gx + gy,
                                        np.prod(gx.shape)
                                        )
        return Areg

    def conjugate_gradient(self, A, D, Wd, B, C, Z) -> tuple[np.ndarray, list]:
        """
        :param A: the vectorized data-driven A from MatrixA_data class method.
        :param D: the operator of finite differences from the spatial regularisation preprocess method
        :param Wd: the weights matrix retrieved from the spatial regularisation preprocess method
        :param B: the vectorized matrix B from MatrixB_data class method
        :param C: the vectorized matrix C from MatrixB_data class method
        :param Z: the spectral-reduced hyperspectral image stored as a class attribute.
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
        obj = [
            0.5 * np.dot(
                np.conj(Z).T, A.dot(Z)
            ) + np.dot(
                np.conj(B.T), Z
            ) + C + 0.5 * np.dot(
                np.conj(Z).T,
                self.spatial_regularisation(D, Wd, Z)
            )
        ]
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
            obj.append(
                0.5 * np.dot(
                    np.conj(Z).T, A.dot(Z)
                ) + np.dot(
                    np.conj(B.T), Z
                ) + C + 0.5 * np.dot(
                    np.conj(Z).T,
                    self.spatial_regularisation(D, Wd, Z)
                )
            )
            print(str(nb_it) + ' -- Objective function value : ' + str(obj[nb_it]))
            stop = (obj[-2] - obj[-1]) / obj[-2]

        t2 = time()
        print('Cg Computation time : ' + str(np.round((t2 - t1) / 60)) + 'min ' + str(np.round((t2 - t1) % 60)) + 's.')

        return Z, obj

    def __call__(self, save_it: bool = False, **kwargs) -> tuple[np.ndarray, list]:
        """
        @author Lina Issa

        :param save_it: by default the product fusion is not saved.
        """
        mu = self.mu
        A = self.A
        B = self.B
        C = self.C
        Z = np.reshape(  # to be consistent with the output of Z in set_inputs of FRHOMAGE
            self.Z,
            np.prod(self.Z.shape)
        )

        print("-------------- Spatial Regularisation Implementation : --------------")

        D, Wd = self.preprocess_spatial_regularisation()

        print("-------------- Conjugate Gradient procedure :-------------- ")

        Zfusion, obj = self.conjugate_gradient(A, D, Wd, B, C, Z)

        print("-------------- Postprocessing of the product function --------------")

        Zfusion = self.postprocess(Zfusion)

        if save_it is True:
            hdu = fits.PrimaryHDU(Zfusion)
            hdu.writeto(os.path.join(self.outputDir, f'Zfusion_SobolevWeightsReg_{mu}.fits'), overwrite=True)

        print("-------------- Fusion performed successfully ! --------------")

        return Zfusion, obj


class Sobolev_Regularisation(Fusion):
    """
    This fusion class implements a simple Sobolev Regularisation method as described in papers [1,2] and in Claire
    Guilloteau's thesis manuscrit.

    :param Lh        : spectral degradation operator onto the hyperspectral image
    :param outputdir : path to the output directory in which the results are stored
    :param first_run : boolean by default True. The first run computes the matrix A, B C and performs the spectral
     reduction and stores them so that they can be loaded for the other runs, assuming that the same images are fused
    """
    def __init__(self, cubeMultiSpectral: CubeMultiSpectral,
                 cubeHyperSpectral: CubeHyperSpectral,
                 Lm: np.ndarray, Lh: np.ndarray,
                 PSF_MS: str, PSF_HS: str,
                 nc: int, nr: int,
                 output_dir: str, mu: int = 10,
                 first_run: bool = True) -> None:

        super().__init__(cubeMultiSpectral, cubeHyperSpectral, Lm, PSF_MS, PSF_HS, nc, nr, mu)
        self.outputDir = output_dir
        # ***************************************************************************************************************
        #                                               Compute Linear System
        # ***************************************************************************************************************
        if first_run is True:

            print("Constructing the linear system : ")

            self.A = self.MatrixA_data(Lh)
            self.B = self.MatrixB_data(Lh)
            self.C = self.MatrixC_data()

            with open(os.path.join(output_dir, 'A.dat'), 'wb') as f:
                pickle.dump(self.A, f)
            with open(os.path.join(output_dir, 'B.dat'), 'wb') as f:
                pickle.dump(self.B, f)
            with open(os.path.join(output_dir, 'C.dat'), 'wb') as f:
                pickle.dump(self.C, f)

        else:

            print("Loading the pre-computed linear system")

            with open(os.path.join(output_dir, 'A.dat'), 'rb') as f:
                self.A = pickle.load(f)
            with open(os.path.join(output_dir, 'B.dat'), 'rb') as f:
                self.B = pickle.load(f)
            with open(os.path.join(output_dir, 'C.dat'), 'rb') as f:
                self.C = pickle.load(f)

    def __call__(self, save_it: bool = False, **kwargs) -> tuple[np.ndarray, list]:
        """
        @author Lina Issa

        :param save_it: by default the product fusion is not saved.
        """
        mu = self.mu
        A = self.A
        B = self.B
        C = self.C
        Z = np.reshape(
            self.Z,
            np.prod(self.Z.shape)
        )

        print("-------------- Conjugate Gradient procedure :-------------- ")

        Zfusion, obj = self.conjugate_gradient(A, B, C, Z)

        print("-------------- Postprocessing of the product function --------------")

        Zfusion = self.postprocess(Zfusion)

        if save_it is True:
            hdu = fits.PrimaryHDU(Zfusion)
            hdu.writeto(os.path.join(self.outputDir, f'Zfusion_SobolevReg_{mu}.fits'), overwrite=True)

        print("-------------- Fusion performed successfully ! --------------")

        return Zfusion, obj

    def spatial_regularisation(self, *args, **kwargs):
        pass

    def conjugate_gradient(self, A, B, C, Z, save_it=False) -> tuple[np.ndarray, list]:
        """
        :param A: the vectorized data-driven A from MatrixA_data class method.
        :param B: the vectorized matrix B from MatrixB_data class method
        :param C: the vectorized matrix C from MatrixB_data class method
        :param Z: the spectral-reduced hyperspectral image stored as a class attribute.
        :param save_it: boolean, by default False
        :return: Z and obj
        """
        #### Conjugate gradient iterations #
        print('--- CONJUGATE GRADIENT ALGORITHM ---')
        t1 = time()
        nb_it = 0
        z0 = Z.copy()
        # Initialize

        ########## Control procedure ##########
        # print('NANs in Az, b and A_reg z : '+str(np.sum(np.isnan(A.dot(z))))+' ; '+str(np.sum(np.isnan(b)))+' ; '+str(np.sum(np.isnan(compute_Areg(lacp, mu1, D, Wd, z)))))
        #######################################

        r = A.dot(Z) + B
        p = -r.copy()
        # Objective function
        obj = [0.5 * np.dot(np.conj(Z).T, A.dot(Z)) + np.dot(np.conj(B.T), Z) + C]
        print(str(nb_it) + ' -- Objective function value : ' + str(obj[nb_it]))

        # Stopping criterion
        stop = obj[nb_it]
        # Iterations
        while stop > 1e-4 and nb_it < 200:

            alpha = np.dot(np.conj(r).T, r) / (np.dot(np.conj(p).T, A.dot(p)))
            Z = Z + alpha * p
            r_old = r.copy()
            r = r + alpha * (A.dot(p))
            beta = np.dot(np.conj(r).T, r) / np.dot(np.conj(r_old).T, r_old)
            p = -r + beta * p
            nb_it += 1
            obj.append(0.5 * np.dot(np.conj(Z).T, A.dot(Z)) + np.dot(np.conj(B.T), Z) + C)
            print(str(nb_it) + ' -- Objective function value : ' + str(obj[nb_it]))
            stop = (obj[-2] - obj[-1]) / obj[-2]

        t2 = time()

        print('Cg Computation time : ' + str(np.round((t2 - t1) / 60)) + 'min ' + str(np.round((t2 - t1) % 60)) + 's.')

        return Z, obj


    # def __call__(self, regP =[], regKw = {},
    #                   gradP=[], gradKw= {}) -> np.array:
# @abstractmethod
# def __call__(self, regP=[], regKw={},
#                gradP=[], gradKw={}) -> np.array:
#         out = self.conjugate_gradient(*gradP, **gradKw)#

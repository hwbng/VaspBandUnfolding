#!/usr/bin/env python

import numpy as np

try:
    from pysbt import pysbt
except ImportError:
    print("Please install pySBT (https://github.com/QijingZheng/pySBT)!")

from scipy.fft import fftn, ifftn
from scipy.sparse import block_diag

from paw import gvectors, nonlq, pawpotcar
from vasp_constant import *


def calc_dipole_mat(wfc_i, wfc_j, ks_i, ks_j):
    """
    Dipole transition within the electric dipole approximation (EDA).
    Please refer to this post for more details.

    https://qijingzheng.github.io/posts/Light-Matter-Interaction-and-Dipole-Transition-Matrix/

    The dipole transition matrix elements in the length gauge is given by:

            <psi_nk | e r | psi_mk>

    In periodic systems, the position operator "r" is not well-defined.
    Therefore, we first evaluate the momentum operator matrix in the velocity
    gauge, i.e.

            <psi_nk | p | psi_mk>

    And then use simple "p-r" relation to apprimate the dipole transition
    matrix element

                                    -i⋅h
        <psi_nk | r | psi_mk> =  -------------- ⋅ <psi_nk | p | psi_mk>
                                m⋅(En - Em)

    Apparently, the above equaiton is not valid for the case Em == En. In
    this case, we just set the dipole matrix element to be 0.

    ################################################################################
    NOTE that, the simple "p-r" relation only applies to molecular or finite
    system, and there might be problem in directly using it for periodic
    system. Please refer to this paper for more details.

    "Relation between the interband dipole and momentum matrix elements in
    semiconductors"
    (https://journals.aps.org/prb/pdf/10.1103/PhysRevB.87.125301)

    ################################################################################
    """

    # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
    # initial and final states
    assert len(ks_i) == len(ks_j) == 3, "Must be three indexes!"
    assert ks_i[1] == ks_j[1], "k-point of the two states differ!"
    wfc_i._pswfc.checkIndex(*ks_i)
    wfc_j._pswfc.checkIndex(*ks_j)

    # energy differences between the two states
    Emk = wfc_i._pswfc._bands[ks_i[0] - 1, ks_i[1] - 1, ks_i[2] - 1]
    Enk = wfc_j._pswfc._bands[ks_j[0] - 1, ks_j[1] - 1, ks_j[2] - 1]
    dE = Enk - Emk

    # if energies of the initial and final states are the same, set the
    # dipole transition moment zero.
    if np.allclose(dE, 0.0):
        return 0.0

    moment_mat = calc_moment_mat(wfc_i, wfc_j, ks_i, ks_j)
    dipole_mat = -1j / (dE / (2 * RYTOEV)) * moment_mat * AUTOA * AUTDEBYE

    return Emk, Enk, dE, dipole_mat


def calc_moment_mat(wfc_i, wfc_j, ks_i, ks_j):
    """
    The momentum operator matrix in the velocity gauge

            <psi_nk | p | psi_mk> = hbar <u_nk | k - i nabla | u_mk>

    In PAW, the matrix element can be divided into plane-wave parts and
    one-center parts, i.e.

        <u_nk | k - i nabla | u_mk> = <tilde_u_nk | k - i nabla | tilde_u_mk>
                                    - \sum_ij <tilde_u_nk | p_i><p_j | tilde_u_mk>
                                    \times i [
                                        <phi_i | nabla | phi_j>
                                        -
                                        <tilde_phi_i | nabla | tilde_phi_j>
                                    ]

    where | u_nk > and | tilde_u_nk > are cell-periodic part of the AE/PS
    wavefunctions, | p_j > is the PAW projector function and | phi_j > and
    | tilde_phi_j > are PAW AE/PS partial waves.

    The nabla operator matrix elements between the pseudo-wavefuncitons

        <tilde_u_nk | k - i nabla | tilde_u_mk>

    = \sum_G C_nk(G).conj() * C_mk(G) * [k + G]

    where C_nk(G) is the plane-wave coefficients for | u_nk >.

    """

    # ks_i and ks_j are list containing spin-, kpoint- and band-index of the
    # initial and final states
    assert len(ks_i) == len(ks_j) == 3, "Must be three indexes!"
    assert ks_i[1] == ks_j[1], "k-point of the two states differ!"
    wfc_i._pswfc.checkIndex(*ks_i)
    wfc_j._pswfc.checkIndex(*ks_j)

    # k-points in direct coordinate
    k0 = wfc_i._pswfc._kvecs[ks_i[1] - 1]
    # plane-waves in direct coordinates
    G0 = wfc_i._pswfc.gvectors(ikpt=ks_i[1])
    G1 = wfc_j._pswfc.gvectors(ikpt=ks_j[1])
    assert np.array_equal(G0, G1), "Plane waves of the two states differ!"
    # G + k in Cartesian coordinates
    Gk = np.dot(
        G0 + k0,  # G in direct coordinates
        wfc_i._pswfc._Bcell * TPI,  # reciprocal basis x 2pi
    )

    # plane-wave coefficients for initial (mk) and final (nk) states
    CG_mk = wfc_i._pswfc.readBandCoeff(*ks_i)
    CG_nk = wfc_j._pswfc.readBandCoeff(*ks_j)
    ovlap = CG_nk.conj() * CG_mk

    ################################################################################
    # Momentum operator matrix element between pseudo-wavefunctions
    ################################################################################
    if wfc_i._pswfc._lgam:
        # for gamma-only, only half the plane-wave coefficients are stored.
        # Moreover, the coefficients are multiplied by a factor of sqrt2

        # G > 0 part
        moment_mat_ps = np.sum(ovlap[:, None] * Gk, axis=0)

        # For gamma-only version, add the other half plane-waves, G' = -G
        # G < 0 part, C(G) = C(-G).conj()
        moment_mat_ps -= np.sum(ovlap[:, None].conj() * Gk, axis=0)

        # remove the sqrt2 factor added by VASP
        moment_mat_ps /= 2.0
    elif wfc_i._pswfc._lsoc:
        moment_mat_ps = np.sum(ovlap[:, None] * np.r_[Gk, Gk], axis=0)
        # raise NotImplementedError('Non-collinear version currently not supported!')
    else:
        moment_mat_ps = np.sum(ovlap[:, None] * Gk, axis=0)

    ################################################################################
    # One-center correction
    ################################################################################

    projector = nonlq(
        wfc_i._atoms,
        wfc_i._pscut,
        wfc_i._pawpp,
        k=k0,
        lgam=wfc_i._pswfc._lgam,
        gamma_half=wfc_i._pswfc._gam_half,
    )

    if wfc_i._pswfc._lsoc:
        nplw = Gk.shape[0]
        # spin-up component of the spinor
        beta_mk = projector.proj(CG_mk[:nplw])
        beta_nk = projector.proj(CG_nk[:nplw])

        # spin-down component of the spinor
        beta_mk2 = projector.proj(CG_mk[nplw:])
        beta_nk2 = projector.proj(CG_nk[nplw:])
    else:
        beta_mk = projector.proj(CG_mk)
        beta_nk = projector.proj(CG_nk)

    # one-center term of momentum operator matrix element
    moment_mat_oc = np.zeros(3, dtype=complex)

    # nproj = 0
    # for ii in range(self._natoms):
    #     itype = self._element_idx[ii]
    #     lmmax = self._pawpp[itype].lmmax
    #     nabla = self._pawpp[itype].get_nablaij(lreal=True)
    #
    #     moment_mat_oc += np.dot(
    #         beta_nk[nproj:nproj+lmmax].conj(),
    #         np.dot(nabla, beta_mk[nproj:nproj+lmmax]).T
    #     )
    #
    #     if self._pswfc._lsoc:
    #         moment_mat_oc += np.dot(
    #             beta_nk2[nproj:nproj+lmmax].conj(),
    #             np.dot(nabla, beta_mk2[nproj:nproj+lmmax]).T
    #         )
    #
    #     nproj += lmmax

    for ii in range(3):
        moment_mat_oc[ii] = beta_nk.conj() @ (wfc_i.get_nablaijs()[ii] @ beta_mk)
        if wfc_i._pswfc._lsoc:
            moment_mat_oc[ii] += beta_nk2.conj() @ (wfc_i.get_nablaijs()[ii] @ beta_mk2)

    return moment_mat_ps - 1j * moment_mat_oc

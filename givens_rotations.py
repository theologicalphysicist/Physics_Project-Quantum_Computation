import numpy as np
import scipy
import openfermion as of
from openfermion import (
	linalg as of_lg
)

#! MAKE SURE YOU "pip install scipy" nd "pip install openfermion" first!!

def getInvDiag(in_matrix):
	a = np.matmul(in_matrix, np.transpose(in_matrix.conjugate()))
	a = scipy.linalg.fractional_matrix_power(a, 0.5)

	return np.linalg.inv(a)

def GivensRotations():
	nsites = 2
	#* Parameters for setting up the Gaussian potential.
	l_up = 1
	m_up = 1.5
	sigma_up = 1
	#* For spin-dependent potentials

	site_index = np.arange(1, nsites + 1)
	spin_up_ham = np.diag([-1.] * (nsites - 1), k=1) + np.diag([-1.] * (nsites - 1), k=-1)
	spin_up_ham += np.diag(-l_up * np.exp((-0.5 * ((site_index - m_up)**2) / (sigma_up**2))))
	spin_up_ham_u = np.matmul(getInvDiag(spin_up_ham), spin_up_ham)

	spin_down_ham = np.diag([-1.] * (nsites - 1), k=1) + np.diag([-1.] * (nsites - 1), k=-1)
	spin_down_ham_u = np.matmul(getInvDiag(spin_down_ham), spin_down_ham)

	up_decomp, up_diag = of_lg.givens_decomposition_square(spin_up_ham_u)
	down_decomp, down_diag = of_lg.givens_decomposition_square(spin_down_ham_u)

	return (up_decomp[0][0][2], down_decomp[0][0][2])
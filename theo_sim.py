import math
import cmath
import os
import sys
# from turtle import down
sys.path.append("C:/Users/dadug/Coding/AduGyamfi-David/Physics_Project-Quantum_Computation/utils")

import scipy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import openfermion as of
from openfermion import (
	linalg as of_lg
)
# import plotly.express as px

from utils import (
	 gates as gts,
	 quant_funcs as qf
)

ANGLE = math.pi/4
G_ANGLE_UP = -1.5620465856164656
G_ANGLE_DOWN = -1.5707963267948966
IDENTITY2 = np.array([[1, 0], [0, 1]], dtype=complex)
IDENTITY4 = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=complex)

qubits = np.array([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=complex)
#* qubits written as a|0>+b|1>, and only a & b stored in qubit array
#DONE - rewrite qubit as a sum of |0> & |1>
data_sources = ["manila", "bogota", "quito", "theoretical"]

def OneQubitTest():
	test_qubits = qf.apply1QGate(np.array([1, 0]), gts.H_GATE, 0)
	# test_qubits = gts.H_GATE @ test_qubits
	probs = qf.getProbabilities(test_qubits)

	file = open("./ibm_data/tests/1Q_RESULT.csv", "r")
	ibm_data = {}
	count = 0
	for line in file:
		if (count == 0):
			count += 1
		else: 
			data = line.split(",")
			ibm_data[data[0]] = int(data[1]) / 4000

	X_axis = np.arange(len(list(probs.keys())))

	plt.bar(x=X_axis - 0.2, height=list(probs.values()), width=0.4, label="Theoretical Values", color="#a0f")
	plt.bar(x=X_axis + 0.2, height=list(ibm_data.values()), width=0.4, label="ibmq_armonk", color="#fb0")
	plt.xticks(X_axis, list(probs.keys()))
	plt.legend(loc="lower right")
	plt.show()

def TwoQubitTest():
	q = np.array([[1, 0], [1, 0]], dtype=complex)
	q = np.kron(q[0], q[1])

	# q = qf.apply1QGate(q, gts.H_GATE, 0)
	q = qf.apply2QGate(q, gts.CNOT, [0, 1])
	# q = qf.apply2QGate(q, gts.ISWAP, [0, 1])

	probs = qf.getProbabilities(q)
	print(sum(list(probs.values())))

	file = open("./ibm_data/2Q_RESULT.csv", "r")
	ibm_data = {}
	count = 0
	for line in file:
		if (count == 0):
			count += 1
		else: 
			data = line.split(",")
			ibm_data[data[0]] = int(data[1]) / 4000

	X_axis = np.arange(len(list(probs.keys())))

	plt.bar(x=X_axis - 0.2, height=list(probs.values()), width=0.4, label="Theoretical Values")
	plt.bar(x=X_axis + 0.2, height=list(ibm_data.values()), width=0.4, label="ibmq_manila")
	plt.xticks(X_axis, list(probs.keys()))
	plt.legend(loc="lower right")
	plt.show()

def ThreeQubitTest():
	q = np.array([1, 0, 0, 0, 0, 0, 0, 0], dtype=complex)

	# q = qf.apply1QGate(q, gts.H_GATE, 0)
	q = qf.apply2QGate(q, gts.getGivens(), [0, 1], controlled=False)
	# q = qf.apply2QGate(q, gts.CNOT, [0, 1], True)
	# q = qf.apply2QGate(q, gts.ISWAP, [0, 1])
	# q = qf.apply2QGate(q, gts.getCPhase(), [1, 2])

	# apply1QGate(0, H_Gate, test_qubits)
	# apply1QGate(1, X_Gate, test_qubits)
	# apply2QGate([0, 1], iSWAP_Gate, test_qubits)
	# apply2QGate([0, 2], P_Gate, test_qubits)

	probs = qf.getProbabilities(q)

	file = open("./ibm_data/tests/3Q_RESULT_G_01.csv", "r")
	ibm_data = {}
	count = 0
	for line in file:
		if (count == 0):
			count += 1
		else: 
			data = line.split(",")
			ibm_data[data[0]] = int(data[1]) / 4000

	for k in probs.keys():
		if (k not in list(ibm_data.keys())):
			ibm_data[k] = 0

	X_axis = np.arange(len(list(probs.keys())))

	# plt.hist([list(probs.values()), list(ibm_data.values())], bins=np.linspace(0, 1, 1), label=["Theoretical Values", "ibmq_armonk"])
	# bins=np.linspace(0, 2, 1)
	plt.bar(x=X_axis - 0.2, height=list(probs.values()), width=0.4, label="Theoretical Values")
	plt.bar(x=X_axis + 0.2, height=list(ibm_data.values()), width=0.4, label="ibmq_quito")
	plt.xticks(X_axis, list(probs.keys()))
	plt.legend(loc="lower right")
	plt.show()

def FourQubitTest():
	qubits = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=complex)
	qubits = np.kron(np.kron(np.kron(qubits[0], qubits[1]), qubits[2]), qubits[3])

	qubits = qf.apply2QGate(qubits, gts.getGivens(math.pi / 4), [0, 2])
	qubits = qf.apply2QGate(qubits, gts.getGivens(math.pi / 4), [2, 3])

	probs = qf.getProbabilities(qubits)

	file = open("./ibm_data/tests/givens.csv", "r")
	ibm_data = {}
	count = 0
	for line in file:
		if (count == 0):
			count += 1
		else: 
			data = line.split(",")
			ibm_data[data[0]] = int(data[1]) / 4000

	X_axis = np.arange(len(list(probs.keys())))

	# plt.hist([list(probs.values()), list(ibm_data.values())], bins=np.linspace(0, 1, 1), label=["Theoretical Values", "ibmq_armonk"])
	# bins=np.linspace(0, 2, 1)
	plt.bar(x=X_axis - 0.2, height=list(probs.values()), width=0.4, label="Theoretical Values")
	plt.bar(x=X_axis + 0.2, height=list(ibm_data.values()), width=0.4, label="ibmq_lima")
	plt.xticks(X_axis, list(probs.keys()))
	plt.legend(loc="lower right")
	plt.show()

def InitialPreparation(q, uG, dG):

	q = qf.apply1QGate(q, gts.X, 2)
	q = qf.apply1QGate(q, gts.X, 3)

	q = qf.apply1QGate(q, gts.X, 0)
	# # print(q)
	q = qf.apply1QGate(q, gts.X, 2)
	# # print(q)

	q = qf.apply2QGate(q, gts.getGivens(uG), [0, 1], controlled=False)
	q = qf.apply2QGate(q, gts.getGivens(dG), [2, 3], controlled=False)
	return q

def TrotterStep(q):
	q = qf.apply2QGate(q, gts.getK(-0.3), [0, 1], controlled=False)
	q = qf.apply2QGate(q, gts.getK(-0.3), [2, 3], controlled=False)
	q = qf.apply2QGate(q, gts.getCPhase(0.9), [0, 2], controlled=True)
	q = qf.apply2QGate(q, gts.ISWAP, [0, 1], controlled=False)
	q = qf.apply2QGate(q, gts.ISWAP, [2, 3], controlled=False)
	q = qf.apply2QGate(q, gts.getCPhase(0.9), [0, 2], controlled=True)
	q = qf.apply2QGate(q, gts.getK(-0.3 + (math.pi / 2)), [0, 1], controlled=False)
	q = qf.apply2QGate(q, gts.getK(-0.3 + (math.pi / 2)), [2, 3], controlled=False)

	return q

def getIBMData(machine:str):
	file = open("./ibm_data/Fermi_Hubbard/initial_prep/" + machine + ".csv", "r")
	data = {}
	count = 0
	for line in file:
		if (count == 0):
			count += 1
		else: 
			measurement = line.split(",")
			data[measurement[0]] = int(measurement[1]) / 4000
		
	return data

def processData(ibm_data:dict):
	site_1, site_2 = [0, 0], [0, 0]
	# print(site_1, site_2)
	for k in ibm_data.keys():
		if (k[0] == "0"):
			site_1[0] += ibm_data[k]
		if (k[1] == "0"):
			site_2[0] += ibm_data[k]
		if (k[2] == "0"):
			site_1[1] += ibm_data[k]
		if (k[3] == "0"):
			site_1[1] += ibm_data[k]
	return site_1, site_2

def FermiHubbardModel(uG, dG):
	qubits = np.array([[1, 0], [1, 0], [1, 0], [1, 0]], dtype=complex)
	qubits = np.kron(np.kron(np.kron(qubits[0], qubits[1]), qubits[2]), qubits[3])
	qubits = InitialPreparation(qubits, uG, dG)
	for i in range(0, 0):
		qubits = TrotterStep(qubits)

	# qubits = TrotterStep(qubits)
	# qubits = TrotterStep(qubits)

	# print(qubits)

	probs = qf.getProbabilities(qubits)
	probs_sc = processData(probs)

	quito_data = getIBMData("quito")
	quito_sc = processData(quito_data)
	santiago_data = getIBMData("santiago")
	santiago_sc = processData(santiago_data)
	lima_data = getIBMData("lima")
	lima_sc = processData(lima_data)

	# X_axis = np.arange(len(list(probs.keys())))

	fig = plt.figure()
	ax = fig.add_subplot()

	ax.scatter([1, 2], [abs(probs_sc[0][0] - probs_sc[0][1]), abs(probs_sc[1][0] - probs_sc[1][1])], s=20, c="#a0f", marker="^", label="Theoretical Values")
	ax.scatter([1, 2], [abs(quito_sc[0][0] - quito_sc[0][1]), abs(quito_sc[1][0] - quito_sc[1][1])], s=20, c="#00a", marker="o", label="ibmq_quito")
	ax.scatter([1, 2], [abs(santiago_sc[0][0] - santiago_sc[0][1]), abs(santiago_sc[1][0] - santiago_sc[1][1])], s=20, c="#a00", marker="v", label="ibmq_santiago")
	ax.scatter([1, 2], [abs(lima_sc[0][0] - lima_sc[0][1]), abs(lima_sc[1][0] - lima_sc[1][1])], s=20, c="#0a0", marker=",", label="ibmq_lima")

	# plt.bar(x=X_axis - 0.3, height=list(probs.values()), width=0.2, label="Theoretical Values", color="#a0f")
	# plt.bar(x=X_axis - 0.1, height=list(quito_data.values()), width=0.2, label="ibmq_quito", color="#00a")
	# plt.bar(x=X_axis + 0.1, height=list(santiago_data.values()), width=0.2, label="ibmq_santiago", color="#a00")
	# plt.bar(x=X_axis + 0.3, height=list(lima_data.values()), width=0.2, label="ibmq_lima", color="#0a0")
	# plt.xticks(X_axis, list(probs.keys()), rotation=45)
	plt.title("Spin Density plot with 0 trotter steps")
	plt.legend(loc="upper center")
	plt.show()

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
	spin_up_ham += np.diag(-l_up * np.exp(-0.5 * ((site_index - m_up)**2) / (sigma_up**2)))
	spin_up_ham_u = np.matmul(getInvDiag(spin_up_ham), spin_up_ham)

	spin_down_ham = np.diag([-1.] * (nsites - 1), k=1) + np.diag([-1.] * (nsites - 1), k=-1)
	spin_down_ham_u = np.matmul(getInvDiag(spin_down_ham), spin_down_ham)

	up_decomp, up_diag = of_lg.givens_decomposition_square(spin_up_ham_u)
	down_decomp, down_diag = of_lg.givens_decomposition_square(spin_down_ham_u)
	# print("up = {}, down = {}".format(up_decomp, down_decomp))

	return (up_decomp[0][0][2], down_decomp[0][0][2])

def main():

	up_G, down_G = GivensRotations()
	# print("Up angle = {}, down angle = {}".format(up_G, down_G))

	# OneQubitTest()
	# TwoQubitTest()
	# ThreeQubitTest()
	# FourQubitTest()
	FermiHubbardModel(up_G, down_G)
	
main()

#! a|00> + b|01>...
#! a|0> + b|1> || c|a> + d|1>
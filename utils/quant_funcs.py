import numpy as np
import gates as gts

def getNumberOfQubits(l:int):
	i = 0
	while (2 ** i != l):
		i += 1
	return i

def apply1QGate(q, G, index: int):
	gate = G
	for i in range(getNumberOfQubits(len(q)) - 1, index, -1):
		gate = np.kron(gate, gts.IDENTITY2)
	for i in range(index, 0, -1):
		gate = np.kron(gts.IDENTITY2, gate)

	# print(gate)
	return gate @ q

def ABCDecomposition(G, q_len):
	gate = gts.IDENTITY2
	for i in range(1, q_len):
		gate = np.kron(gate, gts.IDENTITY2)

def apply2QGate(q, G, indexes: list, controlled: bool):
	gate = G
	q_count = getNumberOfQubits(len(q))
	if (controlled):
		selector0 = np.array([[1, 0], [0, 0]], dtype=complex)
		selector1 = np.array([[0, 0], [0, 1]], dtype=complex)
		gate_part_1 = selector0
		for i in range(1, q_count):
			gate_part_1 = np.kron(gate_part_1, gts.IDENTITY2)
		# gate_part_1 = np.kron(np.outer([1, 0], [1, 0]), gate_part_1) if (indexes[0] == 0) else np.kron(gate_part_1, np.outer([1, 0], [1, 0]))
		# for i in range(indexes[0] + 1, q_count - 1):
		# 	gate_part_1 = np.kron(gate_part_1, gts.IDENTITY2)

		gate_part_2 = selector1
		for i in range(1, indexes[1]):
			gate_part_2 = np.kron(gate_part_2, gts.IDENTITY2)
		gate_part_2 = np.kron(gate_part_2, G)
		for i in range(indexes[1], q_count - 1):
			gate_part_2 = np.kron(gate_part_2, gts.IDENTITY2)
		# gate_part_2 = np.kron(gate_part_2, gts.X) if (G.shape == (4, 4)) else np.kron(gate_part_2, gts.getCPhase())

		gate = gate_part_1 + gate_part_2
	else:
		for i in range(q_count - 1, indexes[1], -1):
			gate = np.kron(gts.IDENTITY2, gate)
		for i in range(indexes[0], 0, -1):
			gate = np.kron(gate, gts.IDENTITY2)

	return gate @ q

def convertToNibbleString(n, len):
	temp = ""
	nums = []
	for i in range(0, len):
		nums.append(2 ** i)
	nums.reverse()
	
	for num in nums:
		if (n // num > 0):
			temp += "1"
			n -= num
		else:
			temp += "0"
	return temp

def populateAllCombinations(ac: dict, q, l = 0):
	for i in range(0, len(q)):
		ac[convertToNibbleString(i, l)] = 0
	return ac

def getProbabilities(q):
	all_combinations = {}
	q_len = getNumberOfQubits(len(q))
	# if (q_len == 1):
	# 	all_combinations = populateAllCombinations(all_combinations, q, q_len)
	# 	for o in list(all_combinations.keys()):
	# 		prob = 1.0
	# 		for i in range(0, len(o)):
	# 			prob *= (q[1].real ** 2 + q[1].imag ** 2) if (o[i] == "1") else prob * (q[0].real ** 2 + q[0].imag ** 2)
	# 		all_combinations[o] = prob
	# else:
	all_combinations = populateAllCombinations(all_combinations, q, q_len)
	for i in range(0, len(all_combinations.keys())):
		all_combinations[list(all_combinations.keys())[i]] = q[i].real ** 2 + q[i].imag ** 2

	return all_combinations
import math, cmath

import numpy as np

IDENTITY2 = np.array(
	[[1, 0], [0, 1]], 
	dtype=complex
	)
IDENTITY4 = np.array(
	[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], 
	dtype=complex
)
H_GATE = np.array(
	[[1 / math.sqrt(2), 1 / math.sqrt(2)], [1 / math.sqrt(2), -1 / math.sqrt(2)]],
	dtype=complex
)
X = np.array(
	[[0, 1], [1, 0]], 
	dtype=complex
)
ISWAP = np.array(
	[[1, 0, 0, 0], [0, 0, complex(0, 1), 0], [0, complex(0, 1), 0, 0], [0, 0, 0, 1]], 
	dtype=complex
)
def getGivens(angle:float=math.pi / 4):
	return np.array(
		[[1, 0, 0, 0], [0, math.cos(angle), -math.sin(angle), 0], [0, math.sin(angle), math.cos(angle), 0], [0, 0, 0, 1]], 
		dtype=complex
	)
CNOT = np.array(
	[[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
	dtype=complex
)
def getCPhase(angle=math.pi/20):
	return np.array(
		[[1, 0], [0, cmath.exp(-1 * complex(0, 1) * angle)]],
		dtype=complex
	)

def getK(angle:float=math.pi / 4):
	return np.array(
		[[1, 0, 0, 0], [0, math.cos(angle), -complex(0, 1) * math.sin(angle), 0], [0, -complex(0, 1) * math.sin(angle), math.cos(angle), 0], [0, 0, 0, 1]], 
		dtype=complex
	)

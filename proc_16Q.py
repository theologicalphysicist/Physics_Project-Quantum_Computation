import numpy as np
import matplotlib.pyplot as plt

def getIBMData(ibm_data:str):
	file = open("./ibm_data/Fermi_Hubbard/16Q/" + ibm_data + ".csv", "r")
	data = {}
	count = 0
	for line in file:
		if (count == 0):
			count += 1
		else: 
			measurement = line.split(",")
			data[measurement[0]] = int(measurement[1]) / 4000
		
	return data

init_prep_data = getIBMData("2t")

del_list = []
for k in init_prep_data.keys():
	if(init_prep_data[k]) < 40/4000:
		del_list.append(k)

for i in del_list:
	del init_prep_data[i]

print(list(init_prep_data.keys()))

X_axis = np.arange(len(list(init_prep_data.keys())))

plt.bar(x=X_axis, height=list(init_prep_data.values()), label="ibm_qasm_simulator", color="#a9f")
plt.xticks(X_axis, list(init_prep_data.keys()), rotation=45)
plt.legend(loc="upper right")
plt.show()


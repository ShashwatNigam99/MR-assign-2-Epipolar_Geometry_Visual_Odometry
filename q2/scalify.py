import numpy as np

scale_list = []
with open("ground-truth.txt") as file:
	for line in file:
		l = line.split()
		test_list = list(map(float, l)) 
		scale = np.linalg.norm([test_list[3], test_list[7], test_list[11]])
		scale_list.append(scale)

cnt = 0
f = open('final.txt', 'w')

with open("output.txt") as file:
	for line in file:
		l = line.split()
		
		test_list = list(map(np.float64, l)) 

		test_list[3] *= scale_list[cnt]
		test_list[7] *= scale_list[cnt]
		test_list[11] *= scale_list[cnt]

		# print(test_list)
		for i in range(len(test_list)):
			item = test_list[i]
			if i != 11:
				f.write("%s " % item)
			else:
				f.write("%s" % item)
		f.write("\n")

		cnt += 1

		# print([test_list[3], test_list[7], test_list[11]])

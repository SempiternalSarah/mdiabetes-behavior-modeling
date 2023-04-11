import numpy as np

array = np.loadtxt("temp.out", dtype=str, delimiter="\n")
checklist = []
# print(array)
for x in array:
    if x in checklist:
        print(x)
    checklist.append(x)
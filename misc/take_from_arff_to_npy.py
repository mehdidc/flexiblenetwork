import arff
import sys
take = int(sys.argv[2])
i = 0
print "filename.arff take out.npy"
import numpy as np
data = []
targets = []
f = open(sys.argv[1])

out_arff = open("out.arff", "w")
while i < take:
    l = f.readline()
    if l[0] == '@' or l == "\n":
        out_arff.write(l)
        continue
    else:
        l = l[0:-1]
        D = l.split(",")
        if len(D)>1:
            out_arff.write(",".join(D[0:-2]) + "," + D[-1].strip() + "\n")
            data.append(map(lambda x:float(x.strip()), D[0:-2]))
            targets.append(D[-1].strip())
            i += 1
f.close()

out_arff.close()
np.save(sys.argv[3], (data, targets))


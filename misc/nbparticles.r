
datafile = "/Users/dubard/Documents/Projects/RawData/NbParticles.arff"

data = read.arff(datafile)
nbparticles = data[1:nrow(data), 9722]
print(nbparticles)
hist(nbparticles)

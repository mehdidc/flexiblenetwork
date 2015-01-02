library(foreign)
datafile = "../../data/ILC10gev_2500/data.arff"
data = read.arff(datafile)

energy = vector(length=30)
energy[] = 0


x = 1
for(i in (seq(1, 9720, by=18*18)) ){
    energy[x] = sum(data[1:nrow(data), i:(i+18*18-1) ]) / nrow(data)
    x = x + 1
}

par(mfrow=c(2,2))
plot(1:30, energy, type='l')
plot(cumsum(energy), type='l')
hist(energy)

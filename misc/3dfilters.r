library(foreign)
library(plot3D)
library(misc3d)
library(rgl)
#data = read.arff('/Users/dubard/Documents/Projects/Databases/ILC_DeepLearning_Experiment/Deep/test.arff')
#data = array(unlist(data), dim=c( 1230, 9721))
#targets = data[1:1230, 9721]
#data = data[1:1230, 1:9720]
#data = array(data, dim=c(1230, 30, 18, 18))

#m = apply(data, c(2, 3, 4), mean)
#s = apply(data, c(2, 3, 4), sd)
#s = s + (s == 0)

show = function(data){
i = 1
D = data[i, 1:30, 1:18, 1:18]
print(targets[i])
D = (D - m) / s
#image3d(D, alpha.power=0.6, jitter=T, radius=3)
print(h)
plot(h)
}


rgl.bg(color=gray(0.4))
show_filters = function(filename, shape){
        a = read.table(filename, sep=",")
        a = array(unlist(a), dim=shape)
        a = a * (a> (max(a)/1.5))
        image3d(a, alpha.power=0.8, radius=1, jitter=F, material=rgl.material)
}


show_filters("filters.txt", c(5, 5, 5))
axes3d()

filename1 = 'dataout/bof_data_20clusters_a594.csv'
# filename1 = 'dataout/bof_data_20clusters_a594_notScaled.csv'
filename2 = 'dataout/bof_data_20clusters_cy5.csv'
mydata1 = read.csv(filename1, header=FALSE)
mydata2 = read.csv(filename2, header=FALSE)
mydata = mydata2
mydata <- cbind( mydata1 , mydata2 )
# Ward Hierarchical Clustering
rownames(mydata) = c(165,    375,    475,    559,    705,    740,    913,
  1126,   1261,   1312,   1517,   1630,   1689,   1867,
  1901,   2156,   2203,   2350,   3014,   3209,   3594,
  3954,   3995,   5030,   5071,   5376,   5597,   5918,
  5933,   6120,   6392,   6474,   6555,   6600,   6927,
  7569,   8533,   8845,   8878,   9064,   9291,   9959,
  10391,  10872,  11198,  11200,  11664,  11716,  12114)
canberra_dist <- dist(mydata, method = "canberra") # distance matrix
fit <- hclust(d=canberra_dist, method="ward.D2") 
plot(fit) # dendogram with p values

# Ward Hierarchical Clustering with Bootstrapped p values
library(pvclust)
fit <- pvclust(t(mydata), method.hclust="ward.D2",
               method.dist="canberra")
plot(fit) # dendogram with p values
# add rectangles around groups highly supported by the data
pvrect(fit, alpha=.90)

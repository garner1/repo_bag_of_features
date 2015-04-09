mydata = read.csv('cy5_pca_c20_t10.csv', header=FALSE)
#mydata = read.csv('a594_pca_c20_t10.csv', header=FALSE)
#mydata = read.csv('cy5_pca_data_50comp_normalized.csv', header=FALSE)
#mydata = read.csv('cy5_pca_data_20comp_normalized.csv', header=FALSE)
#mydata = read.csv('partitioning_data_20clusters_notnormalized.csv', header=FALSE)
#mydata = read.csv('pca_data_20comp_notnormalized.csv', header=FALSE)
#mydata = read.csv('minibatchkmeans_data.csv', header=FALSE)
# Ward Hierarchical Clustering
rownames(mydata) = c(165,    375,    475,    559,    705,    740,    913,
  1126,   1261,   1312,   1517,   1630,   1689,   1867,
  1901,   2156,   2203,   2350,   3014,   3209,   3594,
  3954,   3995,   5030,   5071,   5376,   5597,   5918,
  5933,   6120,   6392,   6474,   6555,   6600,   6927,
  7569,   8533,   8845,   8878,   9064,   9291,   9959,
  10391,  10872,  11198,  11200,  11664,  11716,  12114)
d <- dist(mydata, method = "euclidean") # distance matrix
fit <- hclust(d, method="ward.D2") 
plot(fit) # dendogram with p values

# Ward Hierarchical Clustering with Bootstrapped p values
library(pvclust)
fit <- pvclust(t(mydata), method.hclust="ward.D2",
               method.dist="euclidean")
plot(fit) # dendogram with p values
# add rectangles around groups highly supported by the data
pvrect(fit, alpha=.95)

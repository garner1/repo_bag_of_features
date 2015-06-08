setwd("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features")

data = read.table("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/joined.mytable.table", quote="\"")

data_a594 = read.table("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/joined.mytable.table.a594", quote="\"")

data_cy5 = read.table("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/joined.mytable.table.cy5", quote="\"")

plot(data_a594$V2,data_a594$V3)
plot(data_cy5$V2,data_cy5$V3)

hist(abs(data_cy5$V2-data_cy5$V3))
hist(abs(data_a594$V2-data_a594$V3))

hist(data_cy5$V3/data_cy5$V2,breaks=c(0:1000),xlim=c(0,20))
hist(data_a594$V3/data_a594$V2,breaks=c(0:2000),xlim=c(0,20))

hist(data_cy5$V2/data_cy5$V3,breaks=c(0:1000),xlim=c(0,40))
hist(data_a594$V2/data_a594$V3,breaks=c(0:2000),xlim=c(0,40))

hist(abs(data_cy5$V2-data_cy5$V3)/(data_cy5$V2+data_cy5$V3))
hist(abs(data_a594$V2-data_a594$V3)/(data_a594$V2+data_a594$V3))

metric = abs(data_cy5$V2-data_cy5$V3)/(data_cy5$V2+data_cy5$V3)

newdata_cy5 = cbind(data_cy5, metric)

good_data = subset(newdata_cy5,newdata_cy5$metric<=0.1)
bad_data = subset(newdata_cy5,newdata_cy5$metric>=0.9)
plot(good_data$V2,good_data$V3)
plot(bad_data$V2,bad_data$V3)
summary(good_data)
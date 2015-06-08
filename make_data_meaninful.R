setwd("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features")

# set the threshold for the dissimilarity metric
threshold = 0.4
#  load feature counts using sift and manual counts
data_cy5 = read.table("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/datain/joined.mytable.table.cy5", quote="\"")
colnames(data_cy5)<-cbind("case","img","sift","manual")

# load her2 score data
her2score = read.csv("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/datain/her2_score.csv", header = F)
colnames(her2score)<-cbind("case","her2")

# load MLPA score data
mlpa.ratio = read.csv("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/datain/mlpa.ratio.csv", header = F)
colnames(mlpa.ratio)<-cbind("case","mlpa")

# load PLA score data
pla.data = read.csv("~/Work/ClassifyImages/bag-of-features/repo_bag_of_features/datain/pla.signal.per.cell.csv", header = F)
colnames(pla.data)<-cbind("case","pla")

aux = merge(her2score, mlpa.ratio, by="case")
patient.data = merge(aux, pla.data, by="case")

# use this metric to evaluate the similarity between SIFT and manual; 
# 0-neighborought is for similar data, 1-neighborought is for dissimilar data
metric = abs(data_cy5$sift-data_cy5$manual)/(data_cy5$sift+data_cy5$manual)
data_cy5 = cbind(data_cy5, metric)

# set the cutoff for dissimilarity btw sift and manual
data = subset(data_cy5,data_cy5$metric<=threshold)
# plot(data$sift,data$manual)

# sum all features in each case and prepare data-set total
features.per.case = aggregate(data, list(data$case), sum)
case.sift.manual = cbind(features.per.case$Group.1, features.per.case$sift, features.per.case$manual)

total <- merge(case.sift.manual, patient.data, by.x = "V1", by.y = "case")
colnames(total)<-cbind("case","sift","manual","her2","mlpa","pla")

# HER2 --------------------------------------------------------------------

library("ROCR")
library("OptimalCutpoints")
target_pred <- total$manual
target_class <- total$her2

pred <- prediction(target_pred, target_class)
perf <- performance(pred,"tpr","fpr")

cutoffs <- data.frame(cut=perf@alpha.values[[1]], fpr=perf@x.values[[1]], 
                      tpr=perf@y.values[[1]])
library("fields")
x2 = cbind(0,1)
x1 = cbind(cutoffs$fpr,cutoffs$tpr)
cutpoint = cutoffs[which.min(rdist(x1, x2)),]

auc <- performance(pred,"auc")
auc <- unlist(slot(auc, "y.values"))
auc

par(mfrow=c(3,1))

plot(perf,col="black",lty=3, lwd=3, print.cutoffs.at=cutpoint$cut,main=cbind("ROC01=",min(rdist(x1, x2))," auc=",auc))
abline(a=0, b= 1)

# MLPA --------------------------------------------------------------------

spearman=cor(cbind(scale(total$manual),scale(total$mlpa)), use="pairwise.complete.obs", method="spearman") 
plot(scale(total$manual),scale(total$mlpa), main=cbind("manual vs mlpa with Spearman=",spearman[1,2]))

# PLA SIGNAL PER CELL ---------------------------------------------------------------------

spearman=cor(cbind(scale(total$manual),scale(total$pla)), use="pairwise.complete.obs", method="spearman") 
plot(scale(total$manual),scale(total$pla), main=cbind("manual vs pla with Spearman=",spearman[1,2]))

cor(cbind(scale(total$pla),scale(total$mlpa)), use="pairwise.complete.obs", method="spearman") 


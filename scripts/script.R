# load the metrics data
data = read.csv(processed-X.csv)
data$X = NULL
# load the engineered variable info
target = read.csv(processed-y.csv, header = FALSE)
# add the engineered variable to the data 
data$eng = target$V2
# to obtain the means for engineered and non-engineered methods for each metric
sapply(split(data, data$eng == 0), function(x) colMeans(x))
# to run wilcoxon tests for each metric comparing engineered and non-engineered methods
sapply(data[,1:27], function(i) wilcox.test(i ~ data$eng))

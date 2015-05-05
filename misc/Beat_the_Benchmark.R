library(e1071)

#read the data
train <- read.csv("train.csv",header=TRUE)
test <- read.csv("test.csv",header=TRUE)

#select relevant columns
train_cols<-train[,c(2:42)]
labels<-as.matrix(train[,43])
testdata<-test[,2:42]

#convert to numeric
train_cols <- data.frame(lapply(train_cols,as.numeric))
testdata<-data.frame(lapply(testdata,as.numeric))

#run supprt vector regression model and predict on test data
fit<- svm(x=as.matrix(train_cols),y=labels,cost=10,scale=TRUE,type="eps-regression")
predictions<-as.data.frame(predict(fit,newdata=testdata))

#create submission file
submit<-as.data.frame(cbind(test[,1],predictions))
colnames(submit)<-c("Id","Prediction")
write.csv(submit,"submission.csv",row.names=FALSE,quote=FALSE)

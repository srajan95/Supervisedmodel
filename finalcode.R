# Load the dataset and information data
solarsystem<-read.csv("//Users//apple//Documents//UCD//sem2//machine learning//Project//data_project_deepsolar.csv",na.strings = "na")

info<-read.csv("//Users//apple//Documents//UCD//sem2//machine learning//Project//data_project_deepsolar_info.csv")




# Data Cleaning, Dividing the data into testing and training data

#the proportion 
prop.table(table(solarsystem$solar_system_count))


#non-numeric columns are made into factor with respective levels
data<-solarsystem
data$solar_system_count <-as.factor(data$solar_system_count)
data$state<-as.factor(data$state)
data$voting_2016_dem_win<-as.factor(data$voting_2016_dem_win)
data$voting_2012_dem_win<-as.factor(data$voting_2012_dem_win)


#Checking for Duplicate variables
dup_data = data[duplicated(data), ]

#Checking for missing values
colSums(is.na(data))


new_data<-data[,-c(1,2)]
new_data<-new_data[,-c(74,77)]


library(caret)
#Correlation matrix is created 
cor_mat<-cor(new_data)
#The variables with highest correlation is mapped by the column number 
corr_variable<-findCorrelation(cor_mat,cutoff = 0.7,)
corr_variable

new_data<-new_data[,-c(45, 76 ,15 ,56 ,53 ,50 ,57 ,51  ,47 , 5, 14 , 1, 34, 73, 72, 37 ,38, 35, 74, 39  ,33,  41 ,77,  2 ,44 , 6 , 4)]


cor_mat<-cor(new_data)
corr_variable<-findCorrelation(cor_mat,cutoff = 0.8)
corr_variable

#Bind the rest of the column to the new_data with 50 variables
temp_data<-new_data
#Scaling is performed since the variation among the variables was comparitively high


temp_data<-scale(temp_data)
#Column with non-numeric data is binded
temp_data<-cbind(temp_data,data[,c(2,76,79,1)])



#Data partition for model building and testing, p=0.75 means 75% of the data is training and rest of it is the testing data


train_data<-createDataPartition(temp_data$solar_system_count,p=.75,list = F)
training<-temp_data[train_data,]
testing<-temp_data[-train_data,]






#Logisitic Regression

lgmmodel<-glm(solar_system_count~.,data = training,family = "binomial")
summary(lgmmodel)
trainpred<-predict(lgmmodel)
lgmpred<-predict(lgmmodel,newdata = testing,type = "response")


#Plot of Fitted Probability Vs Log Odds
par(mfrow=c(1,2))
symb<- c(15,18)
col<- c("red","blue")
#For the training data 
plot(trainpred , jitter(trainpred,amount = 0.1), pch=symb[training$solar_system_count], col= adjustcolor(col[training$solar_system_count],0.7), cex=0.7, xlab = "Log-odds",ylab = "Fitted probabilities")
#for the testing data
plot(lgmpred , jitter(lgmpred,amount = 0.1), pch=symb[testing$solar_system_count], col= adjustcolor(col[testing$solar_system_count],0.7), cex=0.7, xlab = "Log-odds",ylab = "Fitted probabilities")

#Check the accuracy of the training model
tau<- 0.5
p<- fitted(lgmmodel)
predlgm<-ifelse(p> tau,1,0)
tabl<-table(training$solar_system_count,predlgm)
glmaccr<-sum(diag(tabl)/sum(tabl))
#Accuracy of the training model
glmaccr


library(ROCR)
# Plot the ROC Curve for the testing predicted data
predObj <-prediction(lgmpred, testing$solar_system_count)
roc<-performance(predObj,"tpr","fpr")
plot(roc)
abline(0,1,col ="darkorange2",lty =2)
auc<-performance(predObj,"auc")
#Area under the ROC curve 
auc@y.values




# Decision Tree


#load the libraries
library(rpart)
library(partykit)
library(ROCR)

#build the model
dtmodel<-rpart(solar_system_count~.,data=training)
#Summary of the model
summary(dtmodel)
#Predict the response variable for the test data
preddt<-predict(dtmodel,newdata =testing,type = "class")
dttble<-table(preddt,testing$solar_system_count)
#Get the accuracy
dtacrcy<-(sum(diag(dttble)))/sum(dttble)
dtacrcy



# Plot the ROC Curve and area under the curve is calculated
phat<-predict(dtmodel,newdata = testing)
predObj <-prediction(phat[,2], testing$solar_system_count)
roc<-performance(predObj,"tpr","fpr")
plot(roc)
abline(0,1,col ="darkorange2",lty =2)
auc<-performance(predObj,"auc")
auc@y.values




# Random Forest

library(randomForest)


# implement the random forest algorithm oon the training data
fitrf <-randomForest(solar_system_count~.,data =training)
#Important variables are plotted
varImpPlot(fitrf)

#Calculate the accuracy of the training model no the testing data
rdpred <-predict(fitrf,newdata=testing,type="class",importance=TRUE)
rftable<-table(rdpred,testing$solar_system_count)
rfacc<-(sum(diag(rftable)))/sum(rftable)

#Accuracy is 90.58% 


#ROC Curve is Plotted 

colours <- c("#F8766D","#00BA38")
rdpred <-predict(fitrf,newdata=testing,type="prob",importance=TRUE)
classes <- levels(testing$solar_system_count)
# For each class
for (i in 1:2)
{
  # Define which observations belong to class[i]
  true_values <- ifelse(testing[,54]==classes[i],1,0)
  # Assess the performance of classifier for class[i]
  pred <- prediction(rdpred[,i],true_values)
  perf <- performance(pred, "tpr", "fpr")
  if (i==1)
  {
    plot(perf,main="ROC Curve",col=colours[i]) 
  }
  else
  {
    plot(perf,main="ROC Curve",col=colours[i],add=TRUE) 
  }
  # Calculate the AUC and print it to screen
  auc.perf <- performance(pred, measure = "auc")
  print(auc.perf@y.values)
}



# Bagging

#install.packages("adabag")
#load the library
library(adabag)

#Train the model
bagmodel<-bagging(solar_system_count~.,data = training)
#Predict using the model built for the testing data
bagpred<-predict(bagmodel,newdata=testing,type="class")
#Cross table 
bagtb<-table(bagpred$class,testing$solar_system_count)
#Accuracy is checked 
bagacc<-sum(diag(bagtb)/sum(bagtb))
bagacc


#Boosting

#Build the model for the training data
boostmodel<-boosting(solar_system_count~.,data = training,coeflearn ="Breiman",boos =FALSE)
#predict the response variable for the testing data
boostpred<-predict(boostmodel,newdata=testing,type="class")
#Cross table
boosttb<-table(boostpred$class,testing$solar_system_count)
#Accuracy
bagacc<-sum(diag(boosttb)/sum(boosttb))
bagacc
boostpred[c("confusion","error")]
#Accuracy of 88.59%
#error of 0.114
# error is mapped across each trainmodel and test model
eBoostTrain <-errorevol(boostmodel, training)$error
eboosttest<-errorevol(boostmodel,testing)$error

mat <-cbind(eBoostTrain, eboosttest)

cols <-c("deepskyblue4","darkorange3")

matplot(mat,type ="l",lty =rep(2:1,each =2),col =cols,lwd =2,xlab ="Number of trees",ylab ="Classification error")

legend(x =80,y =0.08,cex =0.75,legend =c("Boosting train","Boosting test"),col =cols,lwd =2,bty ="n",lty = 2)

points(apply(mat,2, which.min),apply(mat,2, min),col =cols,pch =rep(c(15,17),each =2),cex =1.5)


# Performance of 5 models

library(nnet)
library(kernlab)
library(rpart)
library(randomForest)
library(adabag)

R<-50
out <-matrix(NA, R,7)

colnames(out) <-c("val_class_tree","val_logistic","val_svm","val_bag","val_randomforest","best","test")

out <-as.data.frame(out)


for (r in 1:R) 
{
  N<- nrow(temp_data)
  train <-sample(1:N,size =0.50*N)
  val <-sample(setdiff(1:N, train),size =0.25*N )
  test <-setdiff(1:N,union(train, val))
  
  #fit the classifiers only to training data
  
  #logistic regression 
  fitlog<- multinom(solar_system_count~.,data = temp_data[train,],trace=FALSE)
  
  #Classification tree
  fitct<-rpart(solar_system_count~.,data = temp_data[train,])
  
  #SVM
  fitsvm<-ksvm(solar_system_count~.,data = temp_data[train,])
  #Bagging
  fitbag<-bagging(solar_system_count~.,data = temp_data[train,])
  
  #random Forest
  
  fitRF<-randomForest(solar_system_count~.,data = temp_data[train,])
  
  
  
  #fit on validation data
  #Classification tree
  predValCt <-predict(fitct,type ="class",newdata =temp_data[val,])
  tabValCt <-table(temp_data$solar_system_count[val], predValCt)
  accCt <-sum(diag(tabValCt))/sum(tabValCt)
  
  #logistic regression
  
  predValLog <-predict(fitlog,type ="class",newdata =temp_data[val,])
  tabValLog <-table(temp_data$solar_system_count[val], predValLog)
  accLog <-sum(diag(tabValLog))/sum(tabValLog)
  
  
  #SVM
  predValSvm <-predict(fitsvm,newdata =temp_data[val,])
  tabValSvm <-table(temp_data$solar_system_count[val], predValSvm)
  accSvm <-sum(diag(tabValSvm))/sum(tabValSvm)
  
  #bagging
  predValBag <-predict(fitbag,newdata =temp_data[val,])
  tabValBag <-table(temp_data$solar_system_count[val], predValBag$class)
  accBag <-sum(diag(tabValBag))/sum(tabValBag)
  
  #RF
  predValRF <-predict(fitRF,newdata =temp_data[val,],type="class",importance=TRUE)
  tabValRF <-table(temp_data$solar_system_count[val], predValRF)
  accRF<-sum(diag(tabValRF))/sum(tabValRF)
  
  #compute accuracy
  acc <-c(class_tree =accCt,logistic =accLog,svm =accSvm,bag=accBag,rf=accRF)
  out[r,1] <-accCt
  out[r,2] <-accLog
  out[r,3] <-accSvm
  out[r,4]<-accBag
  out[r,5]<-accRF
  
  best <-names(which.max(acc) )
  
  
  switch (best,
          class_tree ={
            predTestCt <-predict(fitct,type ="class",newdata=temp_data[test,])
            tabTestCt <-table(temp_data$solar_system_count[test], predTestCt)
            accBest <-sum(diag(tabTestCt))/sum(tabTestCt)
          },
          logistic =
            {
              predTestLog <-predict(fitlog,type ="class",newdata =temp_data[test,])
              tabTestLog <-table(temp_data$solar_system_count[test], predTestLog)
              accBest <-sum(diag(tabTestLog))/sum(tabTestLog)
            },
          svm ={
            predTestSvm <-predict(fitsvm,newdata =temp_data[test,])
            tabTestSvm <-table(temp_data$solar_system_count[test], predTestSvm)
            accBest <-sum(diag(tabTestSvm))/sum(tabTestSvm)
          },
          bag ={
            predTestBag <-predict(fitbag,newdata =temp_data[test,])
            tabTestBag <-table(temp_data$solar_system_count[test], predTestBag$class)
            accBest <-sum(diag(tabTestBag))/sum(tabTestBag)
          },
          rf ={
            predTestRF <-predict(fitRF,newdata =temp_data[test,],type="class",importance=TRUE)
            tabTestRF <-table(temp_data$solar_system_count[test], predTestRF)
            accBest <-sum(diag(tabTestRF))/sum(tabTestRF)
          }
          
  )
  
  out[r,6] <-best
  out[r,7] <-accBest
  
}
table(out[,6])/R


tapply(out[,7], out[,6], summary)
#Box plot
boxplot(out$test~out$best)
stripchart(out$test~out$best,add =TRUE,vertical =TRUE,method ="jitter",pch =19,col =adjustcolor("magenta3",0.2))

summary(out[,4])













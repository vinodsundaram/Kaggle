
setwd("C:/Users/Vinod/Documents/Python Scripts/GiveMeSomeCredit/")
xtrain<-read.csv("xtrain.csv")
ytrain<-read.csv("ytrain.csv")
xtrain<-xtrain[,-1]
ytrain<-ytrain[,-1]

tail(ytrain)

## Validation set
xvalid = read.csv("xvalid.csv")
yvalid = read.csv("yvalid.csv")
xvalid<-xvalid[,-1]
yvalid<-yvalid[,-1]

## kaggle set:
oxtest<- read.csv("oxtest.csv")
oxtest <- oxtest[,-11]
?xgboost

xtrain<-as.matrix(xtrain)

accuracy_roc_plot<- function(model, xtrain, ytrain, xtest, ytest){
  pred<- predict(model, as.matrix(xtest))
  err <- mean(as.numeric(pred > 0.5) != ytest)
  print(paste("test-error=", err))
  
}

## Plain xGBoost
grid.params <- list(
  objective = "binary:logistic",
  max.depth=3,
  eta=0.01,
  eval_metric="auc"
)

nr = c(2,5,10,15,20)

for(i in nr){
  mod <- xgboost(data=as.matrix(xtrain), label=ytrain,
                 nrounds = 5,objective = "binary:logistic",
                 params = grid.params)
  accuracy_roc_plot(mod,xtrain,ytrain,xvalid,yvalid) 
}
## test error =  0.06576 for nrounds = 20 AUC : .831498
## test error =  0.06576 for nrounds = 15 AUC : .831048
## test error =  0.0656 for nrounds = 10 AUC : .8249
## test error =  0.0666 for nrounds = 5 ACU : .822
## test error =  0.0666 for nrounds = 5 ACU : .8099

xgb.importance(model=mod)
xgb.plot.importance(xgb.importance(model=mod))

#pred<- predict(mod, as.matrix(oxtest))
#prediction <- as.numeric(pred > 0.5)
#summary(pred)

#err <- mean(as.numeric(pred > 0.5) != yvalid)
#print(paste("test-error=", err))


## Now looking at cross validation
?xgb.cv ## help documentation to check params
cv_xgb <- xgb.cv(data=as.matrix(xtrain), label=ytrain,
               nrounds = 100,
               params = grid.params,
               nfold = 5, ##  K fold cross validation
               stratified = TRUE, ## imbalanced dataset
              prediction = TRUE, metrics=list("rmse","auc")
              )

cv_xgb_best <- xgboost(data=as.matrix(xtrain), label=ytrain,
                       nrounds = 500,objective = "binary:logistic",
                       params = grid.params)

feat_col<- colnames(xtrain)
xgb.importance(feat_col, model=mod)
xgb.plot.importance(xgb.importance(feat_col,model=mod))

pred<- predict(cv_xgb_best, as.matrix(xvalid))
err <- mean(as.numeric(pred > 0.5) != yvalid)
print(paste("test-error=", err)) ## 93% accuracy ~ base one

pred<- predict(cv_xgb_best, as.matrix(oxtest))
names(pred)<-"Probability"
#write.csv(pred,"xgbkaggle.csv")
write.csv(pred,"xgbkagglecv.csv")


## hyper parameterized tuning
?xgb.train()
xgb.grid= expand.grid(
  nrounds =c(5),
  eta=c(1,0.1,0.01,0.001),
  max_depth = c(2,3)
)

xgb.grid= expand.grid(
  nrounds = 1000,
  eta = c(0.01, 0.001, 0.0001),
  max_depth = c(2, 4, 6, 8, 10),
  gamma = 1
)

?trainControl
xgb.control = trainControl(
    method="cv",
    number=5 ## number of folds
)

?train() ## the function used for tuning. check params
xgb.tuned = train(
  x=as.matrix(xtrain),y=as.factor(ytrain),
  trControl = xgb.control,
  tuneGrid  = xgb.grid,
  method = "xgbLinear"
)

##  Titanic dataset

##  Loading the dataset
setwd("E:/NUS/Kaggle/Titanic")
train<- read.csv("train.csv", header = TRUE, stringsAsFactors=FALSE)
test  <- read.csv("test.csv",  stringsAsFactors=FALSE)

head(train)
summary(train)
library(ggplot2)

##  Setting the factors
train$Survived<-factor(train$Survived, levels=c(1,0))
levels(train$Survived) <- c("Survived","Died")
train$Pclass<-as.factor(train$Pclass)
levels(train$Pclass) <- c("1st Class", "2nd Class", "3rd Class")
train$Sex <-factor(train$Sex)

####  Overall data discovery  ####

##  Total survived and died
prop.table(table(train$Survived)) ##  38% survived the disaster

##  People travelled vs Class
plot(train$Pclass, ylab = "No of people travelled")

##  Survival across different sex
prop.table(table(train$Sex, train$Survived))    ## gives on the overall split
prop.table(table(train$Sex, train$Survived),1) ## columnwise split
##  74% females survive

##  1. Status wrt pclass
plot(train$Pclass,train$Survived , xlab="CLASS", color=TRUE)
mosaicplot(train$Pclass ~ train$Survived,color=c("#8dd3c7", "#fb8072"), xlab = "Class", ylab="Status", main="Class vs Survival status", cex = 0.8)
## Insight : More % 1st class people survived than 3rd class people

##  2. Status wrt Gender
##  More %Female Survived than Male 
mosaicplot(train$Sex ~ train$Survived, main="Passenger Survival by Gender",
           color=c("#8dd3c7", "#fb8072"), shade=FALSE,  xlab="", ylab="Status",
           off=c(0), cex=.8)

median(train$Age,na.rm = TRUE)

## Children
train$Child <- 0
train$Child[train$Age < 18] <- 1

aggregate(Survived ~ Child+Sex, data=train, FUN=sum)
aggregate(Survived ~ Child+Sex, data=train, FUN=length)
aggregate(Survived ~ Child + Sex, data=train, FUN=function(x) {sum(x)/length(x)})



####  Data Manipulation ####
str(train)
prop.table(table(train$Sex, train$Survived),1)
## Females have higher chance of survival than males. Just the base model set all females as survived

test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
## we get an accuracy of 76.5% with the test data <- got after submission on kaggle

summary(train$Age)
## May be children had a higher chance of survival

summary(train$Fare)
## Creating a new dummy variable for fare
train$Fare[is.na(train$Fare)] = median(train$Fare, na.rm = FALSE)
train$Fare2 <- '30+'
train$Fare2[train$Fare  >= 20 & train$Fare <30 ] <- '20-30'
train$Fare2[train$Fare  >= 10 & train$Fare <20 ] <- '10-20'
train$Fare2[train$Fare <10 ] <- '<10'
tail(train$Fare2)

aggregate(Survived ~ Fare2+ Pclass + Sex, data=train, FUN=function(x) {sum(x)/length(x)})
##  Fare is closely related to the class

test$Survived <- 0
test$Survived[test$Sex == 'female'] <- 1
test$Survived[test$Sex == 'female' & test$Pclass == 3 & test$Fare >= 20] <- 0
## 77.9%

?rpart
?anova


library("tree")
train$Age[is.na(train$Age)] <- median(train$Age, na.rm=TRUE)
train$Fare[is.na(train$Fare)] <- median(train$Fare, na.rm=TRUE)
train$Embarked[train$Embarked==""] = "S" ##Mode(tbl$Embarked[tbl$Embarked!=""])
train$Sex      <- as.factor(train$Sex)
train$Embarked <- as.factor(train$Embarked)

## a simple decision tree 
Ttree <- tree(Survived~Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data=train, method="class")
summary(Ttree)
plot(Ttree)
text(Ttree)

## cross validated with the dataset
Ttree.cv<- cv.tree(Ttree)
plot(Ttree.cv, type="b")

# prune the tree
Ttree.pruned <- prune.tree(Ttree, best=5)
plot(Ttree.pruned)
text(Ttree.pruned)

## Data Manipulation : Creating a new variable called family size
## Can we use any information that has not been significantly used previously.
train$FamilySize <- train$SibSp + train$Parch + 1
test$FamilySize <- test$SibSp + test$Parch + 1


test$Age[is.na(test$Age)] <- median(test$Age, na.rm=TRUE)
test$Fare[is.na(test$Fare)] <- median(test$Fare, na.rm=TRUE)
test$Embarked[test$Embarked==""] = "S" ##Mode(tbl$Embarked[tbl$Embarked!=""])
test$Sex      <- as.factor(test$Sex)
test$Embarked <- as.factor(test$Embarked)


library(ggplot2)
library(randomForest)

## Applying Random Forest 
rf <- randomForest(as.factor(Survived) ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked + FamilySize ,data=train, ntree=2000, importance=TRUE, mtry=2)
##  Decreasing order of the importance of variables
varImpPlot(rf)

##  Use RF to predict on the test data and submit on kaggle
prediction <-predict(rf,test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = prediction)
write.csv(submit, file = "firstforest.csv", row.names = FALSE)


head(train,10)
## How much can a title impact the result - Trevor Stephens (https://github.com/trevorstephens/titanic/)
## the idea here is to use all information that are not being used previously
train$Name <- as.character(train$Name)
train$Title <- sapply(train$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
train$Title <- sub(' ', '', train$Title)
table(train$Title)

test$Name <- as.character(test$Name)
test$Title <- sapply(test$Name, FUN=function(x) {strsplit(x, split='[,.]')[[1]][2]})
test$Title <- sub(' ', '', test$Title)
table(test$Title)


# train small title groups
train$Title[train$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
train$Title[train$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
train$Title[train$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
# Convert to a factor
train$Title <- factor(train$Title)

test$Title[test$Title %in% c('Mme', 'Mlle')] <- 'Mlle'
test$Title[test$Title %in% c('Capt', 'Don', 'Major', 'Sir')] <- 'Sir'
test$Title[test$Title %in% c('Dona', 'Lady', 'the Countess', 'Jonkheer')] <- 'Lady'
# Convert to a factor
test$Title <- factor(test$Title)



library(rpart)
fit <- rpart(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked  + FamilySize + Title,
             data=train, method="class")

prediction <-predict(fit,test)
submit <- data.frame(PassengerId = test$PassengerId, Survived = prediction)
write.csv(submit, file = "rpart.csv", row.names = FALSE)

## This gives the highest accuracy on all the different models that are being built

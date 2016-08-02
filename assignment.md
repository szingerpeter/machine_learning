# Predicting activity type


### Data loading, processing and explaration
Credit goes to: http://groupware.les.inf.puc-rio.br/har for the data.

First we load the necessary libraries.


```
## Warning: package 'ggplot2' was built under R version 3.3.1
```

```
## Warning: package 'dplyr' was built under R version 3.3.1
```

```
## Warning: package 'caret' was built under R version 3.3.1
```

```
## Warning: package 'randomForest' was built under R version 3.3.1
```

Then, we download and load the data into RStudio and check the date.


```r
url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv"
download.file(url, destfile = "training.csv")
url = "https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv"
download.file(url, destfile = "test.csv")
training = read.csv("training.csv")
test = read.csv("test.csv")
Sys.Date() #2016-08-01
```

Let us see the dimensions of the training set and the first 5 observations.


```r
dim(training) ## [1] 19622   160
head(training,5)
```

This dataset has a several possible predictors. Eventually 160 variables seems a bit too much, so we need to be careful to avoid overfitting our training set.

Let us see the proportion of NA values for each column, so hopefully we can get rid of several variables.


```r
prop = sapply(training, function(x) sum(is.na(x))/length(x))
prop[order(prop, decreasing = TRUE)[1:3]]
```

```
##  max_roll_belt max_picth_belt  min_roll_belt 
##      0.9793089      0.9793089      0.9793089
```

As you can see for specific variables, there is a really huge number of missing values (sometimes even over 90% of the observations are missing), so we will exclude variables, which have a higher proportion of NA values than 40%. Then we look at the dimensions of the dataset and the first 5 observations.


```r
training = training[,-which(prop>0.4)]
dim(training) ## [1] 19622    93
head(training, 5)
```

You can see that there are columns, where even though there are no NA values, there is no observation value either.

Let us see the proportion of missing values (not NA) for each column.


```r
prop = sapply(training, function(x) sum(x=="")/length(x))
prop[order(prop, decreasing = TRUE)[1:3]]
```

```
##  kurtosis_roll_belt kurtosis_picth_belt   kurtosis_yaw_belt 
##           0.9793089           0.9793089           0.9793089
```

As you can see for specific variables, there is a really huge number of missing values (sometimes even over 90% of the observations are missing), so we will exclude variables, which have a higher proportion of missing values than 40%. Then we look at the dimensions of the dataset and the first 5 observations.


```r
training = training[,-which(prop>0.4)]
dim(training) ## [1] 19622    60
head(training, 5)
```

Looking at the data, we can see that there are several measures, which are used to identify the user such as username, timestamp. Since we are trying to build a model based on quantitative measueres such as total acceleration, we will exclude these non-necessary variables.


```r
training = training %>% dplyr::select(-(1:7))
```

We were able to reduce the number of possible predictors by more than 100, what is a quite good start.

### Analysis

We start our analysis by creating a training and a validation datasets.


```r
set.seed(1)
intrain = createDataPartition(y=training$classe, p=3/4, list=FALSE)
train_set <- training[intrain,]
validation_set <- training[-intrain,]
```

First we are developing a model via the Random forest method and take a look at the 10 most influential predictors.


```r
rfFit1 = randomForest(classe ~ ., data=train_set )
rfFit1$importance[order(rfFit1$importance, decreasing = T),][1:10]
```

```
##         roll_belt          yaw_belt     pitch_forearm magnet_dumbbell_z 
##          942.4495          670.3213          581.0855          558.8471 
##        pitch_belt magnet_dumbbell_y      roll_forearm magnet_dumbbell_x 
##          536.2156          490.8956          466.0280          370.4816 
##     roll_dumbbell      accel_belt_z 
##          315.3480          302.1466
```

Let us see how good our model works on the validation set.


```r
predicted = predict(rfFit1, validation_set)
confusionMatrix(predicted, validation_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1394    3    0    0    0
##          B    1  945    2    0    0
##          C    0    1  851    5    0
##          D    0    0    2  798    1
##          E    0    0    0    1  900
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9967          
##                  95% CI : (0.9947, 0.9981)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9959          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9993   0.9958   0.9953   0.9925   0.9989
## Specificity            0.9991   0.9992   0.9985   0.9993   0.9998
## Pos Pred Value         0.9979   0.9968   0.9930   0.9963   0.9989
## Neg Pred Value         0.9997   0.9990   0.9990   0.9985   0.9998
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2843   0.1927   0.1735   0.1627   0.1835
## Detection Prevalence   0.2849   0.1933   0.1748   0.1633   0.1837
## Balanced Accuracy      0.9992   0.9975   0.9969   0.9959   0.9993
```

```r
rfFit1
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train_set) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 7
## 
##         OOB estimate of  error rate: 0.44%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4180    3    1    0    1 0.001194743
## B    9 2835    4    0    0 0.004564607
## C    0    9 2554    4    0 0.005064277
## D    0    0   26 2384    2 0.011608624
## E    0    0    1    5 2700 0.002217295
```

It looks quite good, however let us try decreasing the number of predictors, taking only those predictors, which importance score is over 300.


```r
colnames = names(rfFit1$importance[rfFit1$importance>300,])
colnames = append(colnames, "classe")
train_set = train_set[,colnames]
rfFit2 = randomForest(classe ~ ., data = train_set)
```

Let us see how good our model is.


```r
predicted = predict(rfFit2, validation_set)
confusionMatrix(predicted, validation_set$classe)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1378   10    2    0    0
##          B   11  926    6    2    1
##          C    5   10  843    5    1
##          D    1    3    4  795    4
##          E    0    0    0    2  895
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9863          
##                  95% CI : (0.9827, 0.9894)
##     No Information Rate : 0.2845          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9827          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9878   0.9758   0.9860   0.9888   0.9933
## Specificity            0.9966   0.9949   0.9948   0.9971   0.9995
## Pos Pred Value         0.9914   0.9789   0.9757   0.9851   0.9978
## Neg Pred Value         0.9952   0.9942   0.9970   0.9978   0.9985
## Prevalence             0.2845   0.1935   0.1743   0.1639   0.1837
## Detection Rate         0.2810   0.1888   0.1719   0.1621   0.1825
## Detection Prevalence   0.2834   0.1929   0.1762   0.1646   0.1829
## Balanced Accuracy      0.9922   0.9854   0.9904   0.9929   0.9964
```

```r
rfFit2
```

```
## 
## Call:
##  randomForest(formula = classe ~ ., data = train_set) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 3
## 
##         OOB estimate of  error rate: 1.32%
## Confusion matrix:
##      A    B    C    D    E class.error
## A 4150   19   13    3    0 0.008363202
## B   28 2773   29   18    0 0.026334270
## C    2   17 2530   18    0 0.014413713
## D    0    1   22 2385    4 0.011194030
## E    0   12    3    5 2686 0.007390983
```

Even though, we reduced the number of predictors significantly, its accuracy did not change, so for its simplicity, we will use it!

### Conclusion

Even though the 2nd model is a bit less accurate than the 1st model, we will prefer using the 2nd one, since it is less complicated and uses less predictors than the 1st one. Moreover, it is more likely we are avoiding the problem of overfitting with reducing the number of predictors.

# Classification 1 Quiz

This quiz is part of Algoritma Academy assessment process. Congratulations on completing the first Classification in Machine Learning course! We will conduct an assessment quiz to test practical classification model techniques you have learned on the course. The quiz is expected to be taken in the classroom, please contact our team of instructors if you missed the chance to take it in class.

To complete this assignment, you are required to build your classification model to classify the characteristics of employees who have resigned and have not. Use Logistic Regression and k-Nearest Neighbor algorithms by following these steps:

# Data Exploration

Let us start by preparing and exploring the data first. In this quiz, you will be using the turnover of employee data (`turnover`). The data is stored as a .csv format in this repository as `turnover_balance.csv` file. Import your data using `read.csv` or `read_csv` and save as `turnover` object. Before building your classification model, you will need to perform an exploratory analysis to understand the data. Glimpse the structure of our `turnover` data! You can choose either `str()` or `glimpse()` function.

```
# your code here
```

Turnover data consists of 10 variables and 7.142 rows. This dataset is a human resource data that shows historical data of employee characteristics who will resign or not. Below is more information about the variable in the dataset:

  - `satisfaction_level`: the level of employee satisfaction working in a company
  - `last_evaluation`: employee satisfaction level at the last evaluation
  - `number_project`: the number of projects the employee has received
  - `average_montly_hours`: average hours worked per month
  - `time_spend_company`: length of time in the company (years)
  - `work_accident`: presence or absence of work accident, 0 = none, 1 = there
  - `promotion_last_5years`: ever got a promotion in the last 5 years, 0 = no, 1 = yes
  - `division`: name of department or division
  - `salary`: income level, divided into low, medium and high
  - `left`: employee history data resigned, 0 = no, 1 = yes
  
In this quiz, we will try to predict whether or not the employee has a resignation tendency using the `left` column as our target variable. Please change the class of `Work_accident`, `left`, and `promotion_last_5years` column to be in factor class as it should be.

```
# your code here
```

For example, as an HR staff, we are instructed to investigate divisions that has a long history of an employee resigning and analyze their average monthly hours. Let's do some aggregation of `average_monthly_hours` for each division. Because you only focused on the employee who left, you should filter the historical data with the condition needed. 

Using **dplyr** functions, you can use `filter()`, then `group_by()` function by `division` and `summarise()` the mean of `average_montly_hours`, then arrange it based on `average_montly_hours` from high to low using `arrange()` function.

As an alternative, if you are more familiar using **base R** code style, you can filter the data using conditional sub setting `data["condition needed",]`, than assign it into `df_left` object. After that, you can aggregate `df_left` based on `division` and `average_montly_hours` column using `aggregate()` function. Don't forget to use `mean` in `FUN` parameter and assign it into `df_agg`. In order to get the ordered mean value from high to low of the `average_montly_hours`, you can use `order()` function in conditional sub setting `data[order(column_name, decreasing = T), ]`.

```
# your code here
```
___
1. Based on the aggregation data that you have analyzed, which division has the highest average of monthly hours?
  - [ ] Marketing division
  - [ ] Technical division
  - [ ] Sales division
  - [ ] Accounting division
___

# Data Preprocessing

After conducting the data exploratory, we will go ahead and perform preprocessing steps before building the classification model. Before we build the model, let us take a look at the proportion of our target variable in the `left` column using `prop.table(table(data))` function.

```
# your code here
```

It seems like our target variable has a balance proportion between both classes. Before we build the model, we should split the dataset into train and test data in order to perform model validation. Split `turnover` dataset into 80% train and 20% test proportion using `sample()` function and use `set.seed()` with the seed 100. Store it as a `train` and `test` object.

> **Notes:** Make sure you use `RNGkind()` before splitting

```
RNGkind(sample.kind = "Rounding")
set.seed(100)
# your code here

```

Let's take a look distribution of proportion in `train` and `test` data using `prop.table(table(data))` function to make sure in train and test data has balance or not distribution of each class target. Please round the proportion using two decimal numbers using the `round()` function.

```
# your code here

```

___
2. Based on the proportions of `train` and `test`, can the distribution of each class be considered balanced? Why do we need to ensure that each class has a balanced proportion especially in the training data set?
  - [ ] No, it is not.
  - [ ] Yes, it is, but it is not necessary to balance the class proportion.  
  - [ ] No, it is not. The distribution of each class needs to be balanced to prevent any misclassified observation.  
  - [ ] Yes, it is. The distribution of each class in training set data needs to be balanced so when doing model fitting, the algorithm can learn the characteristics for each class equally.
___

# Logistic Regression Model Fitting

After we have split our dataset in train and test set, let's try to model our `left` variable using all of the predictor variables to build a logistic regression. Please use the `glm(formula, data, family = "binomial")` to do that and store your model under the `model_logistic` object. Remember, we are not using `turnover` dataset any longer, and we will be using `train` dataset instead.

```
# model_logistic <- glm()
```

Based on the `model_logictic` you have made above, take a look at the summary of your model using `summary()` function.

```
# your code here
```
___
3. Logistic regression is one of interpretable model. We can explain how likely each variable are predicted to the class we observed. Based on the model summary above, what can be interpreted from the `Work_accident1` coeficient?
  - [ ] The probability of an employee that had a work accident not resigning is 0.21.
  - [ ] Employee who had a work accident is about 0.21 more likely to resign than the employee who has not.  
  - [ ] Employee who had a work accident is about 1.57 less likely to resign than the employee who has not.  
___

# K-Nearest Neighbor Model Fitting

Now let's try to explore the classification model using the k-Nearest Neighbor algorithm. In the k-Nearest Neighbor algorithm, we need to perform one more step of data preprocessing. For both our `train` and `test` set, drop the categorical variable from each column except our `left` variable. Separate the predictor and target in-out `train` and `test` set.

```
# predictor variables in `train`
train_x <-

# predictor variables in `test`
test_x <-

# target variable in `train`
train_y <-

# target variable in `test`
test_y <-
```

Recall that the distance calculation for kNN is heavily dependent upon the measurement scale of the input features. If any variable that have high different range of value could potentially cause problems for our classifier, so let's apply normalization to rescale the features to a standard range of values.

To normalize the features in `train_x`, please using `scale()` function. Meanwhile, in testing set data, please normalize each features using the attribute *center* and *scale* of `train_x` set data.

Please look up to the following code as an example to normalize `test_x` data:

```
scale(data_test, center = attr(data_train, "scaled:center"),
scale = attr(data_train, "scaled: scale"))
```

Now it's your turn to try it in the code below:

```
# your code here

# scale train_x data
train_x <- scale()

# scale test_x data
test_x <- scale()
```

After we have done performing data normalizing, we need to find the right **K** to use for our K-NN model. In practice, choosing k depends on the difficulty of the concept to be learned and the
number of records in the training set data.

___
4. The method for getting K value, does not guarantee you to get the best result. But, there is one common practice for determining the number of K. What method can we use to choose the number of k?
  - [ ] square root by number of row 
  - [ ] number of row
  - [ ] use k = 1
___

After answering the questions above, please find the number of k in the following code:

Hint: If you have got a decimal number, do not forget to round it and make sure you end up with an odd number to prevent voting tie break.

```
# your code here

```


Using K value, we have calculated in the section before, try to predict `test_y` using `train_x` dan `train_y` dataset. To make the k-nn model, please use the `knn()` function and store the model under the `model_knn` object.

Next, please look up at the following code:

```
library(class)
model_knn <- knn(train = ______, test = ________, cl = _______, k = _____)
```

___
5. Fill the missing code here based on the picture above and choose the right code for building the knn model!
  - [ ] model_knn <- knn(train = train_y, test = test_y, cl = test_y, k = 75)
  - [ ] model_knn <- knn(train = train_x, test = test_y, cl = test_x, k = 89)
  - [ ] model_knn <- knn(train = train_x, test = test_x, cl = train_y, k = 75)
  - [ ] model_knn <- knn(train = train_x, test = train_y, cl = train_x, k = 89)
___

# Prediction

Now let's get back to our `model_logistic`. In this section, try to predict `test` data using `model_logistic` return the probability value using `predict()` function with `type = "response"` in the parameter function and store it under `prob_value` object.

```
prob_value <-
```

Because the prediction results in the logistic model are probabilities, we have to change them to categorical / class according to the target class we have. Now, given a threshold of 0.45, try to classify whether or not an employee can be predicted to resign. Please use `ifelse()` function and store the prediction result under the `pred_value` object.

```
pred_value <-
```


Based on the prediction value above, try to answer the following question.

___
6. In the prescriptive analytics stage, the prediction results from the model will be considered for business decision making. So, please take your time to check the prediction results. How many predictions do our `model_logistic` generate for each class?
  - [ ] class 0 = 714, class 1 = 715
  - [ ] class 0 = 524, class 1 = 905
  - [ ] class 0 = 590, class 1 = 839
 ___ 

# Model Evaluation

In the previous sections, we have performed a prediction using both Logistic Regression and K-NN algorithm. However, we need to validate whether or not our model did an excellent job of predicting unseen data. In this step, try to make the confusion matrix of model performance in the logistic regression model based on `test` data and `pred_value` and use the positive class is "1".

**Note:** do not forget to do the explicit coercion `as.factor()`.

```
# your code here
```

Make the same confusion matrix for `model_knn` prediction result of `test_y`.

```
# your code here
```

Let's say that we are working as an HR staff in a company and are utilizing this model to predict the probability of an employee resigning. As an HR, we would want to know which employee has a high potential of resigning so that we can take a precautionary approach as soon as possible. Now try to answer the following questions.

___
7. Which one is the right metric for us to evaluate the numbers of resigning employees that we can detect?
  - [ ] Recall
  - [ ] Specificity  
  - [ ] Accuracy  
  - [ ] Precision  

___
8. Using the metrics of your answer in the previous question, which of the two models has a better performance in detecting resigning employees?
  - [ ] Logistic Regression
  - [ ] K-Nearest Neighbor  
  - [ ] Both has more or less similar performance  

___
9.  Now, recall what we have learned the advantage of each model. Which one is more suitable to use if we aimed for model interpretability?
  - [ ] K-NN, because it tends to have a higher performance than logistic regression
  - [ ] Logistic regression, because it has a lower performance than K-nn
  - [ ] Logistic regression, because each coefficient can be transformed into an odds ratio
  - [ ] K-NN, because it results in a better precision score for the positive class
___


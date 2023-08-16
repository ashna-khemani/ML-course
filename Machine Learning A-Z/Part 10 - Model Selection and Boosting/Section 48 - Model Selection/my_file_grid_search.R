# Grid Search - finding optimal hyperparameter values
# Importing the dataset
dataset = read.csv('Social_Network_Ads.csv')
dataset = dataset[3:5]

# Encoding the target feature as factor
dataset$Purchased = factor(dataset$Purchased, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Purchased, SplitRatio = 0.75)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature Scaling
training_set[-3] = scale(training_set[-3])
test_set[-3] = scale(test_set[-3])

# Apply Grid Search - creating new KernelSVM using Caret, with Grid Search
# install.packages('caret')
library(caret)
classifier = train(form=Purchased~., data=training_set, method="svmRadial")
classifier
classifier$bestTune

plot(classifier)

y_pred = predict(classifier, newdata = test_set[-3])
cm = table(test_set[, 3], y_pred)

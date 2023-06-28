# ANN

# Import Dataset
dataset = read.csv('Churn_Modelling.csv')
dataset = dataset[4:14]

# Encode categorical variables as factors
dataset$Geography = as.numeric(factor(dataset$Geography,
                           levels = c('France', 'Spain', 'Germany'),
                           labels = c(1, 2, 3)))
dataset$Gender = as.numeric(factor(dataset$Gender,
                                      levels = c('Female', 'Male'),
                                      labels = c(1, 2)))

# Splitting into Train and Test
install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Exited, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)

# Feature scaling
training_set[-11] = scale(training_set[-11])  #"factored" columns aren't numerical --> exclude them
test_set[-11] = scale(test_set[-11])

# fit ANN to training set
# install.packages('h2o')
library(h2o)
h2o.init(nthreads=-1)
classifier = h2o.deeplearning(y='Exited', training_frame=as.h2o(training_set), activation='Rectifier', hidden=c(6, 6), epochs=100, train_samples_per_iteration = -2)

# Predict test set results
prob_pred = h2o.predict(classifier, newdata=as.h2o(test_set[-11]))
y_pred = ifelse(prob_pred>0.5,
                1,
                0)
y_pred = as.vector(y_pred)

# Evaluate using confusion matrix
cm = table(test_set[, 11], y_pred)

h2o.shutdown()
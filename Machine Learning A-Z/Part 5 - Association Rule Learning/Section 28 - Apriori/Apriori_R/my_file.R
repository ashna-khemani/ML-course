# Apriori

# Data preprocessing - need to get a sparse matrix
# install.packages('arules')
library(arules)
dataset = read.csv('Market_Basket_Optimisation.csv', header=FALSE)
dataset = read.transactions('Market_Basket_Optimisation.csv', sep=",", rm.duplicates = TRUE)
summary(dataset)
itemFrequencyPlot(dataset, topN=10)

# Train Apriori model on dataset
rules = apriori(data=dataset, parameter = list(support= 0.004, confidence=0.2))

# Visualize results
inspect(sort(rules, by='lift')[1:10])
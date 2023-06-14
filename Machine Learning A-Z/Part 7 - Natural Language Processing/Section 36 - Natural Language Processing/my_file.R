# NLP

# importing dataset
dataset_orig = read.delim("Restaurant_Reviews.tsv", quote = '', stringsAsFactors = FALSE)


# Clean text
# install.packages('tm') # type 'no' when prompted
# install.packages('SnowballC') # helps with stopwords
library(tm)
library(SnowballC)
corpus = VCorpus(VectorSource(dataset_orig$Review))
corpus = tm_map(corpus, content_transformer(tolower)) # get all lowercase
corpus = tm_map(corpus, removeNumbers) # remove numbers
corpus = tm_map(corpus, removePunctuation) # remove punctuation
corpus = tm_map(corpus, removeWords, stopwords()) # remove stopwords
corpus = tm_map(corpus, stemDocument) # stem reviews (keep only roots, loved->love)
corpus = tm_map(corpus, stripWhitespace) # remove extra spaces "  "

# Create bag of words model
dtm = DocumentTermMatrix(corpus) # create the sparse matrix
dtm = removeSparseTerms(dtm, 0.999)  # keep top 99.9% frequent words

################ Random forest classifier #################
dataset = as.data.frame(as.matrix(dtm))
dataset$Liked = dataset_orig$Liked

# Encoding the target feature as factor
dataset$Liked = factor(dataset$Liked, levels = c(0, 1))

# Splitting the dataset into the Training set and Test set
# install.packages('caTools')
library(caTools)
set.seed(123)
split = sample.split(dataset$Liked, SplitRatio = 0.8)
training_set = subset(dataset, split == TRUE)
test_set = subset(dataset, split == FALSE)


# Fitting Random Forest Classifier to the Training set
# install.packages("randomForest")
library(randomForest)
classifier = randomForest(x=training_set[-692], 
                          y=training_set$Liked,
                          ntree=10)

# Predicting the Test set results
y_pred = predict(classifier, newdata = test_set[-692])

# Making the Confusion Matrix
cm = table(test_set[, 692], y_pred)




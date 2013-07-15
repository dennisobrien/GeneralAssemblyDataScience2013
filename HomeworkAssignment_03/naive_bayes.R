library("e1071")
library("caret")
library("ggplot2")

data(iris)

naive.bayes.nfold <- function(n)
{
  # use createFolds from the caret library to create n-fold partition of dataset
  n.rows = nrow(iris)
  folds = createFolds(1:n.rows, k=n)
  
  # perform naive bayes n times
  error.rates = data.frame()
  for (fold in folds)
  {
    train.data = iris[fold, ]
    test.data = iris[-fold, ]
    
    classifier = naiveBayes(train.data[, 1:4], train.data[,5])
    predictions = predict(classifier, test.data[,-5])
    table(predictions, test.data[,5], dnn=list('predicted', 'actual'))
    error.rate = sum(predictions != test.data[, 5]) / length(predictions)
    error.rates = rbind(error.rates, error.rate)
  }
  # n-fold generalization error = average over all iterations
  error.rates = cbind(error.rates, 1:nrow(error.rates))
  names(error.rates) = c('error.rate', 'run')
  print(error.rates)
  mean.error = mean(error.rates[, 1])
  print(paste("mean error:", mean.error))
  results.plot = ggplot(error.rates, aes(x=run, y=error.rate)) + 
                    geom_point() + 
                    geom_line() +
                    geom_hline(yintercept=mean.error, color="red") +
                    aes(ymin=0) +
                    ggtitle("Naive Bayes Classification with n-fold validation") +
                    scale_x_discrete()
  print(results.plot)
}



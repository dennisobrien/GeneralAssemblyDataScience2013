#!/usr/bin/Rscript --vanilla

library(class)
library(ggplot2)

knn.nfold = function(n)
{
  data = iris
  labels = data$Species
  data$Species = NULL
  set.seed(1)
  train.pct = 1/n
  n.total = nrow(data)
  n.test = as.integer(n.total / n)
  n.train = n.total - n.test
  print(paste("total data points:", n.total, ", train:", n.train, ", test:", n.test))
  shuffled.index = sample(1:n.total, n.total)
  #shuffled.index = 1:n.total # used to test/debug without shuffling indices
  err.rates = data.frame(run=integer(0), k=integer(0), error=numeric(0))
  max.k = 100
  for (i in 1:n)
  {
    print(paste("run", i))
    # create n-fold partition of the data
    test.index.end = i * n.test
    test.index.start = test.index.end - n.test + 1
    print(paste("test indices (", test.index.start, ",", test.index.end, ")"))
    train.index = shuffled.index
    train.index = train.index[-(test.index.start:test.index.end)]
    #print(train.index)
    train.data = data[train.index,]
    test.data = data[-train.index,]
    train.labels = as.factor(as.matrix(labels)[train.index, ])
    test.labels = as.factor(as.matrix(labels)[-train.index, ])
    #k = 14  # FIXME: determine the optimal k
    k = get.optimal.k(train.data, train.labels, test.data, test.labels, max.k)
    knn.fit = knn(train=train.data,
                    test = test.data,
                    cl = train.labels,
                    k=k)
    cat('\n', 'k = ', k, ', train.pct = ', train.pct, '\n', sep='') # print params
    print(table(test.labels, knn.fit))  # print confusion matrix
    this.err = sum(test.labels != knn.fit) / length(test.labels)  # store generalization error
    err.rates = rbind(err.rates, c(i, k, this.err))  # append error to total results
  }
  names(err.rates) = c('run', 'k', 'error')
  title = paste('knn resultes (train.pct = ', train.pct, ')', sep='')
  results.plot = ggplot(err.rates, aes(x=run, y=error)) + geom_point() + geom_line()
  print(results.plot)
#  results.plot = ggplot(err.rates, aes(x=run, y=error)) + geom_smooth()
#  print(results.plot)
#   results = data.frame(1:n, err.rates)
#   names(results) = c('run', 'k', 'err.rate')
#   results.plot = ggplot(results, aes(x=k, y=err.rate)) + geom_point() + geom_line()
#   print(results.plot)
  
}

# Determine the optimal k value to use for K Nearest Neighbors.
get.optimal.k = function(train.data, train.labels, test.data, test.labels, n.max)
{
  min.error = Inf
  min.indices = vector()
  for (k in 1:n.max)
  {
    knn.fit = knn(train=train.data,
                  test = test.data,
                  cl = train.labels,
                  k=k)
    #print(table(test.labels, knn.fit))  # print confusion matrix
    this.error = sum(test.labels != knn.fit) / length(test.labels)  # store generalization error
    if (this.error < min.error)
    {
      min.error = this.error
      min.indices = c(k)
    }
    else if (this.error == min.error)
    {
      min.indices = append(min.indices, k)
    }
  }
  k.optimal = min.indices[ceiling(length(min.indices)/2)]
  print(sprintf("Minimum error: %f, optimal k: %d, indices: [%s]", min.error, k.optimal, toString(min.indices)))
  return(k.optimal)
}


if (interactive()==FALSE)
{
  knn.nfold(10)
  message("Press Return To Continue")
  invisible(readLines("stdin", n=1))
}


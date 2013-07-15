library("e1071")
data(iris)
classifier = naiveBayes(iris[, 1:4], iris[,5])
table(predict(classifier, iris[,-5]), iris[,5], dnn=list('predicated', 'actual'))

classifier$apriori
classifier$tables$Petal.Length
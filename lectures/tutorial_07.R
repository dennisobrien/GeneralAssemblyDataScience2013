library(stats)
library("ggplot2")
set.seed(1)

d = data.frame()
d = rbind(d, data.frame(x=1 + rnorm(20, 0, 0.1), y=1 + rnorm(20, 0, 0.1), label=as.factor(rep(1, each=20))))
d = rbind(d, data.frame(x=1 + rnorm(20, 0, 0.1), y=3 + rnorm(20, 0, 0.1), label=as.factor(rep(2, each=20))))
d = rbind(d, data.frame(x=3 + rnorm(20, 0, 0.1), y=1 + rnorm(20, 0, 0.1), label=as.factor(rep(3, each=20))))
d = rbind(d, data.frame(x=3 + rnorm(20, 0, 0.1), y=3 + rnorm(20, 0, 0.1), label=as.factor(rep(4, each=20))))

p = ggplot(d, aes(x=x, y=y)) + geom_point(aes(color=label)) + ggtitle('d == easy clusters')
print(p)

result1 = kmeans(d[, 1:2], 4)

d$cluster1 = as.factor(result1$cluster)
p1 = ggplot(d, aes(x=x, y=y)) + geom_point(aes(color=cluster1)) + 
  ggtitle('kmeans result1 -- success!\n(k=4)')
print(p1)

result2 = kmeans(d[, 1:2], 4)
d$cluster2 = as.factor(result2$cluster)
p2 = ggplot(d, aes(x=x, y=y)) + geom_point(aes(color=cluster2)) + 
  ggtitle('kmeans result2 -- success!\n(k=4)')
print(p2)

result3 = kmeans(d[, 1:2], 4, nstart=10)
d$cluster3 = as.factor(result3$cluster)
p3 = ggplot(d, aes(x=x, y=y)) + geom_point(aes(color=cluster3)) + 
  ggtitle('kmeans result3 -- stable convergence\n(k=4, nstart=10)')
print(p3)


# add another variate
d2 = rbind(d[, 1:3], data.frame(x=1000 + rnorm(20, 0, 50), y=1000 + rnorm(20, 0, 50), label=as.factor(rep(5, each=20))))
p4 = ggplot(d2, aes(x=x, y-y)) + geom_point(aes(color=label)) + ggtitle('d2 -- multiple length scales')
print(p4)

result4 = kmeans(d2[, 1:2], 5, nstart=10)
d2$cluster4 = as.factor(result4$cluster)
p5 = ggplot(d2, aes(x=x, y=y)) + geom_point(aes(color=cluster4)) +
  ggtitle('kmeans result4 == trouble\n((k=k, nstart=10)')


data(iris)
iris.result = kmeans(iris[, 1:4], 3)
iris.result$cluster
iris2 = cbind(iris, cluster=as.factor(iris.result$cluster))

library(gridExtra)
p10 = ggplot(iris2, aes(x=Sepal.Width, y=Petal.Width)) +
  geom_point(aes(color=cluster)) + ggtitle('clustering results')
p11 = ggplot(iris, aes(x=Sepal.Width, y=Petal.Width)) + geom_point(aes(color=Species)) + ggtitle('true labels')
p12 = grid.arrange(p10, p11)
#print(p12)




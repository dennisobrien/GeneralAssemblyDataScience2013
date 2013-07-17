library ("ggplot2")

x = read.csv("http://www.ats.ucla.edu/stat/data/binary.csv")

# look at the data
xtabs(~admit + rank, data=x)

# attempt a linear regression model
lin.fit = lm(admit ~ ., data=x)
lin.fit2 = lm(admit ~ 0 + ., data=x)
summary(lin.fit2)

# convert parameters to factors
x$rank = as.factor(x$rank)

logit.fit = glm(admit ~ ., family='binomial', data=x)
summary(logit.fit)

# odds ratios can be found by exponentiating the log-odds ratios
exp(coef(logit.fit))

# transform the data
new.data = with(x, data.frame(gre=mean(gre), gpa=mean(gpa), rank=factor(1:4)))
new.data$rank.prob = predict(logit.fit, newdata=new.data, type='response')

new.data2 = with(x, data.frame(gre=rep(seq(from=200, to=800, length.out=100), 4),
                               gpa=mean(gpa), rank=as.factor(rep(1:4, each=100))))
new.data2$pred = predict(logit.fit, newdata=new.data2, type='response')
ggplot(new.data2, aes(x=gre, y=pred)) + geom_line(aes(color=rank), size=1)

# now vary gpa
new.data3 = with(x, data.frame(gpa=rep(seq(from=0, to=4.0, length.out=100), 4),
                               gre=mean(gre), rank=factor(rep(1:4, each=100))))
ggplot(new.data3, aes(x=gpa, y=pred)) + geom_line(aes(color=rank), size=1)
# http://bit.ly/uclaerrorbars


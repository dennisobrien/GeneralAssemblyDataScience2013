# See communities.py to retrieve the dataset and create a csv with named headers.
# Datafiles from http://archive.ics.uci.edu/ml/datasets/Communities+and+Crime

df = read.csv('communities.csv')
df[df=='?'] = NA

# create a model with one predictor variable
model.1 = lm(ViolentCrimesPerPop ~ PctTeen2Par, data=df)
summary(model.1)
plt.1 = plot(df$PctTeen2Par, df$ViolentCrimesPerPop)
abline(reg=model.1, col="red")
print(plt.1)

# create a model with polynomial factors of PctTeen2Par
df_poly = cbind(df, PctTeen2Par_2=df$PctTeen2Par^2)
df_poly = cbind(df_poly, PctTeen2Par_3=df$PctTeen2Par^3)
model.poly = lm(ViolentCrimesPerPop ~ PctTeen2Par + PctTeen2Par_2 + PctTeen2Par_3, data=df_poly)
summary(model.poly)
plot(df$PctTeen2Par, df$ViolentCrimesPerPop)


# can we add even more?
df_poly = cbind(df_poly, PctTeen2Par_4=df$PctTeen2Par^4)
df_poly = cbind(df_poly, PctTeen2Par_5=df$PctTeen2Par^5)
df_poly = cbind(df_poly, PctTeen2Par_6=df$PctTeen2Par^6)
df_poly = cbind(df_poly, PctTeen2Par_7=df$PctTeen2Par^7)
df_poly = cbind(df_poly, PctTeen2Par_8=df$PctTeen2Par^8)
df_poly = cbind(df_poly, PctTeen2Par_9=df$PctTeen2Par^9)
df_poly = cbind(df_poly, PctTeen2Par_10=df$PctTeen2Par^10)
model.poly_10 = lm(ViolentCrimesPerPop ~ PctTeen2Par + PctTeen2Par_2 + PctTeen2Par_3 +
                     PctTeen2Par_4 + PctTeen2Par_5 + PctTeen2Par_6 + PctTeen2Par_7 +
                     PctTeen2Par_8 + PctTeen2Par_9 + PctTeen2Par_10, data=df_poly)
# it doesn't have much of an effect on R^2
summary(model.poly_10)

# Use lm.ridge to get estimates of the coefficients
# This requires more than one input variable, so including two here.
library(MASS)
model.1.ridge = lm.ridge(ViolentCrimesPerPop ~ PctTeen2Par + PctPopUnderPov, data=df)
coef(model.1.ridge)
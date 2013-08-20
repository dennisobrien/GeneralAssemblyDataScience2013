Final Project Outline
=========

General Assembly Data Science
Dennis O'Brien
<dennis@dennisobrien.net>

* * *

The Problem
-----------

I work for a social game company in the free-to-play space.  For most free-to-play games, the vast majority of players pay nothing to play the game, but some small fraction of players do monetize.  New players to your game are either acquired or organic.

Acquired players are paid for through any number of channels and strategies.  In general, you are able to provide some criteria, along with the bid, for the players you are hoping to acquire.  The metric CPI (cost per install) refers to the average prices per install you a playing for some segment of your players.

Organic players are those players who installed your game without you having to pay.  For example, the player may have been invited by a friend to play, or followed a link from a game review site.  The combination of acquired and organic installs is encapsulated in the term eCPI (effective cost per install).

That is a very simplified version of the user acquisition end of things.  On the other side of the equation is the lifetime value of a player, or LTV.  This is always partly a projection since it is attempting to measure the monetary value you can expect to get from the average player over his or her lifetime playing your game.

The ability to quickly and accurately predict the LTV for a given cohort can help a game to grow effectively by finding the best opportunities to spend the limited marketing budget.

* * *

Description of Dataset
----------------------

The data available is in three broad categories:

* Player gameplay behavior
* Player purchase behavior
* Facebook profile data (for those players who choose to connect to our game via Facebook)

Both player gameplay and purchase behavior is available as granular events with timestamps, amounts, and other pertinent information.  Some subset of available Facebook profile data is stored in our database and the data not stored can always be retrieved from the Facebook Graph API.

We have data for several million players and several billion gameplay events stretching back almost a year.

Since the raw data itself is sensitive, I may only be able to provide the analysis of the data without any of the actual data.  It may be possible to prepare some anonomyzed data for verification, but that is to be determined.

* * *

Hypothesis
----------

I expect that the application of machine learning techniques to the problem of LTV prediction will outperform the current retention-based curve fitting model in use.  Some specific problems to investigate:

* Given the available Facebook profile data, what is the probability that a new player will monetize within some time frame?
* Given the available Facebook profile data, what is the expected revenue for a new player within some time frame?
* Given actual gameplay behavior and spending behavior for a player already playing the game, what is the probability the player will monetize within some timeframe?
* Given actual gameplay behavior and spending behavior for a player already playing the game, what is the expected revenue from the player within some time frame?

* * *

Proposed Methods
-----------------

Predicting whether a player (either prospective or current) will monetize in some time frame, given historical data, can be approached as a classification problem.  Scikit-learn provides a number of classification algorithms that might be suitable for this part of the problem.

* Stochastic Gradient Descent Classifier
* Linear Support Vector Classifier
* Naive Bayes
* Ensemble techniques

For the problem of predicting the revenue expected in some given time frame, we can consider some of the available 

* Stochastic Gradient Descent Regressor
* Ridge Regression
* Lasso

The available historical data will be randomly separated for training and testing, with standard error metrics used to compare the various techniques, and for comparison agains the current process in place here today.

* * *

Business Applications
---------------------

Being able to quickly estimate the LTV of a user acquisition campaign and compare that to the eCPI can allow a company to quickly ramp up or shut down a particular campaign.



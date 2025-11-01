# How to Execute the Code

Clone the repository with git and install "EconML_NumAs" Environment
from environment.yml with conda/mamba to start checking the Code. The
finished repo has all data and save files required for running any
.ipynb notebook or .py script, but the following order is how I ran the
code initially:

1.  Run Preperation.ipynb to create a cleaned dataset for further
    processing.

2.  Run Lasso.ipynb to build and visualize an optimized LASSO-Model and
    save it together with the scaler used for scaling training data.

3.  Run BootstrappedLasso.ipynb to assess LASSO robustness (Boxplots,
    red line indicates Lasso Model coefficients) and generate a stacked
    LASSO model based on bootstrapped samples.

4.  Run PrunedTrees.ipynb to generate a single decision tree with
    optimal pruning and compare impact of tree size on test error.

5.  Run MCMC_Tuning.py from CMD with either RandomForest or XGBoost set
    to true to activate the MCMC process. Either set load to none or
    increase the chain_length when running with XGBoost as true.

6.  Run RandomForest.ipynb to generate a random forest model using grid
    search and visualize the chains from a MCMC algorithm to fit random
    forest parameters.

7.  Run GradientBoosting.ipynb to generate and visualize grid-searching
    for the best XGBoost-regressor model and use the outputs of a full
    MCMC algorithm to fit a stacked XGBoost model with the Top 10% of
    models found in the MCMC proposals. Also visualises the MCMC
    process.

8.  Run Prediction.ipynb to predict the to be submitted unknown sale
    prices based on the stacked XGBoost model.

I use DataWrangler for inspecting data instead of .head() or .info()
commands. This was also used to get quick overviews of generated models,
for example to assess the grid searched XGBoost models.

# Model Selection and Overview

All Models were created and compared using the same 95%-5% training-test
split. Models are evaluated on the exponentiated root mean squared
prediction error (in essence, how much predicted sale price tends to
deviate from the actual sale price) for the test set, furtherly called
validationset-error

## Lasso

I have experience with LASSO models from a recent paper I wrote and
thought it usable in this case. The regularization promised to drop
unimportant variables in a large dataset like the one given, with 260
variables after creating dummies.

For scaling, PowerTransform from sciki-learn was used as it performs
well with strongly skewed data like housing data, where eg. "lot area
"can have significant outliers. "Sale Price", the variable we are
interested in, is also log-transformed for this model and all others to
have less skewed dependant variables.

Overall, LASSO performed well, shrinking the prediction model to use
less than 100 predictors and resulting in a validationset-error of ca.
17743.11\$.

## Bootstrapped Lasso 

Using Bootstrapping, another model based on LASSO models created from
the bootstrapped samples from the training data was constructed. Using
Ridge Regression, all 1000 bootstrapped LASSO models were stacked to
create a model that uses their predictions, weighted to reduce K-Fold
cross-validation error.

The stacked model had a validationset-error of 18429.47\$, worse than
the LASSO model, which is probably either due to bootstrapping only
using about 66% of data to create each model or the Stacking procedure
not being designed correctly (too many models considered, no
differentiation using model coefficients). Sadly, due to time
constraints this couldn't be further analysed.

The coefficient distribution from the bootstrapped models was also used
to assess the robustness of the LASSO model, showing that generally and
for the most influential variables, LASSO is very robust.

## Decision Tree with Pruning

To gain experience with Decision Tree methods, I implemented a simple
Decision Tree with pruning, using scikit-learns pruning path method to
test which tree size is optimal. This remained an experiment, as a
single decision tree produced high validationset-error (26656.88\$),
making me opt for ensemble methods..

## Random Forest

Random Forest is presumed to usually perform very good, but in my case,
it didn't. Iterative Grid searching (only the last search is still in
the code to ensure code-readability and navigation for both you and me)
resulted in models that are only marginally better than a pruned
decision tree (validationset-error of 22627.51\$). Usign MCMC tuning
only resulted in models that were on average worse than XGBoost, which
is why MCMC parameter sampling was stopped prematurely and not fully
analysed.

## XGBoost

XGBoost is generally believed to generate very good models which is why
I used XGBoost instead of simple gradient boosting. Like random forest,
it is an esemble method that requires a lot of tuning. GridSearch was
used but is constrained due to computational restraints not allowing for
exploring the whole parameter range, especially in XGBoost which has
more parameters than Random Forest. Instead, I used an MCMC algorithm to
explore the parameter space and used the best performing 10% of models
to build a stacked model just like with bootstrap, resulting a
validationset-error of 14929.40\$, the lowest by far. The stacked
XGBoost model was chosen for prediction due to its performance.

# Further Explanation

## Data Cleaning

Preperation.ipynb is used for data cleaning and feature engineering. The
bulk of changes are i) making some categorical columns numerical if they
can be interpreted as ordinal, ii) adding features that have high
explanatory value (eg. Has_Basement) but are only latent as combinations
of other features, and iii) summarising variables if possible. It also
masks values where no interpretability is given (and later scaling
shouldn't be applied) with -1.

## MCMC Algorithm

As ensemble methods require tuning a lot of different parameters without
a clear way -- to my knowledge -- of finding the best parameter
combination without using a lot of computational resources, I opted to
employ an Markov-Chain Monte-Carlo Metropolis-Hastings inspired
Algorithm to explore the parameter space and find good combinations,
inspired by a previous class on Bayesian Statistics. Essentially, I used
RandomSearchCV to find a set of good starting parameters (representing
burn-in, where normally good initial values are searched for) and then
slightly changed all parameters over many iterations, accepting a
parameter permutation if it improved root mean squared error or if the
increase was neglibly small, thereby exploring parameter space and
avoiding local minima.

While normally MCMC aims to find stable parameters, this was
unsuccessful in my case due to either bad MCMC tuning (step size and
temperature) or XGBoost and RandomForest allowing for different
parameter combinations that achieve good results. Instead, I used model
stacking on the 10% of best parameter combinations to generate a final
model.

## Model Stacking

Bootstrapping and MCMC both resulted in many models. Instead of just
choosing the best among them based on cross validation or
validationset-error, I used model stacking to create an ensemble of
multiple models. The idea was to let all models make predictions and fit
a ridge model on the predictions and the true data, using
cross-validation to choose the right amount of regularization. This
allows for using multiple best-models to hopefully generate better
predictions, as different models might perform better in different
areas.

# AI Usage

AI was only used as an helper in this project, allowing me to debug code
quickly, help with syntax or creating quick code for experiments and
visualizations.

However, the mathematics of the acceptance function in the MCMC
algorithm was created by AI. The pertubation function which was hard to
pin down was also developed together with AI, but I should mention that
it was little helpful in that regard.

The idea of model stacking was also initially proposed by AI, the actual
implementation was done based on online information on model stacking.

Overall, AI was helpful but not key to this project.

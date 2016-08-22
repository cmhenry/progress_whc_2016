'''
Colin Henry
University of New Mexico, 2016
Model pipeline for UNC Water Institute working paper. Imports custom JMP data set and cycles through available
linear and non-parametric models using pandas, numpy, statsmodel, and scikit. Runs multiple models on broadly 
stratified data set (country-year, country, and country-year-survey).
'''

import sys

import numpy as np 
import pandas as pd
from pandas import Series, DataFrame

import statsmodels.api as sm 
from sklearn.linear_model import LinearRegression
import scipy, scipy.stats

import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
import matplotlib.pyplot as plt

# NOTE: Importing from R imports R-vector objects.
# e.g., 'pi' is actually [1] 3.14... and referencing the value is pi[0]. 
# Moreover, we can run R code via the robjects.r function:
# robjects.r('''
#         # create a function `f`
#         f <- function(r, verbose=FALSE) {
#             if (verbose) {
#                 cat("I am calling f().\n")
#             }
#             2 * pi * r
#         }
#         # call the function `f` with argument value 3
#         f(3)
#         ''')
# This returns 18.85

# Allows printing of entire DataFrame in terminal without breaks.
pd.options.mode.chained_assignment = None

def load_datafile(fname) : 
	'''
	Import Stata file in chunks; provide loading metric; return DataFrame. 
	'''
	import sys

	reader = pd.read_stata(fname, iterator=True)
	df = DataFrame()

	try : 
		chunk = reader.get_chunk(100*100)
		while len(chunk) > 0 : 
			df = df.append(chunk, ignore_index=True)
			chunk = reader.get_chunk(100*100)
			print '.'
			sys.stdout.flush()
	except(StopIteration, KeyboardInterrupt) : 
		pass

	print 'Loaded {} rows.'.format(len(df))

	return df

def ols_reg(df, indicator) : 
	'''
	Run OLS regression on indicator (y), year (x), by Gleditsch-Ward Number of the state.
	Returns fitted model as result.
	'''
	y = df[[indicator]][:]
	x = df.year[:]
	x = sm.add_constant(x)
	model = sm.OLS(y, x, missing='drop')
	result = model.fit()
	print 'Fitting OLS model...'
	return result

def split_sample(df, state) : 
	'''
	Split DataFrame by Gleditsch-Ward Number to fit model to individual cases.
	Returns new, split DataFrame.
	'''
	df = df.loc[df['gwno'] == state]
	# print 'Splitting sample by GWNO {}...'.format(state)
	return df

# def gam(df, indicator) : 
# 	'''
# 	Calls R-script for Generalized Additive Model. 
# 	'''
# 	print 'Checking indicator quality.'
# 	if df[indicator].count() < 3 : 
# 		print 'Indicator has {} observations. Cannot run GAM.'.format(df[indicator].count())
# 	else : 
# 		gam = robjects.r('''
# 			# GAM Modeling with mgcv
# 			library(mgcv)
# 			newd <- data.frame(years)
# 			# Load data here
# 			gam <- gam(indicator ~ s(years, k=4), family=guassian)
# 			gam.pred <- predict.gam(gam, newd)
# 			gam.pred[newd$years<(min(years)-2)] <- gam.pred[newd$years==(min(years)-2)]
# 			gam.pred[newd$years>(max(years)+2)] <- gam.pred[newd$years==(max(years)+2)]
# 			gam.pred[gam.pred>100] <- 100
# 			gam.pred
# 			''')
# 	print 'GAM modeling completed.'

def gam_model(df, indicator) : 
	'''
	Re-constructs R-script for Generalized Additive Model.
	'''
	from rpy2.robjects import numpy2ri
	import rpy2.robjects as robjects
	from rpy2.robjects.packages import importr
	from rpy2.robjects import FloatVector, Formula
	r = robjects.r
	base = importr('base')
	utils = importr('utils')
	numpy2ri.activate()

	mgcv = importr('mgcv')
	if df[indicator].count() < 3 : 
		print '{} observations. Cannot run GAM.'.format(df[indicator].count())
	else : 
		y = np.array(df[indicator])
		x = np.array(df['year'])
		newd = robjects.DataFrame({'year' : robjects.FloatVector(y)})
		gam_formula = Formula('y ~ s(x, k=4)')

		env = gam_formula.environment
		env['x'] = x
		env['y'] = y

		model = mgcv.gam(gam_formula, 'family=gaussian')
		model_pred = r.predict(model, newd)
		print 'GAM modeling completed.'
		return model_pred, model

'''
Begin pipeline. 
'''
# Load datafile.
df = load_datafile("combined_data.dta")
# Iterate through unique GWNO.
for no in df.gwno.unique() : 
	split_df = split_sample(df, no)
	model, pred = gam_model(split_df, 'rtotal1')





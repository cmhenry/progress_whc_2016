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
from scipy.interpolate import interp1d

import pyqt_fit.nonparam_regression as smooth
from pyqt_fit import npr_methods
import matplotlib.pyplot as plt

from rpy2.robjects import numpy2ri
import rpy2.robjects as robjects
from rpy2.robjects.packages import importr
from rpy2.robjects import FloatVector, Formula

r = robjects.r
base = importr('base')
utils = importr('utils')
numpy2ri.activate()
mgcv = importr('mgcv')
locpol = importr('locpol')

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

DEBUGGING = False
if DEBUGGING : 
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
	if DEBUGGING : 
		print 'Vars imported: ', list(df.columns.values)

	return df

def split_sample(df, state) : 
	'''
	Split DataFrame by Gleditsch-Ward Number to fit model to individual cases.
	Returns new, split DataFrame.
	'''
	df = df.loc[df['gwno'] == state]
	# print 'Splitting sample by GWNO {}...'.format(state)
	if DEBUGGING : 
		print list(df.columns.values)
	return df

def ols_reg_model(df, indicator) : 
	'''
	Run OLS regression on indicator (y), year (x), by Gleditsch-Ward Number of the state.
	Returns predicted values as a result.
	'''
	predicted_values = pd.Series(index = df.index)
	model_fit_stat = pd.Series(index = df.index)
	quad_term_sig = pd.Series(index = df.index)
	model_class = pd.Series(index = df.index)

	for num in df.gwno.unique() : 
		split_df = split_sample(df, num)

		if split_df[indicator].count() < 2 : 
			if DEBUGGING : 
				print '{} observations. Cannot run OLS.'.format(split_df[indicator].count())
		else : 
			y = split_df[[indicator]][:]
			x = split_df.year[:]
			x_cons = sm.add_constant(x)
			model = sm.OLS(y, x_cons, missing='drop').fit()

			model_pred = model.predict(x_cons)
			model_pred = pd.Series(model_pred, index = split_df.index)
			model_pred = pd.to_numeric(model_pred, errors='coerce')
			predicted_values.update(model_pred)

			if model.aic : 
				model_fit_stat.update(pd.Series(model.aic, index = split_df.index))

			'''
			Model with quadratic term. Get significance of quadratic term and store if significant alpha = 0.05.
			'''
			quad_model = sm.OLS.from_formula(formula = 'y ~ np.power(x, 2) + x_cons', data=x_cons, missing='drop').fit()
			if quad_model.pvalues[1] and quad_model.pvalues[1] <= 0.05 :
				temp = pd.Series(1, index = split_df.index)
				temp = pd.to_numeric(temp, errors='coerce')
				quad_term_sig.update(temp)
			else : 
				temp = pd.Series(0, index = split_df.index)
				temp = pd.to_numeric(temp, errors='coerce')
				quad_term_sig.update(temp)

			'''
			Classify case by quadratic term significance, quadratic term slope, and linear term slope.
			(1) Linear growth - If the p-value for b4 >= 0.05 and the slope b1 > 0.05 per year
			(2) Linear decline - If the p-value for b4 >= 0.05 and the slope b1 < -0.05 per year
			(0) No change - If the p-value for b4 >= 0.05 the slope b1 is between -0.05 and 0.05 per year
			(3) Saturation - if the p-value for b4 < 0.05 and the slope b4 < 0 and the slope b1 > 0 and the maximum value of any data point is >= 95%
			(4) Acceleration - if the p-value for b4 < 0.05 and the slope b4 > 0 and the slope b1 > 0
			(5) Deceleration - if the p-value for b4 < 0.05 and the slope b4 < 0 and the slope b1 > 0 and the maximum value of all data points is < 95%
			(6) Negative acceleration - if the p-value for b4 < 0.05 and the slope b4 < 0 and the slope b1 < 0
			(7) Negative deceleration - if the p-value for b4 < 0.05 and the slope b4 > 0 and the slope b1 < 0

			model.params[1] is slope b1 (from simple OLS model)
			quad_model.params[1] is slope b4 (from quadratic OLS model)
			quad_model.pvalues[1] is p-value for b4 (from quadratic OLS model)
			'''
			# Classify linear / no change
			if quad_model.pvalues[1] and quad_model.pvalues[1] >= 0.05 : 
				if model.params[1] > 0.05 : 
					temp = pd.Series(1, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
				if model.params[1] < -0.05 : 
					temp = pd.Series(2, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
				if model.params[1] >= -0.05 and model.params[1] <= 0.05 :
					temp = pd.Series(0, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
			else :
				# Classify acceleration
				if quad_model.params[1] > 0 and model.params[1] > 0 : 
					temp = pd.Series(4, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
				# Classify deceleration
				if quad_model.params[1] > 0 and model.params[1] < 0 and y[indicator].max() > 95 : 
					temp = pd.Series(5, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
				# Classify saturation
				if quad_model.params[1] < 0 and model.params[1] > 0 and y[indicator].max() >= 95 : 
					temp = pd.Series(5, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
				# Classify negative acceleration
				if quad_model.params[1] < 0 and model.params[1] < 0 : 
					temp = pd.Series(6, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)
				# Classify negative deceleration
				if quad_model.params[1] > 0 and model.params[1] < 0 : 
					temp = pd.Series(7, index = split_df.index)
					temp = pd.to_numeric(temp, errors='coerce')
					model_class.update(temp)

	return predicted_values, quad_term_sig, model_fit_stat, model_class

def loess_model(df, indicator) : 
	'''
	Run loess smoothing function on indicator (y), year (x), by Gleditsch-Ward
	number of the state. Returns predicted values as a result.
	'''
	predicted_values = pd.Series(index = df.index)
	model_fit_stat = pd.Series(index = df.index)
	lowess = sm.nonparametric.lowess

	for num in df.gwno.unique() : 
		split_df = split_sample(df, num)
		if split_df[indicator].count() < 5 : 
			if DEBUGGING : 
				print '{} observations. Cannot run GAM.'.format(split_df[indicator].count())
		else : 
			y = np.array(split_df[indicator])
			x = np.array(split_df['year'])

			model = lowess(y, x)

			lowess_x = list(zip(*model))[0]
			lowess_y = list(zip(*model))[1]
			func = interp1d(lowess_x, lowess_y, bounds_error=False)

			model_pred = func(x)
			model_pred = pd.Series(model_pred, index = split_df.index)
			model_pred = pd.to_numeric(model_pred, errors='coerce')
			predicted_values.update(model_pred)

			# k = len(model_pred)
			# loglik = -np.sum(scipy.stats.norm.logpdf({lowess_y,lowess_x}, loc=model_pred, scale=sd)) 
			# aic = 2*k - 2*(loglik)
			# print aic

	return predicted_values

def locpol_model(df, indicator) : 
	'''
	Re-constructs R-script for local polynomial regression.
	'''
	out_column = pd.Series(index = df.index)

	for num in df.gwno.unique() : 
		split_df = split_sample(df, num)
		if split_df[indicator].count() < 2 : 
			if DEBUGGING : 
				print '{} observations. Cannot run LocPol.'.format(split_df[indicator].count())
		else : 
			y = np.array(split_df[indicator])
			x = np.array(split_df['year'])
			newd = robjects.DataFrame({'year' : robjects.FloatVector(y)})
			data = robjects.DataFrame({'x' : robjects.FloatVector(x), 'year' : robjects.FloatVector(y)})
			locpol_form = Formula('y ~ x')

			env = locpol_form.environment
			env['x'] = x
			env['y'] = y
			env['data'] = data

			model = locpol.locpol(locpol_form, 'set.NA=TRUE')
			print model
			# model_pred = r.predict(model, newd)
			
			# model_pred = pd.Series(model_pred, index = split_df.index)
			# model_pred = pd.to_numeric(model_pred, errors='coerce')
			
			# out_column.update(model_pred)
	
	return out_column

def gam_model(df, indicator) : 
	'''
	Re-constructs R-script for Generalized Additive Model.
	GAM model object returns attributes: 
	 [1] "coefficients"      "residuals"         "fitted.values"    
 	 [4] "family"            "linear.predictors" "deviance"         
 	 [7] "null.deviance"     "iter"              "weights"          
	[10] "prior.weights"     "df.null"           "y"                
	[13] "converged"         "sig2"              "edf"              
	[16] "edf1"              "hat"               "R"                
	[19] "boundary"          "sp"                "nsdf"             
	[22] "Ve"                "Vp"                "rV"               
	[25] "mgcv.conv"         "gcv.ubre"          "aic"              
	[28] "rank"              "gcv.ubre.dev"      "scale.estimated"  
	[31] "method"            "smooth"            "formula"          
	[34] "var.summary"       "cmX"               "model"            
	[37] "na.action"         "control"           "terms"            
	[40] "pred.formula"      "pterms"            "assign"           
	[43] "offset"            "df.residual"       "min.edf"          
	[46] "optimizer"         "call"
	'''
	# r = robjects.r
	# base = importr('base')
	# utils = importr('utils')
	# numpy2ri.activate()
	# mgcv = importr('mgcv')

	predicted_values = pd.Series(index = df.index)
	model_fit_stat = pd.Series(index = df.index)

	for num in df.gwno.unique() : 
		split_df = split_sample(df, num)
		if split_df[indicator].count() < 5 : 
			if DEBUGGING : 
				print '{} observations. Cannot run GAM.'.format(split_df[indicator].count())
		else : 
			y = np.array(split_df[indicator])
			x = np.array(split_df['year'])
			newd = robjects.DataFrame({'year' : robjects.FloatVector(y)})
			gam_formula = Formula('y ~ s(x, k=4)')

			env = gam_formula.environment
			env['x'] = x
			env['y'] = y

			model = mgcv.gam(gam_formula, 'family=gaussian')
			model_pred = r.predict(model, newd)
			model_pred = pd.Series(model_pred, index = split_df.index)
			model_pred = pd.to_numeric(model_pred, errors='coerce')
			predicted_values.update(model_pred)

			model_aic = pd.Series(model[26], index = split_df.index)
			model_aic = pd.to_numeric(model_aic, errors = 'coerce')
			model_fit_stat.update(model_aic)
	
	return predicted_values, model_fit_stat

'''
Begin pipeline. 
'''
# Load datafile.
df = load_datafile("combined_data.dta")
df.sort_values(['gwno', 'year'], ascending=[False, True], inplace=True)

ols, quad_sig, ols_fit, model_class = ols_reg_model(df, 'rtotal1')
gam, gam_fit = gam_model(df, 'rtotal1')
loess = loess_model(df, 'rtotal1')
# locpol = locpol_model(df, 'rtotal1')

df.loc[:,'pred_rtotal1_gam'] = gam
df.loc[:,'pred_rtotal1_ols'] = ols
df.loc[:,'pred_rtotal1_loess'] = loess

df.loc[:,'rtotal1_quad_sig'] = quad_sig
df.loc[:,'rtotal1_ols_fit'] = ols_fit
df.loc[:,'rtotal1_gam_fit'] = gam_fit

df.loc[:,'model_class'] = model_class

if '-e' in sys.argv : 
	try : 
		# filepath = sys.argv[sys.argv.index('-e')+1]
		df.to_csv("dataframe.csv")
		print 'Exported {} rows.'.format(len(df))
	except IndexError : 
		print "Error: must specify export path."

'''
Future note: preferred method of split-apply-combine data management would be to use
the built in groupby function to apply multiple estimation techniques to multiple
groups concurrently. However, because we are wrapping R in python and returning R code
objects, the groupby function behaves badly. Or possibly I just don't understand groupby.
Further investigation necessary. Sample groupby command included below.
'''
# df.groupby('gwno').apply(gam_model, 'rtotal1')






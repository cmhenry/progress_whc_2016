/*********
Analysis for WHC conference paper
Colin Henry
University of New Mexico
June 2016
**********/

// ID for Time Series
replace code = "NA" if mi(code)
gen temp1 = string(gwno)+"-"+code
encode temp1 , gen(tsset_id)
drop temp1
tsset tsset_id year

// LPoly fitting (Colin measure)
do "Henry_Lpoly.do"
gen lp_agree = 1 if i_rtotal1 > se_minus & i_rtotal1 < se_plus
replace lp_agree = 0 if mi(lp_agree) & !mi(code)

// Linear regression fitting (JMP measure)
quietly bysort gwno : reg i_rtotal1 year

/* Generate examples
// ECUADOR
// L Poly
lpoly i_rtotal1 year if gwno==130 , mlabel(code) ci degree(3)
// Linearized
#delimit ;
twoway  (lfitci i_rtotal1 year if gwno==130 & year > 1988 & year < 2012) 
(scatter i_rtotal1 year if gwno==130 & year > 1988 & year < 2012 , mlabel(code));
#delimit cr
// Lowess
lowess i_rtotal1   year if gwno==130 & year > 1994 & year < 2015 , mlabel(code)
// Cubic spline
#delimit ;
twoway mspline i_rtotal1 year if gwno==130 & year > 1992 & year < 2012 , bands(5)
|| scatter i_rtotal1 year if gwno==130 & year > 1994 & year < 2012 , mlabel(code);
#delimit cr */



# Replicating GAM (Fuller et. al. 2016) for JMP water coverage data set.

library(mgcv)
library(plyr)

# Getting command line arguments.
args <- commandArgs(trailingOnly = TRUE)

df <- read.csv(file=args[1], head=TRUE)

years <- df$year
newd <- data.frame(years=seq(min(years)-6,max(years)+6,by=0.25))
gam <- gam(df$rtotal1 ~ s(years,k=3), family=gaussian)
gam.pred <- predict.gam(gam, newd)
gam.pred[newd$years<(min(years)-2)] <- gam.pred[newd$years==(min(years)-2)]
gam.pred[newd$years>(max(years)+2)] <- gam.pred[newd$years==(max(years)+2)]
gam.pred[gam.pred>100] <- 100
gam.pred

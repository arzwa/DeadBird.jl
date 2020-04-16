library(WGDgc)
library(tictoc)
t = read.simmap("9dicots.simmap")
d = read.csv("9dicots-f01-1000.csv")
tic("WGDgc, ML")
wgdgc_out = MLEGeneCount(t, d, geomMean=1.5, conditioning="oneInBothClades", fixedRetentionRates=TRUE, equalBDrates=FALSE, startingBDrates=c(1., 1.))
toc()
# 35.215 sec elapsed

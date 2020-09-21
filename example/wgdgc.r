library(WGDgc)
library(tictoc)
t = read.simmap("9dicots.simmap")
d = read.csv("9dicots-f01-1000.csv")
tic("WGDgc, ML")
wgdgc_out = MLEGeneCount(t, d, geomMean=1.5, conditioning="oneInBothClades", fixedRetentionRates=TRUE, equalBDrates=FALSE, startingBDrates=c(1., 1.))
toc()
# 35.215 sec elapsed
#> wgdgc_out
#$birthrate
#[1] 3.406232

#$deathrate
#[1] 2.178061

#$loglikelihood
#[1] -11351.37


tre.string = "(D:{0,18.03},(C:{0,12.06},(B:{0,7.06},A:{0,7.06}):{0,4.99}):{0, 5.97});"
tre.phylo4d = read.simmap(text=tre.string)
dat = data.frame(A=c(2,2), B=c(2,2), C=c(3,3), D=c(4,4));
a = processInput(tre.phylo4d)
logLik_CsurosMiklos(log(c(.2,.3)), nLeaf=4, nFamily=2, a$phyloMat, dat, mMax=11, a$wgdTab, a$edgeOrder)


# Drosophila
t = read.simmap("drosophila/tree.simmap")
d = read.csv("drosophila/counts-oib.csv")
a = processInput(t)

out = logLik_CsurosMiklos(log(c(1., 1.)), 
                    nLeaf=12, 
                    nFamily=10, 
                    a$phyloMat, 
                    d[1:10,], 
                    mMax=743, 
                    a$wgdTab, 
                    a$edgeOrder)



>"For many problems, a function of data and parameters can directly address a particular aspect of a model in a way that would be difficult or awkward using a function of the data alone." BDA p.148

Can we think of test quantities that are functions of both $\lambda, \mu$ and the gene counts? What kind of theoretical results for the linear BDP could be used? Note that things like expectations like $E[X|\lambda,\mu]$ might not be very informative (because of the things on p.146, or am I wrong here?).

Maybe functions of things like

$$ X_{ath} - E[X_{ath}|θ] $$

(where $E[X_{ath}|\theta]$ is the expectation integrated over the root distribution) could be relevant? 

$$

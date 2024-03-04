##################################################################################################
################                                                                  ################
################ Stochastic Simulation Algorithm [Simple Birth AND Death Process] ################
################                                                                  ################
##################################################################################################

x_0 <- 0 # Initial population
b <- 1 # Birth
d <- 0.1 # Death
x <- seq(0, 8, 0.1) # Deterministic Solution
y <- x_0*exp((b-d)*x) # Deterministic Solution
sos <- 10000 # plot(x, y, type = "o")
t <- vector(mode = "list", length = 3)
for (i in 1:10) t[[i]] <- c(0, rep(NA, sos-1))
for (j in 1:10) {
  tt <- 1
  n <- rep(NA, sos)
  n[tt] <- x_0
  while (tt < sos) { # while (n[tt] > 0 && tt < sos) {
    y1 <- runif(1)
    y2 <- runif(1)
    a_0 <- d*n[tt]+b
    t[[j]][[tt+1]] <- -log(y1)/(a_0)+t[[j]][[tt]] # t[[j]][[tt+1]] <- -log(y1)/(b*n[tt]+d*n[tt])+t[[j]][[tt]]
    tt <- tt + 1
    if (y2*a_0 < d*n[tt-1]) n[tt] <- n[tt-1]-1 else n[tt] <- n[tt-1]+1
  }
  plot(t[[j]], n, type = "l", col = "blue")
  hist(diff(t[[j]]), col = "gray")
  hist(n)
}

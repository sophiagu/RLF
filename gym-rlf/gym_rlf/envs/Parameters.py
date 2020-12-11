import math

TickSize = .01 # change it to .1 for illiquid stocks
LotSize = 100
OptionSize = 100
M = 10 # max round lots for holding
K = 5 # max round lots for each trading action
H = 5 # mean reversion half life
S0 = 50

Lambda = math.log(2) / H
theta = .5
sigma = .1
sigma_dh = .01
kappa = 1e-4
kappa_dh = .1
alpha = .02
factor_alpha = -.01
factor_sensitivity = .5
factor_sigma = .12
p_e = 50

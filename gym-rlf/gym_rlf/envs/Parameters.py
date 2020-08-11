import math

TickSize = .1
LotSize = 100
M = 10 # max round lots for holding
K = 5 # max round lots for each trading action
H = 5 # mean reversion half life

Lambda = math.log(2) / H
sigma = .1
kappa = 1e-4
alpha = .015
factor_alpha = -.01
factor_sensitivity = .8
factor_sigma = .12
p_e = 50

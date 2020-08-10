import math

TickSize = .1
LotSize = 100
M = 10 # maximum round lots for holding
K = 5 # maximum round lots for each trading action
H = 5 # half life

Lambda = math.log(2) / H
sigma = .1
kappa = 1e-4
p_e = 50

L = 1000 # test size for each episode

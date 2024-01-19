
import JuMP
import Ipopt

using Random, Distributions

# -------------------------------------------------------------------------- #

# Pre define parameters to be used
Random.seed!(78909434)
N = 100
a = 0.5   # Demand intercept
b = 0.5   # Demand slope
α = 0.5   # Supply intercept
β = 0.5   # Supply slope

σ_u = 1.5
σ_v = 2.5

μ_u = 0.0
μ_v = 0.0

# -------------------------------------------------------------------------- #
# 1. Naive OLS estimation
# -------------------------------------------------------------------------- #

# Simulate observational data
u = zeros(N)
v = zeros(N)
for i in 1:N
    u[i] = rand(Normal(μ_u, σ_u))
    v[i] = rand(Normal(μ_v, σ_v))
end

p = zeros(N)
for i in 1:N
    p[i]=((a-α)/(β+b)) + ((u[i]-v[i])/(β+b))
end

y = zeros(N)
for i in 1:N
    y[i]= ((a*β + b*α)/(b+β)) + ((β*u[i] + b*v[i])/(β+b))
end

D = zeros(N)
S = zeros(N)
for i in 1:N
    D[i] = a - b*p[i] + u[i]
    S[i] = α + β*p[i] + v[i]
end

# Now, estimate naive OLS
ols_naive = JuMP.Model(Ipopt.Optimizer)     # Non-linear system to be solved for Pi and Pj
JuMP.@variable(ols_naive, γ0)
JuMP.@variable(ols_naive, γ1)            

JuMP.@NLobjective(ols_naive, Min, sum( (y[i] - γ0 - γ1*p[i])^2  for i in 1:N) )
JuMP.optimize!(ols_naive)

γ1_h = JuMP.value.(γ1)          # Value obtained: -0.214. Biased due to simultaneity.

γ1_2 = cov(y,p)/var(p)          # Value obtained: -0.214

# -------------------------------------------------------------------------- #
# 2. IV estimation
# -------------------------------------------------------------------------- #

# -------------------------------------------------------------------------- #
#   2SLS

# Simulate instrumental data
c_v = 0.5
c_u = 0.5

σ_ϵu = 0.25
σ_ϵv = 0.5

μ_ϵ_u = 0.0
μ_ϵ_v = 0.0


ϵ_u = zeros(N)
ϵ_v = zeros(N)
for i in 1:N
    ϵ_u[i] = rand(Normal(μ_ϵ_u, σ_ϵu))  # Simulate error terms of first stage
    ϵ_v[i] = rand(Normal(μ_ϵ_v, σ_ϵv))
end 

x_u = zeros(N)
x_v = zeros(N)
for i in 1:N
    x_u[i] = ( u[i] - ϵ_u[i] ) / c_u 
    x_v[i] = ( v[i] - ϵ_v[i] ) / c_v 
end

# Now, estimate first stage
tsls_1 = JuMP.Model(Ipopt.Optimizer)    
JuMP.@variable(tsls_1, c_uh)        

JuMP.@NLobjective(tsls_1, Min, sum( (u[i] - c_uh*x_u[i])^2  for i in 1:N) )
JuMP.optimize!(tsls_1)

cu_iv = JuMP.value.(c_uh)               # Value obtained: 0.488

#

tsls_2 = JuMP.Model(Ipopt.Optimizer)    
JuMP.@variable(tsls_2, c_vh)        

JuMP.@NLobjective(tsls_2, Min, sum( (v[i] - c_vh*x_v[i])^2  for i in 1:N) )
JuMP.optimize!(tsls_2)

cv_iv = JuMP.value.(c_vh)               # Value obtained: 0.481

# -------------------------------------------------------------------------- #
#   New estimation from adjusted u and v after recovering estimated cu and cv from first stage.

uh = zeros(N)
vh = zeros(N)
for i in 1:N
    uh[i] = cu_iv*x_u[i] + ϵ_u[i]
    vh[i] = cv_iv*x_v[i] + ϵ_v[i]
end

p_iv = zeros(N)
for i in 1:N
    p_iv[i]=((a-α)/(β+b)) + ((uh[i]-vh[i])/(β+b))
end

y_iv = zeros(N)
for i in 1:N
    y_iv[i]= ((a*β + b*α)/(b+β)) + ((β*uh[i] + b*vh[i])/(β+b))
end

ols_iv = JuMP.Model(Ipopt.Optimizer)     # Non-linear system to be solved for Pi and Pj
JuMP.@variable(ols_iv, b0_h)
JuMP.@variable(ols_iv, b1_h)            

JuMP.@NLobjective(ols_iv, Min, sum( (y_iv[i] - b0_h - b1_h*p_iv[i])^2  for i in 1:N) )
JuMP.optimize!(ols_iv)

b1_iv = JuMP.value.(b1_h)

b1_iv2 = -(cov(y,x_v)/cov(p,x_v))               # Value obtained: 0.558. This is a far better estimation than the -0.21 obtained via naive OLS.


ed = abs(b1_iv2 - γ1_h)     # estimation difference between naive ols and iv coefficient.
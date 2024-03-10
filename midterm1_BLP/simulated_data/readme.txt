## This data was simulated with 
K = 2             # number of characteristics price and caffeine_score
J = 3             # number of products and number of firms since each firm produces 1 product
The products are MD, TH and SB each of them produced by a single firm of the same name MD, TH and SB respectively. 
S = 50            # draws of nu that is from a multivariate gaussian
T = 200           # number of markets
β = ones(K)*2
β[1] = -1.5       # elasticity of price 
σ = ones(K)
σ[1] = 0.2
γ = ones(1)*0.1
Random.seed!(98426)

In the simulated data sim[].x[2:end,:] and sim[].w are uncorrelated with ξ and ω, but price, sim[].x[1,:], is endogenous. 
The price of the jth firm will depend on the characteristics and costs of all other goods, so these are available as instruments.

midterm_simulated_market_data_s.csv contains shares 
midterm_simulated_market_data_x.csv contains attributes
midterm_simulated_market_data_w.csv contains cost attributes
midterm_simulated_market_data_zd.csv contains demand instruments
midterm_simulated_market_data_zs.csv contains supply instruments


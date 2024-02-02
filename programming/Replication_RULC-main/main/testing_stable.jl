## This code was created by Victor H. Aguiar and Nail Kashaev
## The code is part of the paper "Random Utility and Limited Consideration" by Aguiar, Boccardi, Kashaev, and Kim (ABKK) QE, 2022
## This version is modified to work with Ipopt instead of KNITRO
## The code is written in Julia 1.6.4

## This part requires us to use the Pkg package to install the necessary packages and to keep reproducibility, 
## to obtain the exact numbers as in ABKK you need a KNITRO license, you can get it from Artelys. 
using Pkg
## These lines instantiate the packages and activate the environment, KNITRO is not necessary for this part
Pkg.activate(".")
Pkg.instantiate()
using Distributed
using Statistics
using DataFrames, CSV
## This part is to run the code in parallel
addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
  using Ipopt
end

#@everywhere model=$(ARGS[1])   # put "LA", "EBA", or "RUM"
## This line is not necessary if you are running the code from a shell, but if you are running it from a IDE you need to put the model as an argument
@everywhere model="RUM"   # put "LA", "EBA", or "RUM"
println(model)

## Defining the file directories
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("Replication_RULC-main",tempdir1)[end]]

## Functions
@everywhere include($(rootdir)*"/main/functions_common_testing.jl")

## Common parameters
dYm=5                               # Number of varying options in menus
model=="RUM" ? dYu=6 : dYu=5        # For RUM there are 6 options instead of 5
Menus=collect(powerset(vec(1:dYm))) # Menus
gindex=gindexes(Menus,dYu)          # Indices that correspond to nonzero linearly independent frequencies
U=preferences(dYu)                  # All preference orders
G=matrixcons(gindex, Menus, U)      # Marix of 0 and 1
if model=="RUM"
  G=vcat(G,G,G)                       # To impose stability we need to repeat G for every frame for RUM 
else 
  B=G[1:size(G,1)-length(Menus),1:size(G,2)-length(Menus)]
  Bt=[B zeros(size(B,1),3*length(Menus))]
  G=[Bt;
  zeros(length(Menus), size(B,2)) I zeros(length(Menus), 2*length(Menus));
   Bt;
  zeros(length(Menus), length(Menus)+size(B,2)) I zeros(length(Menus), length(Menus));
   Bt;
   zeros(length(Menus), 2*length(Menus)+size(B,2)) I ]
end

Omegadiag=ones(size(G,1))           # Vector of weigts

## Data for all 3 frames
X1=Matrix(CSV.read(rootdir*"/data/menu_choice_high.csv", DataFrame))
X2=Matrix(CSV.read(rootdir*"/data/menu_choice_medium.csv", DataFrame))
X3=Matrix(CSV.read(rootdir*"/data/menu_choice_low.csv", DataFrame))
println("Data is ready!")
# Sample sizes
N1=size(X1,1)
N2=size(X2,1)
N3=size(X3,1)
# Smallest sample size
N=minimum([N1,N2,N3])
# Smallest sample per menu
Ntau=nintaun(X1)
# Tuning parameter as suggested in KS
taun=sqrt(log(Ntau)/Ntau)

## Testing
println("Testing...")
# Estimates of g for all 3 frames
ghat1=estimateg(X1,gindex)
ghat2=estimateg(X2,gindex)
ghat3=estimateg(X3,gindex)
ghat=vcat(ghat1,ghat2,ghat3)
etahat=kstesstat(ghat,G,Omegadiag,taun,true)
@everywhere begin
  X1=$X1; N1=$N1; X2=$X2; N2=$N2; X3=$X3; N3=$N3; N=$N
  gindex=$gindex; ghat=$ghat; G=$G;
  Omegadiag=$Omegadiag; taun=$taun; etahat=$etahat; Menus=$Menus
end
# Test statistic
Tn=N*kstesstat(ghat,G,Omegadiag,0.0,false)
# Bootstrap statistics
@time Boot=pmap(ksbootseedstable,1:1000) # use ksbootseedstable function
# Pvalue. If it is 0.0, then pvalue<0.001
pvalue=mean(Tn.<=N.*collect(Boot))

## Saving the output
CSV.write(rootdir*"/main/results/Tn_pval_$(model)_stable.csv", DataFrame(Tn=Tn, pvalue=pvalue))
println("Done!")

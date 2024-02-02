using Pkg
Pkg.activate(".")

using Distributed
using Statistics
using DataFrames, CSV
addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
  using KNITRO
end
@everywhere model="LA"
println(model)
## Defining the file directories
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("ReplicationABKK",tempdir1)[end]]

## Functions
@everywhere include($(rootdir)*"/appendix_F/functions_common_hom.jl")

## Common parameters
@everywhere dYm=5                               # Number of varying options in menus
@everywhere Menus=collect(powerset(vec(1:dYm))) # Menus
## Data for high frame
X=Matrix(CSV.read(rootdir*"/data/menu_choice_high.csv", DataFrame))
N=size(X,1)
Y=frequencies(X)
p_pie_LA=pipie_cons(Y)
@everywhere begin
  X=$X; N=$N; p_pie_LA=$p_pie_LA;
end

@everywhere function Latestboot(seed)
  rng1=MersenneTwister(seed)
  Xb=X[rand(rng1,1:N,N),:]
  Yb=frequencies(Xb)
  p_pie_LAb=pipie_cons(Yb)
  Tb=p_pie_LAb .- p_pie_LA
  return Tb[:].^2
end

@time Boot=pmap(Latestboot,1:1000) # use ksbootseedstable function
elem=8 #Pr(1,{1,3})
Tn1=(p_pie_LA[:][elem].-0)^2 # Testing that 1<3
Tn2=(p_pie_LA[:][elem].-1)^2 # Testing that 1>3
pvalue1=mean(Tn1 .<=collect([Boot[i][elem] for i in 1:1000]))
pvalue2=mean(Tn2 .<=collect([Boot[i][elem] for i in 1:1000]))

## Saving the output
CSV.write(rootdir*"/appendix_F/results/Test_LA_homog_high.csv", DataFrame(order=["1<3","1>3"], pvalue=[pvalue1,pvalue2]))
println("Done!")

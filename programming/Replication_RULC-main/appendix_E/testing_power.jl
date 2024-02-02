using Pkg
Pkg.activate(".")

using Distributed
using Statistics
using DataFrames, CSV
addprocs(7)

@everywhere begin
  using Random, Distributions
  using Combinatorics
  using LinearAlgebra
  using JuMP
  using KNITRO
end
@everywhere model="LA"
println(model)
@everywhere lambda=parse(Float64,$ARGS[1])
s1=parse(Int16,ARGS[2])
s2=parse(Int16,ARGS[3])
println(s1)
println(s2)
## Defining the file directories
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("ReplicationABKK",tempdir1)[end]]

## Functions
@everywhere include($(rootdir)*"/appendix_E/functions_common_power.jl")

## Common parameters
dYm=5; dYu=5;
Menus=collect(powerset(vec(1:dYm))) # Menus
gindex=gindexes(Menus,dYu)          # Indices that correspond to nonzero linearly independent frequencies
Udgp=preferences(dYu)               # All preference orders
U=preferences(dYu)
G=matrixcons(gindex, Menus, Udgp)   # Marix of 0 and 1
Omegadiag=ones(size(G,1))           # Vector of weigts
## Data on menus
menusdata=Matrix(CSV.read(rootdir*"/data/menu_choice_pooled.csv", DataFrame))[:,1]
## MM probabilities
dU=length(Udgp)
pmm=zeros(32,6)
for i in 2:32, y in 1:5
  p=0.0
  menu=int2menu(i)
  DD=subsetsint(i)[2:end]
  for pref in Udgp, D in DD
    p=p+(y==int2menu(D)[argmax(pref[int2menu(D)])])*(0.5^(length(menu)))/dU
  end
  pmm[i,y+1]=p
end
pmm[:,1]=1.0 .-sum(pmm[:,2:end],dims=2)
##
# Parameters and functions
@everywhere begin
  N=4000; menusdata=$menusdata;
  gindex=$gindex; G=$G; Omegadiag=$Omegadiag; Menus=$Menus; pmm=$pmm; U=$U

  function powerseed(seed)
    X=fundgp(seed, N, lambda, menusdata,pmm)
    Ntau=nintaun(X)
    taun=sqrt(log(Ntau)/Ntau)
    ghat=estimateg(X,gindex)
    etahat=kstesstat(ghat,G,Omegadiag,taun,true)
    Tn=kstesstat(ghat,G,Omegadiag,0.0,false)
    Boot=pmap(seedt->ksbootseedpower(seedt, X, ghat, etahat, taun),1:1000)
    boot=collect(Boot)
    pvalue=mean(Tn.<=boot)
    return pvalue
  end

  function fundgp(seed, sampsize, lambda, menusdata, pmm)
    X=zeros(sampsize,2)
    X[:,1]=menusdata[rand(MersenneTwister(seed),1:length(menusdata),sampsize)]
    for i in 1:sampsize
      menu=int2menu(Int(X[i,1]))
      pmmt=pmm[Int(X[i,1]),:]
      plambda=zeros(6)
      for k in 2:length(plambda)
        if in(k-1,menu)
          plambda[k]=(5-length(menu))/(6*length(menu))
        end
      end
      plambda[1]=1.0 - sum(plambda[2:end])
      p=lambda.*pmmt .+ (1.0 - lambda).*plambda
      X[i,2]=rand(MersenneTwister(i*seed+13),Categorical(p),1)[1] .-1.0
    end
    return Int.(X)
  end
end
## Testing
println("Testing...")
@time begin Output=map(powerseed,s1:s2) end

## Saving the output
println("Saving the results")
CSV.write(rootdir*"/appendix_E/power_results/pval_$(lambda)_$(s1)_$(s2).csv", DataFrame(Output',:auto))
println("Done")

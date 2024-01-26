###Simple illustration of ELVIS
## vhaguiar@gmail.com
import JuMP
import Ipopt
#import KNITRO
import Distributions
using CSV, DataFrames
using Random
using Tables
using LinearAlgebra
using Statistics

## Generate a data Set
##parameters
T=2
K=2
N=100
##utilities
## u(x)=x1^α+x2^(1-α)
Random.seed!(6749)
α=rand(N)/2
## lambdas
λ=randexp(N).+1
## prices
p=randexp(N,K,T)
## true consumption
ct=zeros(N,K,T)
for i in 1:N
    for k in 1:K
            for t in 1:T
                if k==1
                    ct[i,k,t]=(λ[i]/α[i]*p[i,k,t])^(1/(-1+α[i]))
                end
                if k==2 
                    ct[i,k,t]=((1-α[i])/(λ[i]*p[i,k,t]))^(1/α[i])
                end
            end 
    end
end 

##observed consumption
a=.9
b=1.1
ϵ=(b-a).*rand(N,K,T).+a
co=ct.*ϵ

### ELVIS
##Fix a simulation  number
Ns=1000
## We observe co, and p
co
p
## Sample ct such that it satisfies the model 
##prior α, it has to have the correct support
αp=rand(Ns)
##prior lambda
λp=randexp(Ns)

## prior consumption
## you have to know the model
ctp=zeros(N,Ns,K,T)
for is in 1:Ns 
    for i in 1:N
    for k in 1:K
            for t in 1:T
                if k==1
                    ctp[i,is,k,t]=(λp[is]/αp[is]*p[i,k,t])^(1/(-1+αp[is]))
                end
                if k==2 
                    ctp[i,is,k,t]=((1-αp[is])/(λp[is]*p[i,k,t]))^(1/αp[is])
                end
            end 
    end
    end
end 

ctp

## Moment conditions are known
## E[co-ctp]=0
## co=ctϵ, hence w=ct-co, w=(1-ϵ)ct
## E[w]=E[(1-ϵ)ct|ct]=0 iff E[ϵ|ct]=1, which it is. 
## E[w]=E[E[w|ct]]=0 law of iterated expectations. 

## moments
wsim=zeros(N,K,T)

function jump(c,p)
    αp=rand(N)./2
    ##prior lambda
    λp=randexp(N).+1
    for is in 1:Ns 
        for i in 1:N
        for k in 1:K
                for t in 1:T
                    if k==1
                        wsim[i,k,t]=co[i,k,t]-(λp[i]/αp[i]*p[i,k,t])^(1/(-1+αp[i]))
                    end
                    if k==2 
                        wsim[i,k,t]=co[i,k,t]-((1-αp[i])/(λp[i]*p[i,k,t]))^(1/αp[i])
                    end
                    
                end 
        end
        end
    end 
    wsim
end

function myfun(gamma=gamma,w=w)
    gvec=ones(N,T*K)
    @simd for j=1:T
        @simd  for k=1:K
             @inbounds gvec[:,1]=w[:,1,1]
             @inbounds gvec[:,2]=w[:,1,2]
             @inbounds gvec[:,3]=w[:,2,1]
             @inbounds gvec[:,4]=w[:,2,2]
        end
    end
    gvec/100000
end


w=jump(co,p)

wc=zeros(N,K,T)

## repetitions n1=burn n2=sample accept
repn=[10,1000]
chainM=zeros(N,K*T,repn[2])

function gchain(gamma,co,p,wc=wc,w=w,repn=repn,chainM=chainM)
    r=-repn[1]+1
    while r<=repn[2]
      wc[:,:,:]=jump(co,p);
      logtrydens=(-(wc[:,1,1].^2+wc[:,1,2].^2+wc[:,2,1].^2+wc[:,2,2])+ (w[:,1,1].^2+w[:,1,2].^2+w[:,2,1].^2+w[:,2,2].^2))[:,1,1]
      dum=log.(rand(N)).<logtrydens

      @inbounds w[dum,:,:]=wc[dum,:,:]
      if r>0
        chainM[:,:,r]=myfun(gamma,w)

      end
      r=r+1
    end
end

gamma=[1 2]
gchain(gamma,co,p,wc,w,repn,chainM)
myfun(gamma,w)
chainM
chainM[isnan.(chainM)] .= 0

chainM[isinf.(chainM) .& (chainM .> 0)] .= 10e300
chainM[isinf.(chainM) .& (chainM .< 0)] .= -10e300



##############################################
###Maximum Entropy Moment

#Initializing vector of simulated values of g(x,e)
n=N
dg=K*T
nfast=repn[2]
geta=ones(n,dg)
gtry=ones(n,dg)
@inbounds geta[:,:]=chainM[:,:,1]
# initializing integrated moment h
dvecM=zeros(n,dg)
# Log of uniform
logunif=log.(rand(n,nfast))
# Initializing gamma in CUDA
gamma=ones(dg)
# Initializing value for chain generation
valf=zeros(n)

# MEM MC integral
function MEMMC(gamma...)
    #gamma1=[gamma[1] gamma[2] gamma[3] gamma[4]]'

    for i=1:n
        for j=1:nfast
            valf[i]=0.0
            for t=1:dg
                gtry[i,t]=chainM[i,t,j]
                #valf[i]+=gtry[i,t]*gamma1[t]-geta[i,t]*gamma1[t]
                ## change the line below replacing wc[:,1,1] by gtry[i,t]
                valf[i]+=gtry[i,t]*gamma1[t]-geta[i,t]*gamma1[t]
                #valf[i]+=gtry[i,t]*gamma1[t]-geta[i,t]*gamma1[t] -(geta[i,t]*geta[i,t])+ (gtry[i,t]*gtry[i,t])
                #valf[i]+=gtry[i,t]*([gamma[1] gamma[2] gamma[3] gamma[4]]')[t]-geta[i,t]*([gamma[1] gamma[2] gamma[3] gamma[4]]')[t]
                #println("t=", t, ", i=", i, ", j=", j)
            end
            for t=1:dg
                geta[i,t]=logunif[i,j] < valf[i] ? gtry[i,t] : geta[i,t]
                dvecM[i,t]+=geta[i,t]/nfast
            end
        end
    end
    return nothing
end



gamma1=ones(dg)
MEMMC(gamma1)
dvecM
sum(dvecM,dims=1)./n

## Objective Function
#ELVIS = JuMP.Model(KNITRO.Optimizer) 
ELVIS = JuMP.Model(Ipopt.Optimizer) 
JuMP.@variable(ELVIS, gamma[1:dg])


function ElvisGMM(gamma...)
    MEMMC(gamma)
    dvecM[isnan.(dvecM)] .= 0
    dvecM[isinf.(dvecM) .& (dvecM .> 0)] .= 10e300
    dvecM[isinf.(dvecM) .& (dvecM .< 0)] .= -10e300
    dvec=sum(dvecM,dims=1)'/n
    
    numvar=zeros(dg,dg)
    @simd for i=1:n
        BLAS.syr!('U',1.0/n,dvecM[i,:],numvar)
    end
    var=numvar+numvar'- Diagonal(diag(numvar))-dvec*dvec'
    var[isnan.(var)] .= 0
    var[isinf.(var) .& (var .> 0)] .=10e300
    var[isinf.(var) .& (var .< 0)] .= -10e300
    (Lambda,QM)=eigen(var)
    inddummy=Lambda.>0.001
    An=QM[:,inddummy]
    dvecdum2=An'*(dvec)
    vardum3=An'*var*An
    Omega2=inv(vardum3)
    Qn2=1/2*dvecdum2'*Omega2*dvecdum2

    return Qn2[1]
end

JuMP.register(ELVIS,:ElvisGMM,dg,ElvisGMM;autodiff=true)
JuMP.@NLobjective(ELVIS,Min,ElvisGMM(gamma[1],gamma[2],gamma[3],gamma[4]))

JuMP.optimize!(ELVIS)
minf=JuMP.objective_value(ELVIS)
TSMC=2*minf*n
Chisq=9.488
TSMC2=10e300
##Global optimization in general is hard, we can do a 100 optimizations with different starting points
## In the current implementation of JUMP of 1.8 this is the case, we have to manually change the starting points if this is not longer true
## We take the minimum of the TSMC conditional on being numerically feasible, negative values are due to numerical instability
for t in 1:100
    JuMP.optimize!(ELVIS)
    minf=JuMP.objective_value(ELVIS)
    if (minf>=0 && 2*minf*n<TSMC2)
        TSMC2=2*minf*n
    end
end
# degrees of freedom
αsig = 0.05  # significance level

# get the critical value for the chi-square distribution
critical_value = quantile(Distributions.Chisq(4), 1 - αsig)
TSMC2
## In this case we cannot reject the null, as it should be. 

## Generate a new myfun function that now is defined for t in 1:T=4 and k in 1:K=2, recall that w[:,k,t]





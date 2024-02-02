#This function computes the smallest number of observations per menu (needed for τ_n)
## X=is a dataset
## There are 32 menus
function nintaun(X)
    dM=maximum(X[:,1])
    Ntau=10000.0
    for i=2:dM # First menu is empty
      Ntau=minimum([Ntau,sum(X[:,1].==i)])
    end
    return Ntau
end

# Finding nonredundant indices
function gindexes(Menus,dYu)
    dM=length(Menus)
    MM=zeros(dM,dYu)
    if dYu==6
        #For RUM we add the default to every menu
        Menus=[vcat(1,Menus[i].+1) for i in eachindex(Menus)]
        Menus[1]=[]
    end
    for i in 1:dM
        MM[i,Menus[i][1:end-1]].=1.0 # The last option in the menu is dropped
    end
    # Nonzero and linearly independent frequencies in calibrated probabilities
    gindex=findall(MM.==1.0)
    return gindex
end

# Given data computes empirical frequence for all menus and options
function frequencies(data)
    dM=maximum(data[:,1]); dY=maximum(data[:,2])-minimum(data[:,2])+1;
    F=zeros(Float64,dM,dY)
    for i in 1:size(data,1)
        F[data[i,1],data[i,2]+1]= F[data[i,1],data[i,2]+1]+1.0
    end
    # Computing sample frequencies
    P=zeros(Float64,dM,dY)
    P[1,1]=1.0
    P[2:end,:]=F[2:end,:]./sum(F[2:end,:],dims=2)
    # Throwing away P(default) since it is equal to 1-sum(P[1:5])
    P=P[:,2:end]
    return P
end

# This function translates integer menu identifiers to actual menus
function int2menu(number, menus=Menus)
    return menus[number]
end

# This function translates actual menus to integer menu identifiers
function menu2int(vector::Array{Int64,1})
    p=true; i=1;
    while p
        if int2menu(i)==vector
            p=false
        end
        i=i+1
    end
    return i-1
end

# This function computes all subsets of a given set
# Uses Combinatorics package
function subsetsint(intset,menus=Menus)
    Temp=collect(powerset(int2menu(intset, menus)))
    return [menu2int(Temp[i]) for i in 1:length(Temp)]
end

# Compute m_A(D) given the matrix of frequencies ps
function m(D,A,ps)
    if A==1
        return "error"
    end
    if ~issubset(D,subsetsint(A))
        return 0.0
    else
        etavecT=etavec_con(ps) #Computing the attention index
        if model=="LA" # LA
            return etavecT[D]/sum(etavecT[subsetsint(A)])
        elseif model=="EBA"
            beta=0.0   # EBA
            for i in 1:length(etavecT)
                if intersect(int2menu(i),int2menu(A))==int2menu(D)
                    beta=beta+etavecT[i]
                end
            end
            return beta
        end
    end
end

# Compute the atention index eta(D) for all D given the matrix of frequencies ps
# This function uses the direct formula from the paper
function etavec(ps)
    # Compting P(default)
    po=1.0 .- sum(ps,dims=2)
    DD=subsetsint(32) # All subsets
    etavec1=zeros(Float64,length(DD))
    for i=1:length(DD)
        BB=subsetsint(DD[i]) # Subsets of a given set
        Dt=int2menu(DD[i])
        betatemp=0.0
        for j=1:length(BB)
            betatemp=betatemp+elmobinv(Dt,BB[j],po) #adding up elements of inversion
        end
        etavec1[i]=betatemp
    end
    return etavec1
end

# This function computes the elements of the summation in the defintion if
# the atention index eta is directly computed
function elmobinv(Dt,BB,po)
    if model=="LA"
        return (-1.0)^(length(setdiff(Dt,int2menu(BB))))*po[32]/po[BB]
    elseif model=="EBA"
        return (-1.0)^(length(setdiff(Dt,int2menu(BB))))*po[menu2int(setdiff(vec(1:5),int2menu(BB)))]
    end
end


# Compute the constrained atention index eta(D) for all D given the matrix of frequencies ps
# This function restricts etas to be probabilities
function etavec_con(ps)
    # Computing P(default)
    po=1.0 .- sum(ps,dims=2)
    DD=subsetsint(32) # All subsets
    etamin=Model(Ipopt.Optimizer)
    ## commenting next line because of Ipopt
    #set_optimizer_attribute(etamin,"outlev",0)
    @variable(etamin, etaparam[1:length(DD)]>=0)
    @constraint(etamin, addm, sum(etaparam[t] for t in 1:length(DD))==1)
    if model=="LA"
        ## p(o,X)/p(o,A)
        @objective(etamin,Min,sum((po[32]/po[DD[i]]-sum(etaparam[l] for l in subsetsint(DD[i])))^2 for i in 1:length(DD)))
    elseif model=="EBA"
        ## p(o,X-A)
        @objective(etamin,Min,sum((po[menu2int(setdiff(vec(1:5),int2menu(DD[i])))]-sum(etaparam[l] for l in subsetsint(DD[i])))^2 for i in 1:length(DD)))
    end

    JuMP.optimize!(etamin)

    return value.(etaparam)
end


# Computing the constrained calibrated p_\pi from the matrix of frequencies
# This function is constrained to return probabilities
function pipie_cons(x)
    dM,dYu=size(x)
    MM=[ m(D,A,x) for D in 1:dM, A in 2:dM] # Matrix of consideration probabilites
    ConsB=(x.>0.0) # Matrix of 0/1. ConsB=0 if P(a in A)=0 and =1 otherwise
    pipiemin=Model(Ipopt.Optimizer)
    ##commenting next line because of Ipopt
    #set_optimizer_attribute(pipiemin,"outlev",0)
    @variable(pipiemin, pipieparam[1:dM,1:dYu]>=0)
    @constraint(pipiemin, sump[l=2:dM], sum(pipieparam[l,t] for t in 1:dYu)==1)
    @constraint(pipiemin, [l=1:dM,t=1:dYu], pipieparam[l,t]<=ConsB[l,t])

    @objective(pipiemin,Min, sum([(x[A,a]-sum(MM[D,A]*pipieparam[D,a] for D in 1:dM))^2 for A in 1:dM-1, a in 1:dYu]))
    JuMP.optimize!(pipiemin)
    
    return value.(pipieparam)
end


# Computing G
# U is the set of preferences
# M is the set of menus
# gindexsh are coordinates of nonzero linearly independent p_pi
function matrixcons(gindexsh, M, U)
    dYu=length(U[1])
    dYm=length(M)
    if dYu==6
        M=[vcat(1,M[i].+1) for i in eachindex(M)]
        M[1]=[]
    end
    d=length(U)
    d2=length(gindexsh)
    B=zeros(d2,d)
    m1=1
    for j in 1:d
        pp=zeros(dYm,dYu)
        for i in eachindex(M)
            if length(M[i])>0 # Skipping empty menu
                for k in 1:dYu
                    if k==M[i][argmax(U[j][M[i]])]
                        pp[i,k]=1.0
                    end
                end
            end
        end
        B[:,m1]=pp[gindexsh] #picking only relevant elements, indexes that are not always zero
        m1=m1+1
    end
    if dYu==6 # If model is RUM then return B
        return B
    else #otherwise return
        return [[B zeros(size(B,1),dYm)];[zeros(dYm,size(B,2),) I]]
    end
end

# This function computes the vector of linearly independent nonzero elements of
# p_\pi and eta (if needed)
function estimateg(X,gindex)
  Y=frequencies(X)
  if length(gindex)==80 # If model is RUM
       return [1.0 .- sum(Y,dims=2) Y][gindex]
   else # if model is LA or EBA
       return vcat(pipie_cons(Y)[gindex],etavec(Y))
   end
end

# This function computes the test statistic
function kstesstat(ghat,G,Omegadiag,taun,solution)
    if sum(isnan.(ghat))>0
        return -100
    end
    dr,dg=size(G)
    KS=Model(Ipopt.Optimizer)
    ##commenting next line because of Ipopt 
    ##set_optimizer_attribute(KS,"outlev",0)
    @variable(KS,etavar[1:dg]>=taun/dg) #taun is a tuning parameter
    @objective(KS,Min,sum((sqrt(Omegadiag[r])*(ghat[r]-sum(G[r,l]*etavar[l] for l in 1:dg)))^2 for r in 1:dr))
    JuMP.optimize!(KS)
    if solution==true
        return G*value.(etavar)
    else
        return objective_value(KS)
    end
end

# This function computes the bootstrap statistic given the seed for a given frame
function ksbootseed(seed)
  ghatb=estimateg(genbootsample(X,seed),gindex)
  return kstesstat(ghatb-ghat+etahat,G,Omegadiag,taun,false)
end


# This function computes the bootstrap statistic given the seed for the model
# with stabel preferences
function ksbootseedstable(seed)
    ghat1b=estimateg(genbootsample(X1,seed),gindex)
    ghat2b=estimateg(genbootsample(X2,seed^2),gindex)
    ghat3b=estimateg(genbootsample(X3,2*seed+5),gindex)
    ghatb=vcat(ghat1b,ghat2b,ghat3b)
    return kstesstat(ghatb-ghat+etahat,G,Omegadiag,taun,false)
end

#This function generates a bootstrap sample that has positive probability of the outside option for every menu
function genbootsample(Xt,seed)
    rng1=MersenneTwister(seed)
    dd=false
    Xtb=zeros(size(Xt))
    while dd==false
        Xtb=Xt[rand(rng1,1:size(Xt,1),size(Xt,1)),:]
        dd=minimum(1.0 .- sum(frequencies(Xtb),dims=2))>0.0
    end
    return Xtb
end 


function preferences(dYu)
  U=collect(permutations(vec(1:dYu))) # All preference orders
    return U
end

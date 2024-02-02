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


# Compute the constrained atention index eta(D) for all D given the matrix of frequencies ps
# This function restricts etas to be probabilities
function etavec_con(ps)
    # Computing P(default)
    po=1.0 .- sum(ps,dims=2)
    DD=subsetsint(32) # All subsets
    etamin=Model(KNITRO.Optimizer)
    set_optimizer_attribute(etamin,"outlev",0)
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
    pipiemin=Model(KNITRO.Optimizer)
    set_optimizer_attribute(pipiemin,"outlev",0)
    @variable(pipiemin, pipieparam[1:dM,1:dYu]>=0)
    @constraint(pipiemin, sump[l=2:dM], sum(pipieparam[l,t] for t in 1:dYu)==1)
    @constraint(pipiemin, [l=1:dM,t=1:dYu], pipieparam[l,t]<=ConsB[l,t])

    @objective(pipiemin,Min, sum([(x[A,a]-sum(MM[D,A]*pipieparam[D,a] for D in 1:dM))^2 for A in 1:dM-1, a in 1:dYu]))
    JuMP.optimize!(pipiemin)
    
    return value.(pipieparam)
end


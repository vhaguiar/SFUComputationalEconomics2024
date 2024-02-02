using Pkg
Pkg.activate(".")

using DataFrames, CSV
using Combinatorics
using Plots
using LaTeXStrings
using StatsPlots
using CategoricalArrays

tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("ReplicationABKK",tempdir1)[end]]
dirresults=rootdir*"/tables and figures/results"
pooled = Matrix(CSV.read(rootdir*"/data/menu_choice_pooled.csv", DataFrame));
LC = Matrix(CSV.read(rootdir*"/data/menu_choice_low.csv", DataFrame));
MC = Matrix(CSV.read(rootdir*"/data/menu_choice_medium.csv", DataFrame));
HC = Matrix(CSV.read(rootdir*"/data/menu_choice_high.csv", DataFrame));
dYm=5                               # Number of varying options in menus
Menus=collect(powerset(vec(1:dYm))) # Menus
# An indicator from menu index to menu cardinality
ind = zeros(32)
for i = 1:32
    ind[i] = size(Menus[i])[1]
end
#================================Figure 6======================================#
F6 = zeros(5,3)                 #Dataframe for Figure 6. (Row represents differet |A|, column represents cost level(H, M, L))
C = [HC,MC,LC]
#For low cost
for j = 1:3
        for i = 1:5
                k = findall(x-> x== i, ind)                             #Menu index with |A| = i
                r = findall(x->size(intersect(x,k))[1]!==0 ,C[4-j][:,1])  #Rows in Data satsifes |A|=i
                F6[i,j] = count(x->x==0, C[4-j][r,:][:,2])/size(r)[1]
        end
end
F6
# Draw Figure 6
xlab = ["Low Cost", "Medium Cost", "High Cost"];
Figure6 = plot(F6',xticks = (1:1:3, xlab),
        yticks = 0:0.1:0.5, ylim = (0.05, 0.55), linewidth = 2,
        linestyle = [:dash :dot :dashdot :solid :dash],markershape =[:rect :x :star6 :diamond :circ],
        label = ["|A| = 1" "|A| = 2" "|A| = 3" "|A| = 4" "|A| = 5"],legendfontsize = 7, legend=:topleft)
savefig(Figure6,dirresults*"/Figure6.pdf")
#======================================Figure 7 ======================================#
# Data frame of H cost, M cost, L cost, and pooled data.
F7H = zeros(6)
F7M = zeros(6)
F7L = zeros(6)
F7P = zeros(6)
nH = size(HC,1);nM = size(MC,1);nL = size(LC,1);
n = size(pooled,1)
#High cost
for i = 1:6
        if i <= 5
                F7H[i] = size(findall(x-> x==i, HC[:,2]))[1]/nH
        else
                F7H[i] = size(findall(x-> x==0, HC[:,2]))[1]/nH #This simply reorders the column so the outside option is the last one
        end
end
#Medium Cost
for i = 1:6
        if i <= 5
                F7M[i] = size(findall(x-> x==i, MC[:,2]))[1]/nM
        else
                F7M[i] = size(findall(x-> x==0, MC[:,2]))[1]/nM
        end
end
#Low Cost
for i = 1:6
        if i <= 5
                F7L[i] = size(findall(x-> x==i, LC[:,2]))[1]/nL
        else
                F7L[i] = size(findall(x-> x==0, LC[:,2]))[1]/nL
        end
end
#Pooled
for i = 1:6
        if i<= 5
                F7P[i] = size(findall(x-> x==i, pooled[:,2]))[1]/n
        else
                F7P[i] = size(findall(x-> x==0, pooled[:,2]))[1]/n
        end
end
# F7 = hcat(F7H, F7M, F7L)
F7 = hcat(F7L, F7M, F7H)
# Plot figure 7

xlab = CategoricalArray(repeat(["Low Cost", "Medium Cost", "High Cost"], outer = 6))
levels!(xlab,["Low Cost", "Medium Cost", "High Cost"])
legend = repeat(vcat("Lottery" .* string.(1:5), "Outside"), inner=3)
Figure7 = groupedbar(xlab, F7', group = legend, ylim = (0,0.4), yticks = 0:0.05:0.4,
        title = "", bar_width = 0.7, legendfontsize=6, legend=:topleft,
        lw = 0, framestyle = :box, label = ["Lottery 1" "Lottery 2" "Lottery 3" "Lottery 4" "Lottery 5" "Default"])
savefig(Figure7,dirresults*"/Figure7.pdf")

#======================================Figure8======================================#
#Calculate p(a,A) for each A and Cost level.
C = [HC,MC,LC,pooled]
p = zeros(4,31,6)                       #p(a,A); dimension 1: HC,MC,LC and pooled; dimension 2: A, dimension 3: a;
for l = 1:4
        for i = 1:31
                k = findall(x->x == i+1, C[l][:,1])             #row index with A = i+1 and cost l
                lCk = C[l][k,:]; n = size(lCk, 1)               #sub data with A=i+1 and cost l
                for j = 1:6
                        p[l,i,j] = sum(lCk[:,2].==j-1)/n        #prob of j-1 (0,1,2,3,4,5) is choosen under menu i+1 (2,3,...,32) and cost l
                end
        end
end
ind_2 = ones(31,6)   #This is the indicator of a in A (rows are A, columns are a)
for i = 1:31
        for j = 2:6                                                     #item 1 (default) always exist in any menu, so it is not considered
                if size(intersect(Menus[i+1],j-1))[1] == 0              #Condition that item j-1 (1,2,3,4,5) is not in menu i+1 (2,3,4....,32);
                        ind_2[i,j] = 0
                end
        end
end
# Average p(a,A) over identical |A| such that a\in A
# This forms Sumarry Statistics of figure 8
F8 = zeros(4,5,6)
for l = 1:4
        for i = 1:5
                for j = 1:6
                        x = findall(x->x==i, ind).-1    #Menu index with |A| = i (.-1 as there is no A=1 in data)
                        y = findall(x->x==1,ind_2[:,j]) #Menu index with j \in A
                        k = intersect(x,y)              #Menu index with |A| = i && j \in A
                        F8[l,i,j] = sum(p[l,k,j])/length(k)
                end
        end
end
# Using k.-1 in the inner loop is because ind goes from 1 to 32, with the first element represents |A| = 0
# while p[l,:,j] goes from 1:31, with the first element represents |A|= 1
#Plot Figure 8
xlab = ["1"; "2";"|A|=3";"4";"5"]
f2 = plot(F8[1,:,:], xticks = (1:1:5,xlab), linewidth = 1.5, title = "High cost",
        ylim = (0, 0.85), yticks = 0:0.2:0.8,
        label = ["Outside" "Lottery 1" "Lottery 2" "Lottery 3" "Lottery 4" "Lottery 5"],
        linestyle = [:dash :dot :dashdot :solid :dash :dashdotdot],markershape =[:rect :x :star6 :diamond :circ :rtriangle]);
f3 = plot(F8[2,:,:], xticks = (1:1:5,xlab), linewidth = 1.5, title = "Medium cost",
        ylim = (0, 0.85), yticks = 0:0.2:0.8,
        label = ["Outside" "Lottery 1" "Lottery 2" "Lottery 3" "Lottery 4" "Lottery 5"],
        linestyle = [:dash :dot :dashdot :solid :dash :dashdotdot],markershape =[:rect :x :star6 :diamond :circ :rtriangle]);
f4 = plot(F8[3,:,:], xticks = (1:1:5,xlab), linewidth = 1.5, title = "Low cost",
        ylim = (0, 0.85), yticks = 0:0.2:0.8,
        label = ["Outside" "Lottery 1" "Lottery 2" "Lottery 3" "Lottery 4" "Lottery 5"],
        linestyle = [:dash :dot :dashdot :solid :dash :dashdotdot],markershape =[:rect :x :star6 :diamond :circ :rtriangle]);
Figure8 = plot(f2, f3, f4, layout =3 , legend = true,legendfontsize = 4, titlefont = font(8))
savefig(Figure8,dirresults*"/Figure8.pdf")
#==================================Table 2======================================#
LAres=Matrix(CSV.read(rootdir*"/main/results/Tn_pval_LA_stable.csv", DataFrame));
EBAres=Matrix(CSV.read(rootdir*"/main/results/Tn_pval_EBA_stable.csv", DataFrame));
RUMres=Matrix(CSV.read(rootdir*"/main/results/Tn_pval_RUM_stable.csv", DataFrame));
Table2=vcat(RUMres,LAres,EBAres);
CSV.write(dirresults*"/Table2.csv",  DataFrame(Table2,:auto), writeheader=false)
#==================================Table 4======================================#
AverageObs = zeros(31, 2)       #Number of OBS in each choice set (column1), average over cardinality (column2)
                                #Second and third columns in Table 4
for i = 1:31
        AverageObs[i,1] = sum(HC[:,1].==i+1)
        AverageObs[i,2] = AverageObs[i,1]/(ind[i+1]+1)
end
#A function convert vert to a number. Example convert [1,2,3] to 123
function vectonum(vector)
  result = 0
  vector = reverse(vector)
  for (idx, val) in enumerate(vector)
    val_ = val * 10 ^ (idx - 1)
    result += val * 10 ^ (idx - 1)
  end
  return result
end
#The menu of choice (The first column in Table 4)
Choice = Array{Union{Nothing, String}}(nothing, 31)
for i = 1:31
        Choice[i] = "o"*string(convert(Int,vectonum(Menus[i+1])))
end
Table4 = hcat(Choice, AverageObs)
CSV.write(dirresults*"/Table4.csv",  DataFrame(Table4,:auto), writeheader=false)
#==================================Table 5======================================#
p025=Matrix(CSV.read(rootdir*"/appendix_E/power_results/pval_0.25_1_1000.csv", DataFrame));
p05=Matrix(CSV.read(rootdir*"/appendix_E/power_results/pval_0.5_1_1000.csv", DataFrame));
t025_5=sum(p025.<0.05)/length(p025)
t025_10=sum(p025.<0.1)/length(p025)
t05_5=sum(p05.<0.05)/length(p05)
t05_10=sum(p05.<0.1)/length(p05)
Table5=DataFrame(lambda=[0.25,0.5], Confidence95=[t025_5,t05_5],Confidence90=[t025_10,t05_10])
CSV.write(dirresults*"/Table5.csv",  DataFrame(Table5), writeheader=false)
using Pkg
# Create and activate a new environment
Pkg.activate("Clustering_env")

Pkg.add(["DataFrames", "Distances", "Clustering", "Random", "CategoricalArrays","StatsBase"])

using DataFrames, Distances, Clustering, Random, CategoricalArrays

##k means 
# make a random dataset with 1000 random 5-dimensional points
X = rand(5, 1000)

# cluster X into 20 clusters using K-means
R = kmeans(X, 20; maxiter=200, display=:iter)

@assert nclusters(R) == 20 # verify the number of clusters

a = assignments(R) # get the assignments of points to clusters
c = counts(R) # get the cluster sizes
M = R.centers # get the cluster centers

#####################################################
### k-medoids for choice sets
### k-modes
using DataFrames, Clustering, Distances

using DataFrames, Random

using StatsBase

function generate_choices(consumers, ranking_map, choice_sets)
    choices = []
    for i in 1:length(consumers)
        ranking = ranking_map[rankings[i]] # Use the ranking_map dictionary instead of rankings
        choice_set = choice_sets[i]
        choice = choice_set[argmax([findfirst(isequal(x), ranking) for x in choice_set])]
        push!(choices, choice)
    end
    return choices
end



scale=100
n = 120*scale
consumers = 1:n
choice_sets = vcat(fill(["a"], 60*scale), fill(["b", "c"], 60*scale))
rankings = vcat(fill(1, 10*scale), fill(2, 15*scale), fill(3, 20*scale), fill(4, 30*scale), fill(5, 35*scale), fill(6, 10*scale))

ranking_map = Dict(
    1 => ["a", "b", "c"],
    2 => ["a", "c", "b"],
    3 => ["b", "a", "c"],
    4 => ["b", "c", "a"],
    5 => ["c", "a", "b"],
    6 => ["c", "b", "a"]
)

# Shuffle the consumers, choice_sets, and rankings arrays
Random.seed!(42)
shuffled_indices=shuffle(consumers) 
shuffled_indices_1 = shuffle(consumers)
shuffled_indices_2 = shuffle(consumers)
shuffled_indices_3 = shuffle(consumers)
shuffled_choice_sets_1 = choice_sets[shuffled_indices_1]
shuffled_choice_sets_2 = choice_sets[shuffled_indices_2]
shuffled_choice_sets_3 = choice_sets[shuffled_indices_3]


shuffled_rankings = rankings[shuffled_indices]

# Generate choices for each time period
choices_t1 = generate_choices(consumers, ranking_map, shuffled_choice_sets_1)
choices_t2 = generate_choices(consumers, ranking_map, shuffled_choice_sets_2)
choices_t3 = generate_choices(consumers, ranking_map, shuffled_choice_sets_3)

# Create the DataFrame
data = DataFrame(
    consumer_id = consumers,
    choice_t1 = categorical(choices_t1),
    choice_t2 = categorical(choices_t2),
    choice_t3 = categorical(choices_t3)
)

data_matrix = Matrix(data[:, 2:end])



# Define the custom K-modes distance function
function k_modes_distance(a::AbstractVector, b::AbstractVector)
    return sum([count(a[i] .!= b[i]) for i in 1:length(a)])
end



# Compute the pairwise distance matrix using the custom function
n = size(data_matrix, 1)
distances = Matrix{Int}(undef, n, n)

# Compute the distance matrix using the custom function
for i in 1:n
    for j in 1:n
        distances[i, j] = k_modes_distance(data_matrix[i, :], data_matrix[j, :])
    end
end

k = 2^3
Random.seed!(42)
result = kmedoids(distances, k)
resultcomparison=result.totalcost
data[!, "cluster"] = result.assignments
for i in 1:10000
    Random.seed!(i)
    init_centroids = sample(1:n, k, replace=false)
    # Perform k-medoids clustering with random initial centroids
    result = kmedoids(distances, k, init=init_centroids)
    # Store the cost of the clustering
    if result.totalcost<resultcomparison
        data[!, "cluster"] = result.assignments
    end 
end

    





function choice_set_to_label(choice_set)
    if choice_set == ["a"]
        return 2
    elseif choice_set == ["b", "c"]
        return 1
    # elseif choice_set == ["a", "c"]
    #     return 1
    else
        error("Invalid choice set")
    end
end

choice_set_labels_1 = [choice_set_to_label(choice_set) for choice_set in choice_sets[shuffled_indices_1]]
choice_set_labels_2 = [choice_set_to_label(choice_set) for choice_set in choice_sets[shuffled_indices_2]]
choice_set_labels_3 = [choice_set_to_label(choice_set) for choice_set in choice_sets[shuffled_indices_3]]

choice_set_labels=choice_set_labels_1+ choice_set_labels_2*10+choice_set_labels_3*100

choice_set_labels
data[!, "truelabels"]=choice_set_labels 

##validation  111
cluster_111_rows = data[data[!, :truelabels] .== 111, :]
unique(cluster_111_rows.cluster)
countmap(cluster_111_rows.cluster)
##validation 221
cluster_221_rows = data[data[!, :truelabels] .== 221, :]
unique(cluster_221_rows.cluster)
countmap(cluster_221_rows.cluster)
##validation 222
cluster_222_rows = data[data[!, :truelabels] .== 222, :]
unique(cluster_222_rows.cluster)
countmap(cluster_222_rows.cluster)

##validation 211
cluster_211_rows = data[data[!, :truelabels] .== 211, :]
unique(cluster_211_rows.cluster)
countmap(cluster_211_rows.cluster)
##validation 121
cluster_121_rows = data[data[!, :truelabels] .== 121, :]
unique(cluster_121_rows.cluster)
countmap(cluster_121_rows.cluster)
##validation 122
cluster_122_rows = data[data[!, :truelabels] .== 122, :]
unique(cluster_122_rows.cluster)
countmap(cluster_122_rows.cluster)
##validation 212
cluster_212_rows = data[data[!, :truelabels] .== 212, :]
unique(cluster_212_rows.cluster)
countmap(cluster_212_rows.cluster)
##validation 112
cluster_112_rows = data[data[!, :truelabels] .== 112, :]
unique(cluster_112_rows.cluster)
countmap(cluster_112_rows.cluster)



# Initialize an empty dictionary
mapping = Dict{Int, Int}()

# Iterate over the combinations and create the mapping for v1
# Create a dictionary for mapping
mapping = Dict(
    111 => 8,7,
    221 => 2,
    222 => 5,
    211 => 3,7,
    121 => 2,
    122 => 4,
    212 => 1,
    112 => 6,1
)


mapping = Dict(
    111 => 8,
    221 => 2,
    222 => 5,
    211 => 3,
    121 => 2,
    122 => 4,
    212 => 1,
    112 => 6
)

# Print the resulting mapping
println(mapping)

mapping[choice_set_labels[1]]

# Define the choice_set_labels vector with some example values

# Initialize an empty vector to store the mapped values
true_labels = Int[]

# Iterate over the choice_set_labels vector and apply the mapping
for label in choice_set_labels
    mapped_label = mapping[label]
    push!(true_labels, mapped_label)
end

# Print the resulting mapped_choice_set_labels vector
println(true_labels)
println(data.cluster)

# Create the confusion matrix
function create_confusion_matrix(true_labels, predicted_labels)
    n_labels = k
    confusion_matrix = zeros(Int, n_labels, n_labels)
    
    for (true_label, predicted_label) in zip(true_labels, predicted_labels)
        confusion_matrix[true_label, predicted_label] += 1
    end
    
    return confusion_matrix
end

confusion_matrix = create_confusion_matrix(true_labels, data.cluster)

#### 
using Random

Random.seed!(9847587)

# Generate a random initial set of centroids

init_centroids = sample(1:n, k, replace=false)

# Perform k-medoids clustering with random initial centroids
result = kmedoids(distances, k, init=init_centroids)

#result = kmedoids(distances, k, init=:kmpp)

data[!, "cluster"] = result.assignments

##validation  111
cluster_111_rows = data[data[!, :truelabels] .== 111, :]
unique(cluster_111_rows.cluster)
countmap(cluster_111_rows.cluster)
##validation 221
cluster_221_rows = data[data[!, :truelabels] .== 221, :]
unique(cluster_221_rows.cluster)
countmap(cluster_221_rows.cluster)
##validation 222
cluster_222_rows = data[data[!, :truelabels] .== 222, :]
unique(cluster_222_rows.cluster)
countmap(cluster_222_rows.cluster)

##validation 211
cluster_211_rows = data[data[!, :truelabels] .== 211, :]
unique(cluster_211_rows.cluster)
countmap(cluster_211_rows.cluster)
##validation 121
cluster_121_rows = data[data[!, :truelabels] .== 121, :]
unique(cluster_121_rows.cluster)
countmap(cluster_121_rows.cluster)
##validation 122
cluster_122_rows = data[data[!, :truelabels] .== 122, :]
unique(cluster_122_rows.cluster)
countmap(cluster_122_rows.cluster)
##validation 212
cluster_212_rows = data[data[!, :truelabels] .== 212, :]
unique(cluster_212_rows.cluster)
countmap(cluster_212_rows.cluster)
##validation 112
cluster_112_rows = data[data[!, :truelabels] .== 112, :]
unique(cluster_112_rows.cluster)
countmap(cluster_112_rows.cluster)

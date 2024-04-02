## Author: Victor H. Aguiar
## Version Julia 1.7.2
tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("SFUComputationalEconomics2024",tempdir1)[end]]
cd(rootdir)
cd(rootdir*"/NN")
using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
## add Plots, Random, Combinatorics, LinearAlgebra, JuMP, Ipopt, Flux, Statistics, Parameters
using Plots
addprocs(7)

@everywhere begin
  using Random
  using Combinatorics
  using LinearAlgebra
  using JuMP
  #using Gurobi
  #using KNITRO
  #using Ipopt
end

##Machine Learning
using Flux


X1=CSV.read(rootdir*"/NN/data/ABKK_nnvictor.csv", DataFrame)

@everywhere model="RUM"   # put "LA", "RCG", or "RUM"

## Common parameters
dYm=5                               # Number of varying options in menus
Menus=collect(powerset(vec(1:dYm))) # Menus

## Select only big Menu
##data=X1[X1.Menu_Nail.==32,:]

using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

# This Julia script defines a mutable structure named Args using the @with_kw macro from the Parameters.jl package.
# The Args struct has two fields: lr and repeat, which are of type Float64 and Int respectively.
# lr represents a learning rate and has a default value of 0.5.
# repeat represents the number of times an operation is to be repeated and has a default value of 110.
# The @with_kw macro allows for the creation of Args instances using keyword arguments, and fields can be omitted if they have default values.

@with_kw mutable struct Args
    lr::Float64 = 0.5
    repeat::Int = 110
end

function get_processed_data(args)
    labels = string.(X1.choice)
    features = Matrix(X1[:,2:end])'

    # Subract mean, divide by std dev for normed mean of 0 and std dev of 1.
    normed_features = normalise(features, dims=2)

    klasses = sort(unique(labels))
    onehot_labels = onehotbatch(labels, klasses)

    # Split into training and test sets, 2/3 for training, 1/3 for test.
    train_indices = [1:3:12297 ; 2:3:12297]

    X_train = normed_features[:, train_indices]
    y_train = onehot_labels[:, train_indices]

    X_test = normed_features[:, 3:3:12297]
    y_test = onehot_labels[:, 3:3:12297]

    #repeat the data `args.repeat` times
    train_data = Iterators.repeated((X_train, y_train), args.repeat)
    test_data = (X_test,y_test)

    return train_data, test_data
end

# Accuracy Function
accuracy(x, y, model) = mean(onecold(model(x)) .== onecold(y))

# Function to build confusion matrix
function confusion_matrix(X, y, model)
    ŷ = onehotbatch(onecold(model(X)), 1:6)
    y * transpose(ŷ)
end

function train(; kws...)
    # Initialize hyperparameter arguments
    args = Args(; kws...)

    #Loading processed data
    train_data, test_data = get_processed_data(args)

    # Declare model taking 37 features as inputs and outputting 6 probabiltiies,
    # one for each lottery.
    ##Create a traditional Dense layer with parameters W and b.
    ##y = σ.(W * x .+ b), x is of length 37 and y is of length 6.
    model = Chain(Dense(37, 6))

    # Defining loss function to be used in training
    # For numerical stability, we use here logitcrossentropy
    loss(x, y) = logitcrossentropy(model(x), y)

    # Training
    # Gradient descent optimiser with learning rate `args.lr`
    optimiser = Descent(args.lr)

    println("Starting training.")
    Flux.train!(loss, Flux.params(model), train_data, optimiser)

    return model, test_data
end

function test(model, test)
    # Testing model performance on test data
    X_test, y_test = test
    accuracy_score = accuracy(X_test, y_test, model)

    println("\nAccuracy: $accuracy_score")

    # Sanity check.
    @assert accuracy_score > 0.8

    # To avoid confusion, here is the definition of a Confusion Matrix: https://en.wikipedia.org/wiki/Confusion_matrix
    println("\nConfusion Matrix:\n")
    display(confusion_matrix(X_test, y_test, model))
    ##Loss function
    println("Loss test data")
    loss(x, y) = logitcrossentropy(model(x), y)
    display(loss(X_test,y_test))
end

cd(@__DIR__)
model, test_data = train()
test(model, test_data)

normopt=true

labels = string.(X1.choice)
features = Matrix(X1[:,2:end])'
normopt ? normed_features = normalise(features, dims=2) : normed_features=features

klasses = sort(unique(labels))
onehot_labels = onehotbatch(labels, klasses)

# Split into training and test sets, 2/3 for training, 1/3 for test.
train_indices = [1:3:12297 ; 2:3:12297]

X_train = normed_features[:, train_indices]
y_train = onehot_labels[:, train_indices]

X_test = normed_features[:, 3:3:12297]
y_test = onehot_labels[:, 3:3:12297]

#repeat the data `args.repeat` times
# The Iterators.repeated function is used here to create an iterator that repeats the given 
# item, in this case, the tuple (X_train, y_train), a specified number of times, here 1000 times. 
# Breakdown of its components:
# - (X_train, y_train): This tuple contains our training data, where X_train is the matrix of input 
#   features for each sample, and y_train is the corresponding labels or targets for each sample.
# - 1000: The number of repetitions. This does not physically duplicate the data 1000 times; 
#   instead, it creates a virtual loop over the same dataset 1000 times. This is effectively 
#   simulating a scenario where the original dataset is much larger than it actually is. 
# When this iterator is passed to the Flux.train! function for the training loop, it allows the 
# model to train on the same dataset multiple times (1000 times in this case), without the need 
# for physically duplicating the data, thus simulating an extended exposure to the training data.

train_data = Iterators.repeated((X_train, y_train), 1000)
test_data = (X_test,y_test)



##model = softmax(Dense(37, 6),dims=6)

model2 = Chain(
  Dense(37, 37, relu),
  Dense(37, 6),
  softmax)

loss(x, y) = Flux.mse(model2(x), y)
optimiser = Descent(0.5)

#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, Flux.params(model2), train_data, optimiser)
loss(X_test,y_test)
accuracy(X_test,y_test,model2)
######################
#####################
  ## Third model
model3 = Chain(
  Dense(37,37,relu),
  Dense(37, 37, relu),
  Dense(37, 37, relu),
  Dense(37, 6),
  softmax)

loss(x, y) = Flux.mse(model3(x), y)
#optimiser = Descent(0.5)
optimiser = ADAM(0.001, (0.9, 0.8))

#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, Flux.params(model3), train_data, optimiser)

loss(X_train,y_train)
loss(X_test,y_test)
accuracy(X_test,y_test,model3)
model3(X_test)
X_testc=features[:, 3:3:12297]
X_testc[37,:]=zeros(size(X_testc)[2])
X_testc[1,:].=minimum(X_test[1,:])
X_testc=normalise(X_testc,dims=2)
model3(X_testc)
loss(X_testc,y_test)
accuracy(X_testc,y_test,model3)

confusion_matrix(X_testc,y_test,model3)
######################
#####################
## Fourth model
model4 = Chain(
  Dense(37,37,relu),
  Dense(37, 37, relu),
  Dense(37, 37, relu),
  Dense(37, 6,relu),
  Dense(6, 6),
  softmax)

loss(x, y) = Flux.mse(model4(x), y)
#optimiser = Descent(0.5)
optimiser = ADAM(0.01, (0.9, 0.999))
#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, Flux.params(model4), train_data, optimiser)

loss(X_train,y_train)
loss(X_test,y_test)
accuracy(X_testc,y_test,model4)
confusion_matrix(X_testc,y_test,model4)

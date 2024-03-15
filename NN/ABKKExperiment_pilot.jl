tempdir1=@__DIR__
rootdir=tempdir1[1:findfirst("TopicsDecisionMaking",tempdir1)[end]]
cd(rootdir)
cd(rootdir*"\\NN\\")
using Pkg
Pkg.activate()
using Distributed
using Statistics
using DataFrames, CSV
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
model=="RUM" ? dYu=6 : dYu=5        # For RUM there are 6 options instead of 5
Menus=collect(powerset(vec(1:dYm))) # Menus

## Select only big Menu
##data=X1[X1.Menu_Nail.==32,:]

using Flux: logitcrossentropy, normalise, onecold, onehotbatch
using Statistics: mean
using Parameters: @with_kw

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
    # one for each species of iris.
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
    Flux.train!(loss, params(model), train_data, optimiser)

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
Flux.train!(loss, params(model2), train_data, optimiser)
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
Flux.train!(loss, params(model3), train_data, optimiser)

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
  Dense(37, 6),
  softmax,
  Dense(6,6))

loss(x, y) = Flux.mse(model4(x), y)
#optimiser = Descent(0.5)
optimiser = ADAM(0.001, (0.9, 0.8))
#   train_data =  Iterators.repeated((features,labels), 100)
#   test_data = (features,y_test)
Flux.train!(loss, params(model4), train_data, optimiser)

loss(X_train,y_train)
loss(X_test,y_test)
accuracy(X_testc,y_test,model4)
confusion_matrix(X_testc,y_test,model4)

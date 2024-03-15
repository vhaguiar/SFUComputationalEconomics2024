## v"1.10.1"
using Plots
using MLJBase
using StableRNGs # Seeding generator for reproducibility

# Generate fake data
X, y = make_blobs(10_000, 3; centers=2, as_table=false, rng=2020);
X = Matrix(X');
y = reshape(y, (1, size(X, 2)));
f(x) =  x == 2 ? 0 : x
y2 = f.(y);

# Input dimensions
input_dim = size(X, 1);

# Train the model
nn_results = train_network([input_dim, 5, 3, 1], X, y2; η=0.01, epochs=50, seed=1, verbose=true);

# Plot accuracy per iteration
p1 = plot(nn_results.accuracy,
         label="Accuracy",
         xlabel="Number of iterations",
         ylabel="Accuracy as %",
         title="Development of accuracy at each iteration");

# Plot cost per iteration 
p2 = plot(nn_results.cost,
         label="Cost",
         xlabel="Number of iterations",
         ylabel="Cost (J)",
         color="red",
         title="Development of cost at each iteration");

# Combine accuracy and cost plots
plot(p1, p2, layout = (2, 1), size = (800, 600))
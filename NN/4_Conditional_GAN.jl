##Julia Version 1.6
##Author: Victor H. Aguiar
##Date: 2023-04-11
##Description: This script is to train a simple GAN to generate data from a normal distribution.

using Pkg

# Create and activate a new environment
Pkg.activate("GAN_env")
# Pkg.add("Flux")
# Pkg.add("Distributions")
# Pkg.add("Plots")
#Pkg.add("Zygote")
#Pkg.add("Plots")


using Random
using Statistics
using Flux
using Flux: params
using Flux.Optimise: update!
using Zygote: gradient
using Plots
using Distributions
## Set a fixed random seed for reproducibility
using Random
Random.seed!(8942)


# Hyperparameters
epochs = 20000
batch_size = 32
latent_size = 5
lr = 0.001

# Function to create real samples and their corresponding conditions
function real_data(batch_size)
    means = rand(Float32, 1, batch_size) * 20 .- 10
    stds = rand(Float32, 1, batch_size) * 10 .+ 1
    samples = randn(Float32, 1, batch_size) .* stds .+ means
    return samples, means, stds
end

# Function to create random noise (latent vector)
function sample_noise(batch_size, latent_size)
    return randn(Float32, batch_size, latent_size)
end

# Define the Generator and Discriminator networks using Flux
Generator() = Chain(Dense(latent_size + 2, 64, relu), BatchNorm(64, relu), Dense(64, 1))
Discriminator() = Chain(Dense(3, 64, relu), BatchNorm(64, relu),Dense(64, 1, σ))

G = Generator()
D = Discriminator()

# Optimizers
opt_G = ADAM(lr)
opt_D = ADAM(lr)

# Training loop
for epoch in 1:epochs
    # Sample real data and their corresponding conditions
    real_samples, real_means, real_stds = real_data(batch_size)
    real_conditions = vcat(real_means, real_stds)

    # Sample noise and generate fake data
    noise = sample_noise(batch_size, latent_size)
    gen_input = vcat(noise', real_conditions)
    fake_samples = G(gen_input)

    # Train the Discriminator
    d_loss() = -mean(log.(D(vcat(real_samples, real_conditions)))) - mean(log.(1 .- D(vcat(fake_samples, real_conditions))))
    grads_D = gradient(() -> d_loss(), params(D))
    update!(opt_D, params(D), grads_D)

    # Train the Generator
    noise = sample_noise(batch_size, latent_size)
    gen_input = vcat(noise', real_conditions)
    g_loss() = -mean(log.(D(vcat(G(gen_input), real_conditions))))
    grads_G = gradient(() -> g_loss(), params(G))
    update!(opt_G, params(G), grads_G)

    # Print losses
    if epoch % 500 == 0
        println("Epoch: $epoch | Discriminator Loss: $(d_loss()) | Generator Loss: $(g_loss())")
    end
end

# Test the Generator
μ=10
σst=10
size=10000
test_noise = sample_noise(size, latent_size)
test_mean = fill(Float32(μ*1.0), size)
test_std = fill(Float32(σst*1.0), size)
test_input = vcat(test_noise', test_mean', test_std')
generated_samples = G(test_input)
mean(generated_samples)
std(generated_samples)
histogram(vec(generated_samples), bins=30, xlabel="Value", ylabel="Frequency", label="Generated samples", title="Generated Samples Distribution", legend=:topright)

println("Generated sample: $generated_sample")


### old code
# Generate a sample of 100 data points using the trained Generator
@time noise = sample_noise(10000, latent_size)
@time generated_samples = G(noise')
@time validation_samples=real_data(10000)
# Display the generated data
println("Generated data sample: \n", generated_samples)
mean(generated_samples)
var(generated_samples)
mean(validation_samples)
var(validation_samples)

histogram(vec(generated_samples), bins=30, xlabel="Value", ylabel="Frequency", label="Generated samples", title="Generated Samples Distribution", legend=:topright)

histogram(vec(validation_samples), bins=30, xlabel="Value", ylabel="Frequency", label="Generated samples", title="Validation Samples Distribution", legend=:topright)

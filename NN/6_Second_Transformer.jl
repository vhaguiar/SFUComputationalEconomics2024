##Julia Version 1.8
##Author: Victor H. Aguiar
##Date: 2023-04-18
## based on https://chengchingwen.github.io/Transformers.jl/dev/tutorial/
##Description: This script is to train a simple tranformer to do a copy-paste task. The copy task is a toy test case of a sequence transduction problem that simply return the same sequence as the output. Here we define the input as a random sequence of white space separable number from 1~10 and length 10. we will also need a start and end symbol to indicate where is the begin and end of the sequence. We can use Transformers.TextEncoders.TransformerTextEncoder to preprocess the input (add start/end symbol, convert to one-hot encoding, ...).
using Pkg

# Create and activate a new environment
Pkg.activate("Transformer_env")
# Add necessary packages
Pkg.add("Flux")
Pkg.add("Transformers")
Pkg.add("Statistics")
Pkg.add("Optimisers")
Pkg.add("Zygote")
Pkg.add("ChainRulesCore")

using Flux
using Transformers
using Transformers.Layers
using Transformers.TextEncoders

const N = 2
const V = 10
const Smooth = 1e-6
const Batch = 32
const lr = 1e-4

# Roman numeral labels
const roman_labels = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]

# Update vocabulary to include Roman numerals
const labels = ["0", "11", "12", "1:10"...] .+ roman_labels

# Text encoder / preprocess
const textenc = TransformerTextEncoder(split, labels; startsym="11", endsym="12", unksym="0", padsym="0")

function gen_data()
    pairs = [(string(i), roman_labels[i]) for i in 1:V]
    rand_pair = pairs[rand(1:length(pairs))]
    join([rand_pair for _ in 1:10], ' ')
end

# Model definition as before, but with an update to the output vocabulary and loss function to handle the new task.

function translate(x::AbstractString)
    ix = encode(textenc, x).token |> gpu
    seq = ["11"]

    encoder_input = (token = ix,)
    src = encoder(encoder_input)
    enc = src.hidden_state

    len = size(ix, 2)
    for i = 1:2len
        decoder_input = (token = gpu(lookup(textenc, seq)), memory = enc)
        trg = decoder(decoder_input)
        dec = trg.hidden_state
        logit = embed_decode(dec)
        ntok = decode(textenc, argmax(logit[:, end, :]))
        push!(seq, ntok)
        ntok == "12" && break
    end
    join(seq[2:end-1], ' ')
end

# Example usage
translate("5 6 7 8 9 10 1 2 3 4")

##Julia Version 1.8
##Author: Victor H. Aguiar
##Date: 2023-04-18
## based on https://chengchingwen.github.io/Transformers.jl/dev/tutorial/
##Description: This script is to train a simple tranformer to do a copy-paste task. The copy task is a toy test case of a sequence transduction problem that simply return the same sequence as the output. Here we define the input as a random sequence of white space separable number from 1~10 and length 10. we will also need a start and end symbol to indicate where is the begin and end of the sequence. We can use Transformers.TextEncoders.TransformerTextEncoder to preprocess the input (add start/end symbol, convert to one-hot encoding, ...).
using Pkg

# Create and activate a new environment
Pkg.activate("Transformer_env")
# Pkg.add("Flux")
# Pkg.add("Transformers")
# Pkg.add("TimeSeries")
# Pkg.add("Statistics")
# Pkg.add("MarketData")
# Pkg.add("TensorBoardLogger")
# Pkg.add("Logging")
# Pkg.add("BSON")
# Pkg.add("LinearAlgebra")
# Pkg.add("Plots")
#Pkg.add("Optimisers")
#Pkg.add("Zygote")
#Pkg.add("ChainRulesCore")
#Pkg.upgrade_manifest()

using Flux
using Transformers
using Transformers.Layers
using Transformers.TextEncoders

using Statistics
using Flux
using Flux.Losses
import Optimisers
using Zygote
using ChainRulesCore

using Transformers
using Transformers.Layers
using Transformers.TextEncoders
using Transformers.Datasets
## Transfomers are designed to deal with the sequence transduction problem 
# What is Sequence
# Transduction? Any task where input sequences are transformed into output sequences
# Copy task:
# The copy task is a toy test case of a sequence transduction problem that simply return the same sequence as the output. Here we define the input as a random sequence of white space separable number from 1~10 and length 10. we will also need a start and end symbol to indicate where is the begin and end of the sequence. We can use Transformers.TextEncoders.TransformerTextEncoder to preprocess the input (add start/end symbol, convert to one-hot encoding, ...).
# configuration

const N = 2
const V = 10
const Smooth = 1e-6
const Batch = 32
const lr = 1e-4

# text encoder / preprocess
const startsym = "11"
const endsym = "12"
const unksym = "0"
const labels = [unksym, startsym, endsym, map(string, 1:V)...]

const textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)

function gen_data()
    global V
    d = join(rand(1:V, 10), ' ')
    (d,d)
end

# model definition
const hidden_dim = 512
const head_num = 8
const head_dim = 64
const ffn_dim = 2048

const token_embed = todevice(Embed(hidden_dim, length(textenc.vocab); scale = inv(sqrt(hidden_dim))))
const embed = Layers.CompositeEmbedding(token = token_embed, pos = SinCosPositionEmbed(hidden_dim))
const embed_decode = EmbedDecoder(token_embed)
const encoder = todevice(Transformer(TransformerBlock       , N, head_num, hidden_dim, head_dim, ffn_dim))
const decoder = todevice(Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim))

const seq2seq = Seq2Seq(encoder, decoder)
const trf_model = Layers.Chain(
    Layers.Parallel{(:encoder_input, :decoder_input)}(
        Layers.Chain(embed, todevice(Dropout(0.1)))),
    seq2seq,
    Layers.Branch{(:logits,)}(embed_decode),
)

const opt_rule = Optimisers.Adam(lr)
const opt = Optimisers.setup(opt_rule, trf_model)

function preprocess(data)
    global textenc
    x, t = data
    input = encode(textenc, x, t)
    return todevice(input)
end

function smooth(et)
    global Smooth
    sm = fill!(similar(et, Float32), Smooth/length(textenc.vocab))
    p = sm .* (1 .+ -et)
    label = p .+ et .* (1 - convert(Float32, Smooth))
    return label
end

ChainRulesCore.@non_differentiable smooth(et)

function shift_decode_loss(logits, trg, trg_mask)
    label = @view smooth(trg)[:, 2:end, :]
    return logitcrossentropy(mean, @view(logits[:, 1:end-1, :]), label, trg_mask - 1)
end



function train!()
    global Batch, trf_model
    println("start training")
    for i = 1:320*7
        data = batched([gen_data() for i = 1:Batch])
        input = preprocess(data)
        decode_loss, (grad,) = Zygote.withgradient(trf_model) do model
            nt = model(input)
            shift_decode_loss(nt.logits, input.decoder_input.token, input.decoder_input.attention_mask)
        end
        i % 8 == 0 && @show decode_loss
        Optimisers.update!(opt, trf_model, grad)
    end
end

train!()


function translate(x::AbstractString)
    global textenc, embed, encoder, decoder, embed_decode, startsym, endsym
    ix = todevice(encode(textenc, x).token)
    seq = [startsym]

    encoder_input = (token = ix,)
    src = embed(encoder_input)
    enc = encoder(src).hidden_state

    len = size(ix, 2)
    for i = 1:2len
        decoder_input = (token = todevice(lookup(textenc, seq)), memory = enc)
        trg = embed(decoder_input)
        dec = decoder(trg).hidden_state
        logit = embed_decode(dec)
        ntok = decode(textenc, argmax(@view(logit[:, end])))
        push!(seq, ntok)
        ntok == endsym && break
    end
    seq
end

translate("5 6 7 8 9 10 1 2 3 4 5 6")

translate("1 1 1 1 1 1 1 1 1 1 1 1 1")

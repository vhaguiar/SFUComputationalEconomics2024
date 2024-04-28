
##Julia Version 1.8
##Author: Victor H. Aguiar
##Date: 2023-04-18
##Description: This script is to train a transformer to translate digits to Roman numerals.
using Pkg

# Create and activate a new environment
Pkg.activate("Transformer_env")
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

# Configuration
const N = 2
const V = 10
const Smooth = 1e-6
const Batch = 32
const lr = 1e-4
const roman_numerals = ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX", "X"]
const startsym = "11"
const endsym = "12"
const unksym = "0"
const labels = [unksym, startsym, endsym, roman_numerals..., map(string, 1:V)...]

# Encoder and model definitions
const textenc = TransformerTextEncoder(split, labels; startsym, endsym, unksym, padsym = unksym)
const hidden_dim = 512
const head_num = 8
const head_dim = 64
const ffn_dim = 2048
const token_embed = Embed(hidden_dim, length(textenc.vocab); scale = inv(sqrt(hidden_dim)))
const embed = Layers.CompositeEmbedding(token = token_embed, pos = SinCosPositionEmbed(hidden_dim))
const embed_decode = EmbedDecoder(token_embed)
const encoder = Transformer(TransformerBlock, N, head_num, hidden_dim, head_dim, ffn_dim)
const decoder = Transformer(TransformerDecoderBlock, N, head_num, hidden_dim, head_dim, ffn_dim)
const seq2seq = Seq2Seq(encoder, decoder)
const trf_model = Layers.Chain(
    Layers.Parallel{(:encoder_input, :decoder_input)}(Layers.Chain(embed)),
    seq2seq,
    Layers.Branch{(:logits,)}(embed_decode)
)
const opt_rule = Optimisers.Adam(lr)
const opt = Optimisers.setup(opt_rule, trf_model)

# Data generation
function gen_data()
    numbers = rand(1:V, 10)
    d = join(numbers, ' ')
    roman = join([roman_numerals[n] for n in numbers], ' ')
    (d, roman)
end

# Preprocessing
function preprocess(data)
    global textenc
    x, t = data
    # Encode input and target sequences separately.
    input = encode(textenc, x)
    target = encode(textenc, t)
    return (input, target)
end



# Loss function
function shift_decode_loss(logits, trg, trg_mask)
    label = smooth(trg)[:, 2:end, :]
    return logitcrossentropy(mean, logits[:, 1:end-1, :], label, trg_mask - 1)
end

using Flux: logitcrossentropy

function model_specific_loss_function(logits, targets)
    return logitcrossentropy(logits, targets, reduction = sum)
end

# Training loop
function train!()
    global Batch, trf_model, opt
    println("Start training")
    for i in 1:320*7
        batch_data = [gen_data() for _ in 1:Batch]
        processed_data = [(preprocess(d)...,) for d in batch_data]
        inputs = getindex.(processed_data, 1)
        targets = getindex.(processed_data, 2)

        # Assuming your model returns logits directly compatible with logitcrossentropy
        decode_loss, grads = Zygote.gradient(trf_model) do model
            model_loss = 0.0
            for (model_input, target) in zip(inputs, targets)
                model_output = model(model_input)  # model should return logits here
                model_loss += model_specific_loss_function(model_output, target)
            end
            model_loss
        end
        Optimisers.update!(opt, trf_model, grads)
    end
end




# Translation function
function translate(x::AbstractString)
    ix = encode(textenc, x).token
    seq = [startsym]

    encoder_input = (token = ix,)
    src = embed(encoder_input)
    enc = encoder(src).hidden_state

    len = size(ix, 2)
    for i = 1:2*len
        decoder_input = (token = lookup(textenc, seq), memory = enc)
        trg = embed(decoder_input)
        dec = decoder(trg).hidden_state
        logit = embed_decode(dec)
        ntok = decode(textenc, argmax(logit[:, end]))
        push!(seq, ntok)
        ntok == endsym && break
    end
    join(seq[2:end-1], ' ')
end

# Main execution
train!()
translate("5 6 7 8 9 10 1 2 3 4 5 6")
translate("1 1 1 1 1 1 1 1 1 1 1 1 1")



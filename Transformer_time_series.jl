##Julia Version 1.7.2
##Author: Victor H. Aguiar
##Date: 2023-04-18
##Description: This script is to train a simple tranformer to predict time series data.
using Pkg

# Create and activate a new environment
# Pkg.activate("Transformer_env")
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
#Pkg.upgrade_manifest()

using Flux
using Transformers
using Transformers.Layers
using Transformers.TextEncoders

ta = readtimearray("rate.csv", format="mm/dd/yy", delim=',')

function get_src_trg(
    sequence, 
    enc_seq_len, 
    dec_seq_len, 
    target_seq_len
)
    nseq = size(sequence)[2]
    
    @assert  nseq == enc_seq_len + target_seq_len
    src = sequence[:,1:enc_seq_len,:]
    trg = sequence[:,enc_seq_len:nseq-1,:]
    @assert size(trg)[2] == target_seq_len
    trg_y = sequence[:,nseq-target_seq_len+1:nseq,:]
    @assert size(trg_y)[2] == target_seq_len
    if size(trg_y)[1] == 1
        return src, trg, dropdims(trg_y; dims=1)
    else
        return src, trg, trg_y
    end
end


## Model parameters
dim_val = 512
n_heads = 8
n_decoder_layers = 4
n_encoder_layers = 4
input_size = 1
dec_seq_len = 92
enc_seq_len = 153
output_sequence_length = 58
in_features_encoder_linear_layer = 2048
in_features_decoder_linear_layer = 2048
max_seq_len = enc_seq_len



encode_t1 = Transformer(dim_val, n_heads, 64, 2048;future=false,pdrop=0.2)
encode_t2 = Transformer(dim_val, n_heads, 64, 2048;future=false,pdrop=0.2)
    decode_t1 = TransformerDecoder(dim_val, n_heads, 64, 2048,pdrop=0.2)
    decode_t2 = TransformerDecoder(dim_val, n_heads, 64, 2048,pdrop=0.2)
    encoder_input_layer = Dense(input_size,dim_val)
    decoder_input_layer = Dense(input_size,dim_val)
    positional_encoding_layer = PositionEmbedding(dim_val)
    p = 0.2
    dropout_pos_enc = Dropout(p)
    
    linear = Dense(output_sequence_length*dim_val,output_sequence_length)
    function encoder_forward(x)
        x = encoder_input_layer(x)
        e = positional_encoding_layer(x)
        t1 = x .+ e
        t1 = dropout_pos_enc(t1)
        t1 = encode_t1(t1)
        t1 = encode_t2(t1)
        return t1
    end
    
    function decoder_forward(tgt, t1)
        decoder_output = decoder_input_layer(tgt)
        t2 = decode_t1(decoder_output,t1)
        t2 = decode_t1(decoder_output,t2)
        t2 = Flux.flatten(t2)
        p = linear(t2)
        return p
    end
end

function generate_seq(x, seq_len)
    result = Matrix{Float64}[]
    for i in 1:length(x)-seq_len+1
        ele = reshape(x[i:i+seq_len-1],(seq_len,1))    
        push!(result,ele)
    end
    return result
end

#function loss(src, trg, trg

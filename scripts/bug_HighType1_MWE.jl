using MixedModels
using MixedModelsSim
using Random

function runSim(rng)
    dat = simdat_crossed(30,50 ,
        item_btwn = Dict("condition" => ["I", "II"]),
    )
    simMod = LinearMixedModel(@formula(dv~1+condition + (1+condition|subj) + (1+condition | item)), dat)

    σs = [[1. 1.], [0. 0.]]
    simMod = MixedModelsSim.update!(simMod,[create_re(x...) for x in σs]...)


    simMod = simulate!(rng, simMod, β = [0., 0.], σ = 1.)
    dat = DataFrame(dat)
    dat.y = simMod.y
    simMod_inst = LinearMixedModel(@formula( y~1+condition + (1+condition|subj)), dat)
    #    simMod.y .= rand(rng,length(simMod.y))
    fit!(simMod_inst)

    fit!(simMod)
    x1 = coeftable(simMod_inst).cols[4][1]
    x2 = coeftable(simMod).cols[4][1]
    return x1,x2
end
res = map(x->runSim(MersenneTwister(x)),1:100)
@show mean([r[1] for r in res].<0.05)
@show mean([r[2] for r in res].<0.05)


#-----
using DataFrames
using MixedModels
using MixedModelsSim
using ProgressMeter
using Random
using Statistics

function runSim(x; progress=false)
    rng = MersenneTwister(x)
    dat = simdat_crossed(30, 50;
                         item_btwn=Dict("condition" => ["I", "II"]),
    )
    dat = DataFrame(dat)
    imbalance = "subject"
    if imbalance == "trial"
        ix = findall(dat.condition .== "II")
        goodIx = [i  ∉ ix[randperm(length(ix))[Int.(round.(1:0.2*length(ix)))]] for i in 1:nrow(dat)]

    elseif imbalance == "subject"
        goodIx = Int[]
        uniqueSub = unique(dat.subj)
        for s = 1:length(uniqueSub)
            fractionToKeep = s*1/length(uniqueSub)
            ix = findall(dat.subj .== uniqueSub[s])
            #@show fractionToKeep
            goodIx_sub = ix[randperm(length(ix))[Int.(round.(1:fractionToKeep*length(ix)))]]
            #@show length(goodIx_sub)
            append!(goodIx,goodIx_sub)

        
        end
    else
        # nothing to do here
        goodIx = 1:nrow(dat)
    end
    dat = dat[goodIx,:]
    
    
    gen_mod = LinearMixedModel(@formula(dv~1+condition + (1+condition|subj) + (1+condition | item)), dat)

    # the ordering of the grouping terms isn't necessarily the same
    # as the ordering in the formula -- they're sorted by number of associated
    # BLUPs for computational reasns
    #σs = [[1. 1.], [0. 0.]]  # why are you creating row vectors?
    # because this is confusing, we support using keyword arguments
    σs =(;subj=[1., 1.], item=[0., 0.]) # these are now the variances
    re = NamedTuple{propertynames(σs)}(create_re(s...) for s in values(σs))

    gen_mod = MixedModelsSim.update!(gen_mod; re...)

    simulate!(rng, gen_mod; β = [0., 0.], σ = 1.)
    dat.y .= gen_mod.y
    alt_mod = fit(MixedModel, @formula(y ~ 1 + condition + (1+condition|subj)), dat; progress)
    fit!(gen_mod; progress)
        
    gen = coeftable(gen_mod).cols[4][2]
    alt = coeftable(alt_mod).cols[4][2]
    return (;gen, alt)
end

pos_is_sig(n) = tup -> tup[n] < 0.05


res = map(runSim,1:100)

mean(pos_is_sig(:gen), res)
mean(pos_is_sig(:alt), res)
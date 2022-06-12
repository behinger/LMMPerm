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
using Distributed
using MixedModelsSim, MixedModels,MixedModelsPermutations,StatsModels
using ProgressMeter
using SharedArrays
using BlockDiagonals,LinearAlgebra
using DataFrames

 sim_model_getData() =  sim_model_getData(30,30)
function sim_model_getData(nsub,nitem)
    subj_btwn = Dict("age" => ["O", "Y"])

    # there are no between-item factors in this design so you can omit it or set it to nothing
    item_btwn = Dict("stimType" => ["I", "II"])

    # put within-subject/item factors in a Dict
    both_win = Dict("condition" => ["A", "B"])

    # simulate data
    dat = simdat_crossed(
        nsub,
        nitem,
        subj_btwn = subj_btwn,
        item_btwn = item_btwn,
        both_win = both_win,
    )
    return dat

end
function sim_model(f;simulationCoding=DummyCoding)
    dat = sim_model_getData()
    simMod = MixedModels.fit(MixedModel, f, dat,contrasts=Dict(:age=>simulationCoding(),:stimType=>simulationCoding(),:condition=>simulationCoding()),)

    return simMod

end
function run_permutationtest_distributed(n_workers, nRep, simMod,args...)
    
    if nworkers() < n_workers
        # open as many as necessary
        println("Starting Workers, this might take some time")
        addprocs(
            n_workers - nworkers() + 1,
            exeflags = "--project",
            enable_threaded_blas = true,
        )
    end
    
    # activate environment
    eval(macroexpand(Distributed, quote
        @everywhere using Pkg
    end))

    @everywhere Pkg.activate(".")
    # load packages on distributed
    eval(
        macroexpand(
            Distributed,
            quote
                @everywhere using MixedModelsSim,
                    Random, MixedModels, MixedModelsPermutations
            end,
        ),
    )
    β_permResult = SharedArray{Float64}(nRep, length(coef(simMod)))
    z_permResult = SharedArray{Float64}(nRep, length(coef(simMod)))
    
    @everywhere include("src/sim_utilities.jl")
    #b = srcdir("sim_utilities.jl")
    @everywhere include("src/permutationtest_be.jl")
    
    println("starting @distributed")
    println("Note: If nothing is starting, this is likely due to an error which will just freeze everything. Test it locally!")
    # parallel loop
    @showprogress @distributed for k = 1:nRep
        #println("Thread "*string(Threads.threadid()) * "\t Running "*string(k))
        #res = [1,1.]#
        res = run_permutationtest(MersenneTwister(k), deepcopy(simMod),args...)
        
        β_permResult[k, :] .= res[1]
        z_permResult[k, :] .= res[2]

    end
    return β_permResult, z_permResult

end

# add the last one as optional - hope that works :-D
run_permutationtest(args...) = run_permutationtest(args...,DummyCoding())

function run_permutationtest(rng, simMod, nPerm, β, σ, sigmas,residual_method,blup_method,analysisCoding,f)

    simMod = MixedModelsSim.update!(simMod,sigmas...)

    simMod = simulate!(rng, simMod, β = β, σ = σ)
    dat = sim_model_getData() |> x-> DataFrame(x)
    dat.dv = simMod.y
    simMod2 = MixedModels.fit(MixedModel,f ,dat,contrasts=Dict(:age=>analysisCoding(),:stimType=>analysisCoding(),:condition=>analysisCoding()))
    H0 = coef(simMod2)
    H0[2] = 0.0

    perm = permutation(rng, nPerm, simMod2, use_threads = false; β = H0,residual_method=residual_method,blup_method=blup_method)
    p_β = values(permutationtest_be(perm, simMod2; statistic = :β))
    p_z = values(permutationtest_be(perm, simMod2; statistic = :z))

    return (p_β, p_z)
end


function fitsignal(formula, data, signal, contrasts)
    # fit the MixedModel

    fits = Array{Any}(undef, size(signal)[2])
    model = Array{Any}(undef,1)
    cdata = copy(data)

    for i = 1:(size(signal)[2])
        println(i)
        if i==1
            cdata[:,formula.lhs.sym] = (signal[:,i])
            model[1] = MixedModels.fit(MixedModel, formula, cdata, contrasts = contrasts)
        else
            model[1] = refit!(model[1],signal[:,i])
        end
        fits[i] = deepcopy(model[1])
    end
    return fits
end




function circulant(x)
    # returns a symmetric matrix where X was circ-shifted.
    lx = length(x)
    ids = [1:1:(lx-1);]
    a = Array{Float64,2}(undef, lx,lx)
    for i = 1:length(x)
        if i==1
            a[i,:] = x
        else
            a[i,:] = vcat(x[i],a[i-1,ids])
        end
    end
    return Symmetric(a)
end



function exponentialCorrelation(x; nu = 1, length_ratio = 1)
    # generate exponential function
    R = length(x) * length_ratio
    return exp.(-3 * (x / R) .^ nu)
end

function expandgrid(df1, df2)
    # get all combinations of df1&df2

    a = Array{Any}(undef, nrow(df1))
    for i = 1:nrow(df1)
        a[i] = hcat(repeat(df1[[i], :], nrow(df2)), df2)
    end
    return reduce(vcat, a)
end

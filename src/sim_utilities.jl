using Distributed
using MixedModelsSim, MixedModels,MixedModelsPermutations,StatsModels
using ProgressMeter
using SharedArrays
using BlockDiagonals,LinearAlgebra
using DataFrames
using StatsBase

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
function run_test_distributed(n_workers,simMod;nRep = missing,kwargs...)
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
                @everywhere using DrWatson, MixedModelsSim,
                    Random, MixedModels, MixedModelsPermutations
            end,
        ),
    )
    statResult1 = SharedArray{Float64}(nRep, length(coef(simMod)))
    statResult2 = SharedArray{Float64}(nRep, length(coef(simMod)))
    @everywhere @quickactivate "LMMPerm"
    @everywhere include(srcdir("sim_utilities.jl"))
    @everywhere include(srcdir("permutationtest_be.jl"))
    
    println("starting @distributed")
    println("Note: If nothing is starting, this is likely due to an error which will just freeze everything. Test it locally!")
    # parallel loop
    @showprogress @distributed for k = 1:nRep
        #println("Thread "*string(Threads.threadid()) * "\t Running "*string(k))
        res = run_test(MersenneTwister(5000+k), deepcopy(simMod);kwargs...)



        

        if typeof(res) <: NamedTuple
            val = (res[Symbol("(Intercept)")],res[Symbol("condition: B")])
            statResult1[k, :] .= val    
            statResult2[k, :] .= -1
        else
            statResult1[k, :]  = (res[1][Symbol("(Intercept)")],res[1][Symbol("condition: B")])
            statResult2[k, :]  = (res[2][Symbol("(Intercept)")],res[2][Symbol("condition: B")])
        
        end
    end
    return statResult1, statResult2

end

# add the last one as optional - hope that works :-D
#run_permutationtest(args...) = run_permutationtest(args...,DummyCoding()) # this looks dangerous, but I should rewrite everything anyway...
function setup_simMod(rng,simMod; f = missing, β=missing,σ=1,σs=missing,  analysisCoding = DummyCoding,kwargs...)
    @assert all(.!ismissing.([f,β,σs]))
    simMod = MixedModelsSim.update!(simMod,[create_re(x...) for x in σs]...)

    simMod = simulate!(rng, simMod, β = β, σ = σ)
    dat = sim_model_getData() |> x-> DataFrame(x)
    dat.dv = simMod.y
    simMod_inst = MixedModels.fit(MixedModel,f ,dat,contrasts=Dict(:age=>analysisCoding(),:stimType=>analysisCoding(),:condition=>analysisCoding()))
    simMod_inst.optsum.maxtime = 0.5 # restrict per-iteration fitting time
    return simMod_inst
end

function run_test(rng,simMod;statsMethod="permutation", kwargs...)

    simMod_instantiated = setup_simMod(rng,simMod;kwargs...)
    
    if statsMethod == "permutation"
        run_fun = run_permutationtest
    elseif statsMethod == "LRT"
        run_fun = run_LRT
    elseif statsMethod == "waldsT"
           run_fun=  run_waldsT
    elseif statsMethod == "pBoot"
            run_fun = run_pBoot
    else
            error("not implemented")
    end
    
    res = run_fun(rng,simMod_instantiated;kwargs...)     
    
    return res
    
end
function run_pBoot(rng,simMod_instantiated;nBoot = 1000,kwargs...)
    bootRes = parametricbootstrap(rng,nBoot,simMod_instantiated) # bootstrap
    covRes = DataFrame(shortestcovint(bootRes)) # get 95 convint 
    ci95 = covRes[(covRes.type.== "β"),[:names,:lower,:upper] ] # get the right parameter
    significant =  sign.(ci95.lower) .== sign.(ci95.upper) # check if sign equal, if yes, we have significance

    return (;(Symbol.(ci95.names) .=> significant)...)
end

function run_waldsT(rng,simMod_instantiated;kwargs...)
    x = coeftable(simMod_instantiated)
    pvals =  x.cols[x.pvalcol]
    return (;(Symbol.(x.rownms) .=> pvals)...) # we can report two p-vals, might be changed
end

function run_LRT(rng,simMod_instantiated;kwargs...)
    error("not implemented")
    simMod_instantiated

end

function run_permutationtest(rng,simMod_instantiated;nPerm=1000,residual_permutation=:shuffle,blup_method= "ranef",  kwargs...)
    blup_method = getfield(Main,Meta.parse(blup_method))

    H0 = coef(simMod_instantiated)
    H0[2] = 0.0
    
    if first(methods(blup_method)).nargs == 1 # super hacky, but we need this for ranef_covInflation
        perm = permutation(rng, nPerm, simMod_instantiated, use_threads = false;
         β = H0,residual_permutation=residual_permutation,blup_method=ranef,inflation_method=inflation_method_cov)
        
    else
        perm = permutation(rng, nPerm, simMod_instantiated, use_threads = false; β = H0,residual_permutation=residual_permutation,blup_method=blup_method)
    end
    p_β = permutationtest_be(perm, simMod_instantiated; statistic = :β)
    p_z = permutationtest_be(perm, simMod_instantiated; statistic = :z)

    return (p_β, p_z)

   end

#--------------- Functions --------------------

function inflation_method_cov(m::LinearMixedModel, blups=ranef(m), resids=residuals(m))    
        σ = sdest(m)
        σres = std(resids; corrected=false)
          inflation = map(zip(m.reterms, blups)) do (trm, re)
            # inflation
            λmle =  trm.λ * σ                              # L_R in CGR
    
            cov_emp = StatsBase.cov(re'; corrected=false)
                    
            chol = cholesky(cov_emp, Val(true); check=false,tol=10^-5)
    
            #  ATTEMPT 2
             while chol.rank != size(cov_emp, 1)
                 #@info "rep"
                idx = chol.p[(chol.rank+1):end]
                cov_emp[idx, idx] .+= 1e-6
                chol = cholesky(cov_emp, Val(true); check=false,tol=10^-5)
            end
            
            L = chol.L[invperm(chol.p),:]
            cov_emp = L * L'
            cov_mle = λmle * λmle'
            
            return cov_mle / cov_emp
        end
        return [inflation; σ / σres]
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
println("loaded sim_utilities")
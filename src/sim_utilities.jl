using MixedModels: likelihoodratiotest
using Distributed
using MixedModelsSim, MixedModels,MixedModelsPermutations,StatsModels
using ProgressMeter
using SharedArrays
using BlockDiagonals,LinearAlgebra
using DataFrames
using StatsBase
using Distributions
using SlurmClusterManager
using Logging

ENV["LMER"] = "lmerTest::lmer"
ENV["LMER"] = "afex::lmer_alt"
using JellyMe4
using RCall

include(srcdir("sim_parameters.jl"))
include(srcdir("permutationtest_be.jl"))
convertDict = x-> [Symbol(d.first) => d.second for d in x]

function sim_model_getData(;nSubject=missing,nItemsPerCondition=missing,imbalance=nothing,kwargs...)
    subj_btwn = Dict("age" => ["O", "Y"])

    # there are no between-item factors in this design so you can omit it or set it to nothing
    item_btwn = Dict("stimType" => ["I", "II"])

    # put within-subject/item factors in a Dict
    both_win = Dict("condition" => ["A", "B"])
    
    # simulate data 
    dat = simdat_crossed(
        nSubject,
        nItemsPerCondition,
        subj_btwn = subj_btwn,
        item_btwn = item_btwn,
        both_win = both_win,
    )
    dat = DataFrame(dat)
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
    elseif isnothing(imbalance) | ismissing(imbalance)
        # nothing to do here
        goodIx = 1:nrow(dat)
    else
        @error("did you misswrite trial/subject?")
    end
    dat = dat[goodIx,:]
    
    

    return dat

end
function sim_model(f;simulationCoding=DummyCoding,kwargs...)
       dat = sim_model_getData(;kwargs...)
       simMod = LinearMixedModel(f, dat; contrasts=Dict(:age=>simulationCoding(),:stimType=>simulationCoding(),:condition=>simulationCoding()))
       simMod.optsum.maxtime = 10 # restrict per-iteration fitting time
       simMod.optsum.maxfeval = 20000
   
    fit!(simMod)

    return simMod

end
function run_test_distributed(n_workers,simMod;nRep = missing,onesided=true,reml=false,kwargs...)
    if n_workers == "slurm"
        # open as many as necessary
        println("Starting Slurmm workers, this might take some time")
        addprocs(
            SlurmManager(),
            exeflags = "--project",
            #enable_threaded_blas = true, # not sure SlurmClusterManager supports this
        )
    elseif nworkers() < n_workers
            # open as many as necessary
            println("Starting workers, this might take some time")
            addprocs(
                n_workers - nworkers() + 1,
                exeflags = "--project",
                enable_threaded_blas = true,
            )
    end
    println("worker started")    
    
    @everywhere using Pkg
    @everywhere Pkg.activate(".")
    @everywhere using DrWatson
    @everywhere @quickactivate "LMMPerm"
    @everywhere using MixedModelsSim
    @everywhere using Random
    @everywhere using MixedModels
    @everywhere using MixedModelsPermutations


    statResult1 = SharedArray{Float64}(nRep, length(coef(simMod)), (onesided ? 3 : 1)) # if onesided testing is activated, we get twosided + two onesided results
    statResult2 = SharedArray{Float64}(nRep, length(coef(simMod)), (onesided ? 3 : 1))


    time = SharedArray{Float64}(nRep)
    
    @everywhere include(srcdir("sim_utilities.jl");)
    @everywhere include(srcdir("permutationtest_be.jl");)
    
    println("starting @distributed")
    println("Note: If nothing is starting, this is likely due to an error which will just freeze everything. Test it locally!")
    # parallel loop
    #@showprogress 
    simMod = deepcopy(simMod)
    if reml
        refit!(simMod;REML=true) # needed for KenwardRoger
    else
        
    end
    @sync @distributed for k = 1:nRep
        #println("Thread "*string(Threads.threadid()) * "\t Running "*string(k))
        time[k] = @elapsed begin
            res = run_test(MersenneTwister(k), deepcopy(simMod);onesided=onesided,kwargs...)
        end

        


        

        if typeof(res) <: NamedTuple
            val = vcat(res[Symbol("(Intercept)")],res[Symbol("condition: B")])
            statResult1[k, :,:] .= val    
            statResult2[k, :,:] .= -1
        else 
            # from permutation test we get a tuple of namedTuples, one for \beta (first) and one for z-test (second)
            statResult1[k, :,:]  .= vcat(res[1][Symbol("(Intercept)")],res[1][Symbol("condition: B")])
            statResult2[k, :,:]  .= vcat(res[2][Symbol("(Intercept)")],res[2][Symbol("condition: B")])
        
        end

    end

    println("workers finished")
    #unpack
    
    
    df = DataFrame()
    labels = [:twosided, :lesser,:greater]
    for k = 1: (onesided ? 3 : 1)
        df = vcat(df,vcat(
            DataFrame(Dict(:pval => statResult1[:,1,k],:coefname=>repeat(["(Intercept)"],nRep),:test=>"default",:seed => (1:nRep),:side=>labels[k],:runtime=>time)),
            DataFrame(Dict(:pval => statResult1[:,2,k],:coefname=>repeat(["condition: B"],nRep),:test=>"default",:seed => (1:nRep),:side=>labels[k],:runtime=>time)),
            DataFrame(Dict(:pval => statResult2[:,1,k],:coefname=>repeat(["(Intercept)"],nRep),:test=>"default2",:seed => (1:nRep),:side=>labels[k],:runtime=>time)),
            DataFrame(Dict(:pval => statResult2[:,2,k],:coefname=>repeat(["condition: B"],nRep),:test=>"default2",:seed => (1:nRep),:side=>labels[k],:runtime=>time)))
        )
    end
    
    # we use -1 to identify the permutation case
    if statResult2[1,1,1] !=-1.
        # permutation
        df.test[df.test .== "default"] .= "β"
        df.test[df.test .== "default2"] .= "z"
    end

    
    # in the for loop above we create more values than we needed (except for permutation test) thus we remove them here again.
    println("exiting function")
    return df[df.pval .!= -1,:]

end

# add the last one as optional - hope that works :-D
#run_permutationtest(args...) = run_permutationtest(args...,DummyCoding()) # this looks dangerous, but I should rewrite everything anyway...
function setup_simMod(rng,simMod; f = missing, β=missing,σ=1,σs=missing,  analysisCoding = DummyCoding,errorDistribution="normal",kwargs...)
    @assert all(.!ismissing.([f,β,σs]))

    if errorDistribution != "normal" 
        #deactivate normal noise
        σ_org = σ    
        σ = 0.0001
        σs = σs ./ σ
    end

    σs =(;subj=σs[1], item=σs[2]) # these are now the variances
    re = NamedTuple{propertynames(σs)}(create_re(s...) for s in values(σs))

    simMod = MixedModelsSim.update!(simMod;re...)


    simMod = simulate!(rng, simMod, β = β, σ = σ)
    dat = sim_model_getData(;kwargs...) |> x-> DataFrame(x)
    # add noise
    y = simMod.y;
   
    if errorDistribution == "tdist"
        
        y = y .+ (rand(rng,TDist(3),length(y)) .* σ_org)

    elseif errorDistribution == "normal"
        
        #"nothing needs to happen"
    elseif errorDistribution == "skewed"
        snorm = SkewNormal(0,σ_org,10)
        
        y = y .+ rand(rng,snorm-mean(snorm),length(y)) # parameterisation location != mean, thus remove (theoretical) mean
    elseif errorDistribution == "skewed_40"
        snorm = SkewNormal(0,σ_org,40)
        
        y = y .+ rand(rng,snorm-mean(snorm),length(y)) # parameterisation location != mean, thus remove (theoretical) mean
        
    
    else
        @error "not implemented error function"
    end
    
    dat.dv .= y

    simMod_inst = LinearMixedModel(f, dat; contrasts=Dict(:age=>analysisCoding(),:stimType=>analysisCoding(),:condition=>analysisCoding()))
    simMod_inst.optsum.maxtime = 0.5 # restrict per-iteration fitting time
    simMod_inst.optsum.maxfeval = 10000

    fit!(simMod_inst)

    return simMod_inst
end

function run_test(rng,simMod;statsMethod="permutation",kwargs...)
    
    simMod_instantiated = setup_simMod(rng,simMod;kwargs...)
    
    if statsMethod == "permutation"
        run_fun = run_permutationtest
    elseif statsMethod == "LRT"
        run_fun = run_LRT
    elseif statsMethod == "waldsT"
           run_fun=  run_waldsT
    elseif statsMethod == "pBoot"
            run_fun = run_pBoot
    elseif statsMethod == "KenwardRoger"
        run_fun = run_kr
    else
            error("not implemented")
    end
    
    res = run_fun(rng,simMod_instantiated;kwargs...)     
    
    return res
    
end


function run_kr(rng,simMod_instantiated;onesided=false,kwargs...)
    dat = sim_model_getData(;kwargs...)
    # convert with JellyMe4
    refit!(simMod_instantiated;REML=true) # needed for KenwardRoger
    dat.dv = simMod_instantiated.y
    lme4_r = (simMod_instantiated,dat)
    with_logger(NullLogger()) do
        @rput lme4_r;
        R"""
        library(lmerTest)
        #lme4_r = as(lme4_r, "merModLmerTest") # not necessary due to global ENV variable
        sum_res = summary(lme4_r,ddf='Kenward-Roger')$coefficients
        rnames = rownames(sum_res)
        """
    end
    @rget rnames
    @rget sum_res

    rnames[rnames.=="conditionB"] .= "condition: B"
    res = (;((Symbol(k)=>v) for (k,v) in zip(rnames,sum_res[:,5]))...)
    if onesided

        p_pos = sum_res[:,5]./2
        p_neg = sum_res[:,5]./2

        p_neg[sum_res[:,1].<0] .= 1
        p_pos[sum_res[:,1].>0] .= 1
        
        res = (;(k=>[v neg pos] for (k,v,neg,pos) in zip(keys(res),values(res),p_neg,p_pos))...)

    end
    
    #@show res
    return res
end



function run_pBoot(rng,simMod_instantiated;nBoot = nothing,onesided=false,kwargs...)
    if isnothing(nBoot)
        nBoot = kwargs[:nPerm]
    end
    bootRes = parametricbootstrap(rng,nBoot,simMod_instantiated) # bootstrap
    covRes = DataFrame(shortestcovint(bootRes)) # get 95 convint 
    ci95 = covRes[(covRes.type.== "β"),[:names,:lower,:upper] ] # get the right parameter
    significant =  sign.(ci95.lower) .== sign.(ci95.upper) # check if sign equal, if yes, we have significance

    res = (;(Symbol.(ci95.names) .=> significant)...)
    if onesided
        covRes90 = DataFrame(shortestcovint(bootRes,0.90)) # get 90 convint for one sided testing
        ci90 = covRes90[(covRes90.type.== "β"),[:names,:lower,:upper] ] # get the right parameter
        

        sig_high =  ci90.lower .> 0 # e.g. [0.3 0.5]
        sig_low =   ci90.upper .< 0 # e.g. [-1.3, -0.7]

        # concatenate
        res = (;(k=>[v low high] for (k,v,high,low) in zip(keys(res),values(res),sig_high,sig_low))...)
    end
    return res
end

function run_waldsT(rng,simMod_instantiated;onesided=false,kwargs...)
    x = coeftable(simMod_instantiated)
    pvals =  x.cols[x.pvalcol]
    res = (;(Symbol.(x.rownms) .=> pvals)...)

    if onesided
        z_signs =  sign.(x.cols[x.teststatcol])
        flipSign = (x,y) -> x>0. ? 1-y : y

        sig_low = flipSign.(z_signs,pvals ./ 2)
        sig_high = flipSign.(.-z_signs,pvals ./ 2)
        # concatenate
        res = (;(k=>[v low high] for (k,v,high,low) in zip(keys(res),values(res),sig_high,sig_low))...)
    end
    return  res # we can report two p-vals, might be changed
end

function run_LRT(rng,simMod_instantiated;onesided=false,analysisCoding=DummyCoding,kwargs...)
    
    # we have to simplify the formula by removing the  condition effect from the formula
    f = deepcopy(simMod_instantiated.formula)
    f_fixef = f.rhs[1]
    @assert coefnames(f_fixef.terms[2]) == ["condition: B"]
    @assert length(f_fixef.terms) ==2

    f_reduced = FormulaTerm(f.lhs,(MatrixTerm(f_fixef.terms[1]),f.rhs[2]))
    
    dat = sim_model_getData(;kwargs...)
    #dat = sim_model_getData(;convertDict(dl)...)
    dat.dv .= simMod_instantiated.y

    simMod_reduced = LinearMixedModel(f_reduced, dat; contrasts=Dict(:age=>analysisCoding(),:stimType=>analysisCoding(),:condition=>analysisCoding()))
    simMod_reduced.optsum.maxtime = 10 # restrict per-iteration fitting time
    simMod_reduced.optsum.maxfeval = 20000

    fit!(simMod_reduced)

    res = MixedModels.likelihoodratiotest(simMod_reduced,simMod_instantiated)
    return (;Symbol("(Intercept)")=>NaN,Symbol("condition: B")=>res.pvalues[1])

    
end

function run_permutationtest(rng,simMod_instantiated;nPerm=missing,residualMethod=missing,blupMethod=missing,  inflationMethod=missing,residuals=residuals,onesided=false,kwargs...)

    
    if typeof(blupMethod) <: String
        blupMethod = getfield(Main,Meta.parse(blupMethod))
    end


    if typeof(inflationMethod) <:String
        if inflationMethod == "fixRankDeficient=False"
            inflationMethod = (x,y,z) -> MixedModelsPermutations.inflation_factor(x,y,z;fixRankDeficient=false)
        elseif inflationMethod == "noScaling"
            
            inflationMethod = (m,blups,resids)->[I ; sdest(m) ./std(resids; corrected=false)]#(length(coef()))
        
        else
            inflationMethod = getfield(Main,Meta.parse(inflationMethod))
        end
    end
      

    H0 = coef(simMod_instantiated)
    H0[2] = 0.0
    
    perm = MixedModelsPermutations.permutation(rng, nPerm, simMod_instantiated, use_threads = false;
         β = H0,
         residual_permutation=residualMethod,
         residual_method= residuals,
         blup_method=blupMethod,
         inflation_method=inflationMethod)
        

    p_β = permutationtest_be(perm, simMod_instantiated; statistic = :β)
    p_z = permutationtest_be(perm, simMod_instantiated; statistic = :z)

    res = (p_β, p_z)
    if onesided
        p_β_greater = permutationtest_be(perm, simMod_instantiated; statistic = :β,type=:greater)
        p_β_lesser = permutationtest_be(perm, simMod_instantiated; statistic = :β,type=:lesser)
        p_z_greater = permutationtest_be(perm, simMod_instantiated; statistic = :z,type=:greater)
        p_z_lesser = permutationtest_be(perm, simMod_instantiated; statistic = :z,type=:lesser)
        
        resA = NamedTuple()
        resB = NamedTuple()
        # concatenate
        for k in keys(res[1])
            
        resA = merge(resA,(;k => hcat(res[1][k],p_β_lesser[k],p_β_greater[k])))
        resB = merge(resB,(;k => hcat(res[2][k],p_z_lesser[k],p_z_greater[k])))
        end
        res = (resA,resB)
        
    end

    return res

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

function dl_filename(dl)
    dl_save =deepcopy(dl)
    dl_save["f"]  = string(dl_save["f"].rhs)|>x->replace(x," "=>"") # rename formula


    if "residualMethod" ∈ keys(dl_save)
        dl_save["residualMethod"]  = string(dl_save["residualMethod"])
    end

    if "blupMethod" ∈ keys(dl_save)
        dl_save["blupMethod"]  = string(dl_save["blupMethod"])
    end

    fnName = datadir("cluster", string(hash(savename("type1",dl_save, "jld2",allowedtypes=(Array,Float64,Integer,String,DataType,))))*".jld2")

    return fnName,dl_save
end



function dl_filename_old(dl)
    dl_save =deepcopy(dl)
    dl_save["f"]  = string(dl_save["f"].rhs)|>x->replace(x," "=>"") # rename formula


    if "residualMethod" ∈ keys(dl_save)
        dl_save["residualMethod"]  = string(dl_save["residualMethod"])
    end
    fnName = datadir("cluster", string(hash(savename("type1",dl_save, "jld2",allowedtypes=(Array,Float64,Integer,String,DataType,))))*".jld2")

    return fnName,dl_save
end

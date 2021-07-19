using DrWatson
@quickactivate "LMMPerm"
using CSV,Printf,DataFrames,StatsModels,SharedArrays,Random,Unfold
include("sim_utilities.jl")
include("permutationtest_be.jl")
β_permResult = SharedArray{Float64}(4000, 2)
z_permResult = SharedArray{Float64}(4000, 2)

flagForOrFromJaromil = "for"
@showprogress for k = 1:1000
    println("here_we_go: ",k)
    if flagForOrFromJaromil == "for"
        d = CSV.read(datadir("exportForJaromil","sim_"*string(k)*".csv"),DataFrame)

        f =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))

    else
        d = CSV.read(datadir("jaromil",@sprintf("sim%04i.csv",k)),DataFrame,decimal=',')
        f =  @formula(yzcp ~ 1 + fw  + zerocorr(1+fw|part))
    end
    simMod = MixedModels.fit(MixedModel, f, d,contrasts=Dict(:fw=>EffectsCoding()))

    H0 = coef(simMod)
    H0[2] = 0.0

    perm = permutation(MersenneTwister(1), 1001, simMod, use_threads = false; β = H0,residual_method=:signflip,blup_method=Unfold.olsranefjf)
    p_β = values(permutationtest_be(perm, simMod; statistic = :β))
    p_z = values(permutationtest_be(perm, simMod; statistic = :z))
    β_permResult[k, :] .= p_β
    z_permResult[k, :] .= p_z
end
    

#c[!,"<0.05"] = [sum( r.<=0.05)/r.nRep for r in eachrow(c)]

mean(z_permResult[1:1000,:].<0.05,dims=1)
##--
using StatsBase
coef_permResult = SharedArray{Float64}(4000, 2)
@showprogress for k = 1:1000#2000:4000
    println(datadir("jaromil",@sprintf("sim%04i.csv",k)))
    if flagForOrFromJaromil == "for"
        d = CSV.read(datadir("exportForJaromil","sim_"*string(k)*".csv"),DataFrame)

        f =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))

    else
    d = CSV.read(datadir("jaromil",@sprintf("sim%04i.csv",k)),DataFrame,decimal=',')

    f =  @formula(yzcp ~ 1 + fw  + zerocorr(1+fw|part))
    end
    simMod = MixedModels.fit(MixedModel, f, d,contrasts=Dict(:fw=>EffectsCoding()))
    coef_permResult[k,:] .= values(simMod.sigmas[:subj])
end


for ui = u
    if sum( m .== ui)==1
        println(ui)
    end
end
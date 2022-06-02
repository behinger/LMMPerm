#!/home/st/st_us-051950/st_ac136984/julia-1.7.3/bin/julia
#SBATCH --cpus-per-task 40
# :: SBATCH --mem-per-cpu
#SBATCH --nodes 1 
#SBATCH -o slurmm/%x-%j.out
#SBATCH --job-name=LMMPerm
#SBATCH --time 10:0:0 





using DrWatson
quickactivate(pwd(),"LMMPerm")

using Random,TimerOutputs
include(srcdir("sim_utilities.jl"))
include(srcdir("permutationtest_be.jl"))

convertDict = x-> [Symbol(d.first) => d.second for d in x]
f1 =  @formula(dv ~ 1 + condition  + (1|subj))
f2 =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))
f3 =  @formula(dv ~ 1 + condition  + (1+condition|subj))
f4 =  @formula(dv ~ 1 + condition  + (1+condition|subj) + (1+condition|item))

#---h0 tests
paramList = Dict(
    "f" => [f1,f3,f4],
    "σs" => [@onlyif("f"!= f4, [[1., 0.], [0.,0.]]),
             @onlyif("f"!= f4, [[1., 1.], [0.,0.]]),  
             @onlyif("f"!= f4, [[1., 4.], [0.,0.]]),
             @onlyif("f"!= f4, [[4., 1.], [0.,0.]]),

             @onlyif("f"== f4, [[1., 1.], [1., 0.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 4.]])],
    "σ" => 1.,
    "β" => [[0., 0.]],
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "inflationMethod" => [MixedModelsPermutations.inflation_factor,"noScaling"],
    "residualMethod" => [:signflip,:shuffle],#[:signflip,:shuffle],"
    "nRep" => 5000,
    "nPerm"=> 1000,

)

#----
# H1 test
paramList = Dict(
    "f" => [f1,f3,f4],
    "σs" => [@onlyif("f"== f1, [[1., 0.], [0.,0.]]),
             @onlyif("f"== f3, [[1., 1.], [0.,0.]]),  
             @onlyif("f"== f3, [[1., 4.], [0.,0.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
             ],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.1],[0., 1.]],
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "inflationMethod" => [MixedModelsPermutations.inflation_factor,"noScaling"],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "nRep" => 5000,
    "nPerm"=> 1000,

)

#----
# Power calculations
paramList = Dict(
    "statsMethod" => ["waldsT","pBoot","permutation"], # if this is "missing" we run permutation for backward compatibility
    "errorDistribution" => ["normal","tdist"],
    "f" => [f3],
    "σs" => [[[1., 1.], [0.,0.]]],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.1],[0., 0.2],[0., .3],[0., 0.5]],
    "nRep" => 5000,
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],
    "nPerm"=> 1000,
)

#-----
# Varying N
paramList = Dict(
    "statsMethod" => ["waldsT","pBoot","permutation"], # if this is "missing" we run permutation for backward compatibility
    "errorDistribution" => ["normal","tdist"],
    "f" => [f3],
    "σs" => [[[1., 1.], [0.,0.]]],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.3]],
    "nRep" => 5000,
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],
    "nSubject" => [10],
    "nItemsPerCondition" => [2,10],
    "nPerm"=> 1000,
)

##---
include(srcdir("sim_utilities.jl"))



dl = dict_list(paramList)[7]
simMod = sim_model(f4;convertDict(dl)...)
dl["nPerm"] = 10
res = run_test(MersenneTwister(5),simMod; convertDict(dl)...)
##---
include(srcdir("sim_utilities.jl"))

@time begin
nWorkers=40
for dl = dict_list(paramList)
    println(dl)
    
    dl_save =deepcopy(dl)
    dl_save["f"]  = string(dl_save["f"].rhs)|>x->replace(x," "=>"") # rename formula


    if "residualMethod" ∈ keys(dl_save)
        dl_save["residualMethod"]  = string(dl_save["residualMethod"])
    end

    fnName = datadir("22-05_sim", savename("type1",dl_save, "jld2",allowedtypes=(Array,Float64,Integer,String,DataType,)))
    if isfile(fnName)
        # don't calculate again
        continue
    end

    simMod = sim_model(f4)

    t = @elapsed begin
        res = run_test_distributed(nWorkers,simMod;convertDict(dl)...)
    end
    #@warn " \beta is actual res[1]!!"
    dl_save["results"] = res
    dl_save["runtime"] = t
    @tagsave(fnName, dl_save)
end
end


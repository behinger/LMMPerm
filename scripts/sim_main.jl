#!/home/st/st_us-051950/st_ac136984/julia-1.7.3/bin/julia
#SBATCH --cpus-per-task 80
#SBATCH --mem-per-cpu 1500
#SBATCH --nodes 1 
#SBATCH -o slurmm/%x-%j.out
#SBATCH --job-name=LMMPerm
#SBATCH --time 20:0:0 




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

try
	global task = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
catch KeyError

    global task = 5

end
@show task
#---h0 tests
if task == 1
paramList = Dict(
    "f" => [f1,f3,f4],
    "σs" => [@onlyif("f"!= f4, [[1., 0.],[0.,0.]]),
             @onlyif("f"!= f4, [[1., 1.],[0.,0.]]),  
             @onlyif("f"!= f4, [[1., 4.],[0.,0.]]),
             @onlyif("f"!= f4, [[4., 1.],[0.,0.]]),

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
    "nSubject" => [30],
    "nItemsPerCondition" => [30],
    
)
elseif task == 2
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
    "nSubject" => [30],
    "nItemsPerCondition" => [30],

)

elseif task == 3
#----
# Power calculations
paramList = Dict(
    "statsMethod" => ["waldsT","pBoot","permutation"], # if this is "missing" we run permutation for backward compatibility
    "errorDistribution" => ["normal"],#"tdist"],
    "f" => [f3],
    "σs" => [[[1., 1.],[0.,0.]]],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.1],[0., 0.2],[0., .3],[0., 0.5]],
    "nRep" => 5000,
    "blupMethod" => [ranef,olsranef],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],
    "nSubject" => [30],
    "nItemsPerCondition" => [30],
    "nPerm"=> 1000,
)


elseif task == 4
#-----
# Varying N
paramList = Dict(
    "statsMethod" => ["waldsT","pBoot","permutation"], # if this is "missing" we run permutation for backward compatibility
    "errorDistribution" => ["normal","tdist"],
    "f" => [f3],
    "σs" => [[[1., 1.],[0.,0.]]],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.3]],
    "nRep" => 5000,
    "blupMethod" => [ranef,olsranef],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],
    "nSubject" => [4,10,30],
    "nItemsPerCondition" => [2,10,30,50],
    "nPerm"=> 1000,
)

elseif task == 5
    #-----
    # Varying N
    paramList = Dict(
        "statsMethod" => ["waldsT","pBoot","permutation"], # if this is "missing" we run permutation for backward compatibility
        "errorDistribution" => ["normal","tdist","skewed"],
        "imbalance" => ["subject","trial"],
        "f" => [f3],
        "σs" => [[[1., 1.],[0.,0.]]],
        "σ" => 1.,
        "β" => [[0., 0.]],
        "nRep" => 5000,
        "blupMethod" => [ranef],
        "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
        "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],
        "nSubject" => [10,30],
        "nItemsPerCondition" => [30],
        "nPerm"=> 1000,
    )
end

##---
include(srcdir("sim_utilities.jl"))



dl = dict_list(paramList)[1]
dl["imbalance"] = "trial"
#dl["nPerm"] = 10
#dl["nSubject"] = 30
#dl["nItemsPerCondition"] = 50
#dl["σ"] = 0.01
#dl["f"] = f1
#dl["σs"] = [[0.,0.]]#,[0.,0.]]
simMod = sim_model(f4;convertDict(dl)...)
res = run_test(MersenneTwister(1),simMod; convertDict(dl)...)

#x = map(x->run_test(MersenneTwister(x),simMod; convertDict(dl)...),1:100)
#mean([y[2]<0.05 for y in x])




##---
include(srcdir("sim_utilities.jl"))

@time begin
nWorkers=80
for dl = dict_list(paramList)
    println(dl)
    
    dl_save =deepcopy(dl)
    dl_save["f"]  = string(dl_save["f"].rhs)|>x->replace(x," "=>"") # rename formula


    if "residualMethod" ∈ keys(dl_save)
        dl_save["residualMethod"]  = string(dl_save["residualMethod"])
    end

    fnName = datadir("cluster_sim2", savename("type1",dl_save, "jld2",allowedtypes=(Array,Float64,Integer,String,DataType,)))
    if isfile(fnName)
        # don't calculate again
	@show fnName
        continue
    end

    simMod = sim_model(f4;convertDict(dl)...)

    t = @elapsed begin
        res = run_test_distributed(nWorkers,simMod;convertDict(dl)...)
    end
    #@warn " \beta is actual res[1]!!"
    dl_save["results"] = res
    dl_save["runtime"] = t
    @tagsave(fnName, dl_save)
end
end


#!/home/st/st_us-051950/st_ac136984/julia-1.7.3/bin/julia
#SBATCH --cpus-per-task 40
#SBATCH --mem-per-cpu 1500
#SBATCH --nodes 1
#SBATCH -o slurmm/%x-%j.out
#SBATCH --job-name=LMMPerm
#SBATCH --time 40:0:0 




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
paramList = getParamList(task,f1,f2,f3,f4)

##---
include(srcdir("sim_utilities.jl"))



dl = dict_list(paramList)[1]
dl["imbalance"] = "trial"
dl["statsMethod"] = "pBoot"
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
nWorkers=40#"slurm" # 10 for local job
for dl = dict_list(paramList)
    println(dl)
    
    fnName = dl_filename(dl)

    if isfile(fnName)
        # don't calculate again
	@show fnName
        continue
    end

    # only necessary once
    simMod = sim_model(f4;convertDict(dl)...)

    t = @elapsed begin
        res = run_test_distributed(nWorkers,simMod;convertDict(dl)...)
    end
    #@warn " \beta is actual res[1]!!"
    println("saving")
    dl_save["results"] = res
    dl_save["runtime"] = t
    @show fnName
    @tagsave(fnName, dl_save)
    println("end of loop")
end
end


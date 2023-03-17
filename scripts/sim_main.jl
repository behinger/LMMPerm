#!/home/ac136984/.julia/juliaup/bin/julia
#SBATCH --cpus-per-task 125
#SBATCH --time=2-00:00:00           
#SBATCH --nodes=1                   
#SBATCH --ntasks-per-node=1
#SBATCH --partition=cpu,cpu-long         
#SBATCH --cpus-per-task=1
#SBATCH -o slurm/%x-%j.out
#SBATCH --job-name=LMMPerm


try
	global task = Base.parse(Int, ENV["SLURM_ARRAY_TASK_ID"]) # global to access outside of try/catch
        @show ENV["SLURM_ARRAY_TASK_COUNT"]
        @show ENV["SLURM_ARRAY_TASK_ID"]
        @show ENV["SLURM_JOB_ID"]
        @show ENV["SLURM_NTASKS"]
        scrum = true
catch KeyError
    global task = 3
    scrum = false
end
using DrWatson
quickactivate(pwd(),"LMMPerm")

using Revise,Random,TimerOutputs
includet(srcdir("sim_utilities.jl"))
#include(srcdir("permutationtest_be.jl"))


f1 =  @formula(dv ~ 1 + condition  + (1|subj))
f2 =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))
f3 =  @formula(dv ~ 1 + condition  + (1+condition|subj))
f4 =  @formula(dv ~ 1 + condition  + (1+condition|subj) + (1+condition|item))

@show task
#---h0 tests
paramList = getParamList(task,f1,f2,f3,f4)
##---
if 1 == 0
##---
dl = dict_list(paramList)[5]
#dl["imbalance"] = "trial"
#dl["statsMethod"] = "KenwardRoger"
#dl["nPerm"] = 100
#dl["nSubject"] = 30
#dl["nItemsPerCondition"] = 50
#dl["σ"] = 0.01
dl["f"] = f2
#dl["residualMethod"] = :shuffle
#dl["inflationMethod"] = MixedModelsPermutations.inflation_factor
#dl["σs"] = [[0.,0.]]#,[0.,0.]]
simMod = sim_model(f4;convertDict(dl)...)
res = run_test(MersenneTwister(2),simMod; onesided=true,convertDict(dl)...)


##--

loopKey = "statsMethod"

for (k,v) = enumerate(paramList[loopKey])
    p = deepcopy(dl)
    p[loopKey] = v
    @show v
    @time run_test(MersenneTwister(2),simMod; onesided=true,convertDict(p)...)
end

end


##---
include(srcdir("sim_utilities.jl"))

@time begin
nWorkers= scrum ? 125 else 5#"slurm" # 10 for local job
for dl = dict_list(paramList)
    println(dl)
    
    fnName,dl_save = dl_filename(dl)

    if isfile(fnName)
        # don't calculate again
        println("!!!! already finished: "* fnName)
        continue
    elseif  isfile(fnName*"_inprogress")
        println("!!!! already running: "* fnName)
        continue
    end
    touch(fnName*"_inprogress")
    
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
    rm(fnName*"_inprogress")

end
end


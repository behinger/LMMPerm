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
        global scrum = true
catch KeyError
    global task = 1
    global scrum = false
end
using DrWatson
quickactivate(pwd(),"LMMPerm")

using Revise,Random,TimerOutputs
includet(srcdir("sim_utilities.jl"))
includet(srcdir("sim_parameters.jl"))
#include(srcdir("permutationtest_be.jl"))



@show task
#---h0 tests
paramList = getParamList.(1:5)
dl_all = vcat(dict_list.(paramList)...)

f1,f2,f3,f4 = defaultFormulas()

##---
if 1 == 0
    # local testing area :)
##---
paramList = getParamList(1)

dl = dict_list(paramList)[111]
#dl["imbalance"] = "trial"
#dl["statsMethod"] = "KenwardRoger"
#dl["nPerm"] = 100
#dl["nSubject"] = 30
#dl["nItemsPerCondition"] = 50
#dl["σ"] = 0.01
#dl["f"] = f2
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
nWorkers= scrum ? 125 : 5#"slurm" # 10 for local job

# permute the dl_all with random seed the task/batch-id - this might reduce racing conditions that two jobs work on the same task.
for dl = dl_all[randperm(MersenneTwister(task),length(dl_all))]
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


using ClusterManagers
using Distributed

function onNode(k,n_workers)
    addprocs(
                n_workers = n_workers,
                exeflags = "--project",
                enable_threaded_blas = true,
            )
    @everywhere @eval using Clustermanagers
    
    @everywhere println("hello from $k, $(getpid()):$(gethostname())")    
    return
end


addprocs_slurm(2,partition="dev_multiple_e",time="0:0:30",mem_per_cpu=100,cpus_per_task=5,job_name="SlurmTest",o="slurmm/%x-%j.out",nodes=2)
@parallel for k = 1:10
    onNode(k,5)
end

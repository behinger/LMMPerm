using ClusterManagers
function onNode(k,n_workers)
    addprocs(
                n_workers = n_workers,
                exeflags = "--project",
                enable_threaded_blas = true,
            )
    @everywhere using SlurmClusterManager
    
    @everywhere println("hello from $k, $(myid()):$(gethostname())")    
    return
end

#!/home/st/st_us-051950/st_ac136984/julia-1.7.3/bin/julia
#SBATCH --cpus-per-task 28
#SBATCH --mem-per-cpu 1500
#SBATCH --nodes 10
#SBATCH -o slurmm/%x-%j.out
#SBATCH --job-name=LMMPerm
#SBATCH --time 20:0:0 

addprocs_slurm(2,partition="dev_multiple_e",time="0:0:30",mem-per-cpu=100,cpus-per-task=5,job-name="SlurmTest",o="slurmm/%x-%j.out")
@parallel for k = 1:10
    onNode(k,5)
end
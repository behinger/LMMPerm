using FileIO,JLD2, MixedModelsPermutations
g = FileIO.load("data/bug_threading2.jld2")
g  = FileIO.load("/store/users/ehinger/unfoldjl_dev/data/bug_threading/bug_threading2.jld2")
H0 = coef(g["mm"])
H0[2] = 0
perm = permutation(16000,g["mm"];β=H0,blup_method=olsranef,use_threads=false); 

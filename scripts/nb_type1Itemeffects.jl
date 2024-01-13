### A Pluto.jl notebook ###
# v0.19.36

using Markdown
using InteractiveUtils

# ╔═╡ b731d57a-902f-11ee-32c8-975415e4ecb9
begin
	using DrWatson
	quickactivate("..","LMMPerm")
	#using CairoMakie


	using Revise,Random
	using StatsModels,ProgressMeter,BlockDiagonals,DataFrames,StatsBase,Distributions,SlurmClusterManager, JellyMe4,RCall
	using MixedModels,MixedModelsSim, MixedModelsPermutations,DrWatson
	using TimerOutputs
	using Suppressor
	
	using Pkg
	Pkg.add("PlutoLinks")
	using PlutoLinks
	
	
end

# ╔═╡ eb6dadff-bbf8-40e5-9965-955e3d29ad44
using DataFramesMeta

# ╔═╡ d6a0ad10-f533-4e1d-be0a-adfd83bac45f
begin
	include("../src/sim_utilities.jl")
	include("../src/sim_parameters.jl")
	include("../src/permutationtest_be.jl")
end

# ╔═╡ 57ec9f63-b924-4005-b3f0-5e2d9abea319


# ╔═╡ 6d77e807-ecf5-4d94-a902-c0737fcd80c1
begin 
	paramList = getParamList.(1:5)
dl_all = vcat(dict_list.(paramList)...)

f1,f2,f3,f4 = defaultFormulas()

end


# ╔═╡ 3ccf1261-9529-4e6f-b248-0eae9dd9e9b1


# ╔═╡ 67b42335-e4ba-4a98-9d22-b91ec8706160


# ╔═╡ b71af654-b65a-4df3-9aa3-de44f980621a
begin

dl = dict_list( getParamList(1))[106]
#dl["imbalance"] = "trial"
#dl["statsMethod"] = "permutation"
#dl["nPerm"] = 2000
	reslist = []
simMod = sim_model(f4;convertDict(dl)...)
	# irrelevant will be replaced in setup_simMod
 #simMod.optsum.maxtime = 0.00001 # restrict per-iteration fitting time
   #simMod.optsum.maxfeval = 10
	for k = 1:3
res = run_test(MersenneTwister(k),simMod; onesided=true,convertDict(dl)...)
		append!(reslist,res)
	end
end

# ╔═╡ b6191a13-ae93-4a5d-bfe0-0597740e907c

        global warnings = @capture_err begin
			run_test(MersenneTwister(1),simMod; onesided=true,convertDict(dl)...)
		end


# ╔═╡ 0a20a501-3422-4bd2-aca9-7fda1144464b
 dict_list( getParamList(1))[106]

# ╔═╡ de623349-6eab-4859-ba34-af39be8b82b0
dl

# ╔═╡ 4ee1cc56-6742-431a-842e-992d9dbf8d2b
simMod

# ╔═╡ ff750e1c-727a-4dd4-a0f5-7f2e8b6807e5
[r[1] for r in reslist]

# ╔═╡ 5dd1ef5e-7f70-468e-b9ea-2ed39ce4d5b3
# ╠═╡ disabled = true
#=╠═╡
tmp = load("../data/cluster/cluster/9649078935191607504.jld2")
  ╠═╡ =#

# ╔═╡ 25654b5e-3747-4350-abd1-4e00af0b1ed0
tmp = load("../data/cluster/cluster/9649078935191607504.jld2")

# ╔═╡ 1151ec8a-7f97-4775-b22f-86bd2a9d6cbe
tmpWarn = load("../data/cluster_warnings/9649078935191607504.jld2")

# ╔═╡ 1ce8eb1a-4549-4d52-a78f-a754c4bb7b6b
tmp5 = load("../data/cluster_local/8365295882864693161.jld2")

# ╔═╡ c6b1b0c0-e20f-43ee-97b1-977c05d8c62d
reslist[1]

# ╔═╡ 82f67dfa-edc7-4863-9e6c-f6fc9bb9aa1b
@subset(tmp5["results"],:seed .== 1)

# ╔═╡ 5766e109-182f-4c6d-8a88-bceb3209bf41
@by(tmp5["results"],[:side,:coefname,:test],:pval=mean(:pval .<= 0.05))

# ╔═╡ 019aedb9-9138-4ac0-9f17-acf2447f55a1
tmp5["results"][4,:warnings]

# ╔═╡ d224be17-cd27-44b0-92a3-04248a22e23b
tmp5["results"5]

# ╔═╡ 1972b7a9-9ff3-4091-8049-839b59605ff7
@subset(tmp["results"],:seed .== 5001)

# ╔═╡ a0670947-2aa6-498f-a4a3-08b7b9ff24bd
md"""
"(Intercept)" 0.961039 5002 :twosided "β"
"""

# ╔═╡ Cell order:
# ╠═b731d57a-902f-11ee-32c8-975415e4ecb9
# ╠═d6a0ad10-f533-4e1d-be0a-adfd83bac45f
# ╠═57ec9f63-b924-4005-b3f0-5e2d9abea319
# ╠═6d77e807-ecf5-4d94-a902-c0737fcd80c1
# ╠═3ccf1261-9529-4e6f-b248-0eae9dd9e9b1
# ╠═b6191a13-ae93-4a5d-bfe0-0597740e907c
# ╠═67b42335-e4ba-4a98-9d22-b91ec8706160
# ╠═b71af654-b65a-4df3-9aa3-de44f980621a
# ╠═0a20a501-3422-4bd2-aca9-7fda1144464b
# ╠═de623349-6eab-4859-ba34-af39be8b82b0
# ╠═4ee1cc56-6742-431a-842e-992d9dbf8d2b
# ╠═ff750e1c-727a-4dd4-a0f5-7f2e8b6807e5
# ╠═5dd1ef5e-7f70-468e-b9ea-2ed39ce4d5b3
# ╠═25654b5e-3747-4350-abd1-4e00af0b1ed0
# ╠═1151ec8a-7f97-4775-b22f-86bd2a9d6cbe
# ╠═1ce8eb1a-4549-4d52-a78f-a754c4bb7b6b
# ╠═c6b1b0c0-e20f-43ee-97b1-977c05d8c62d
# ╠═82f67dfa-edc7-4863-9e6c-f6fc9bb9aa1b
# ╠═5766e109-182f-4c6d-8a88-bceb3209bf41
# ╠═019aedb9-9138-4ac0-9f17-acf2447f55a1
# ╠═d224be17-cd27-44b0-92a3-04248a22e23b
# ╠═1972b7a9-9ff3-4091-8049-839b59605ff7
# ╠═eb6dadff-bbf8-40e5-9965-955e3d29ad44
# ╠═a0670947-2aa6-498f-a4a3-08b7b9ff24bd

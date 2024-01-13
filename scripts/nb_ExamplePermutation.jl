### A Pluto.jl notebook ###
# v0.19.9

using Markdown
using InteractiveUtils

# ╔═╡ 4a06e976-c55f-11ec-2835-058cd999729b
begin
	using Pkg
	Pkg.activate("..")
	using PlutoUI
	using MixedModelsSim
	using MixedModelsPermutations
	using MixedModels
	using DataFrames
	using StatsModels
	using CairoMakie
	using AlgebraOfGraphics
	using Random
	using LinearAlgebra
	using StaticArrays #
	using ProgressMeter
	using BlockDiagonals
	using StatsBase
	
	
end

# ╔═╡ 8a84246f-4bba-4f1a-ae10-1a7b5bc607c1
using Distributions

# ╔═╡ e8446d68-31bd-42d7-a937-f5dac6da0564
include("../src/sim_utilities.jl");

# ╔═╡ b687d6be-918b-4c8d-b837-2ff3c5327988

snorm = SkewNormal(0,1,10)

# ╔═╡ 6d6bd843-4547-4a26-88f5-ddcd0421a5f0
std(snorm)

# ╔═╡ 6f0780bb-6b76-4430-a053-9b1984eda9ba
hist(rand(snorm-mean(snorm),10000),bins=100)

# ╔═╡ 78934a3d-3001-4022-8ff8-bc6a7df006d1
Legend

# ╔═╡ a1745cb4-5e31-4787-a999-27773800e20d
begin
β_org = [0., 0.,]
σs_org = [[1., 1.],[0.,0.]] # ful rank
	#σs_org = [[1., 0.],[0.,0.]] # instable :S
end

# ╔═╡ 94b41bed-7ce3-45eb-becb-f8c2f00e3a3e
begin
		nsub = 20
		nitem = 80
		f =  @formula(dv ~ 1 + condition  + (1+condition|subj));
		f_full =  @formula(dv ~ 1 + condition  + (1+condition|subj)+(1+condition|item));
		contrasts = Dict(
				:age=>EffectsCoding, 
				:stimType=>EffectsCoding(), 
				:condition=>EffectsCoding()
				);
		simMod_tmp = sim_model(f_full,nSubject=nsub,nItemsPerCondition=nitem)
		simMod = setup_simMod(MersenneTwister(2),simMod_tmp;σ = 1,f = f,β=β_org,σs = σs_org,errorDistribution="skewed",nSubject=nsub,nItemsPerCondition=nitem)
end;

# ╔═╡ 137a608f-770f-474e-a8f4-98a0094c4414
hist(residuals(simMod),bins=300)

# ╔═╡ d6591147-1895-465a-a06d-2de8fe2b41de
p = MixedModelsPermutations.permutation(100,simMod);

# ╔═╡ 127b7bf3-0ebd-4648-a1af-652d023721c9
let
	d = DataFrame(p.allpars)
AlgebraOfGraphics.data(d) * mapping(:value,color=:names	)*mapping(row=:type,col=:names)*visual(Hist) |> x->draw(x,facet = (; linkxaxes = :none,linkyaxes=:minimal),legend=(position="bottom",orientation=:horizontal))
end

# ╔═╡ Cell order:
# ╠═8a84246f-4bba-4f1a-ae10-1a7b5bc607c1
# ╠═6d6bd843-4547-4a26-88f5-ddcd0421a5f0
# ╠═b687d6be-918b-4c8d-b837-2ff3c5327988
# ╠═6f0780bb-6b76-4430-a053-9b1984eda9ba
# ╠═137a608f-770f-474e-a8f4-98a0094c4414
# ╠═d6591147-1895-465a-a06d-2de8fe2b41de
# ╠═127b7bf3-0ebd-4648-a1af-652d023721c9
# ╠═78934a3d-3001-4022-8ff8-bc6a7df006d1
# ╠═e8446d68-31bd-42d7-a937-f5dac6da0564
# ╠═94b41bed-7ce3-45eb-becb-f8c2f00e3a3e
# ╠═a1745cb4-5e31-4787-a999-27773800e20d
# ╠═4a06e976-c55f-11ec-2835-058cd999729b

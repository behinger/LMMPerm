### A Pluto.jl notebook ###
# v0.19.2

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

# ╔═╡ e8446d68-31bd-42d7-a937-f5dac6da0564
include("../src/sim_utilities.jl");

# ╔═╡ dece3491-2a55-43c4-82ca-df7869d42b78


# ╔═╡ 94b41bed-7ce3-45eb-becb-f8c2f00e3a3e
begin
		nsub = 20
		nitem = 80
		f =  @formula(dv ~ 1 + condition+stimType  + (1+condition+stimType|subj));
		contrasts = Dict(
				:age=>EffectsCoding, 
				:stimType=>EffectsCoding(), 
				:condition=>EffectsCoding()
				);
		dat = DataFrame(sim_model_getData(nsub,nitem));
end;

# ╔═╡ a1745cb4-5e31-4787-a999-27773800e20d
begin
β_org = [0., 0.,0.]
σs_org = [1,4,2.,2.,2.,2.] # ful rank
	σs_org = [1,4,0,0,0,0] # instable :S
end

# ╔═╡ b55babb0-eb98-4d91-ae3b-f966701e6919
function sim_model(seed_val)
	# generate simulated "y" vector
	simMod = fit(MixedModel, f, dat;contrasts=contrasts)
	simMod = MixedModelsSim.update!(simMod,[create_re(x...) for x in σs_org]...)
	simMod = simulate!(MersenneTwister(Int(seed_val)),simMod, β = β_org, σ = 1.)
	
	# fit a LMM mod to the y (separate models to maybe have separate formulas later
	datSim = dat
	datSim.dv = simMod.y
	simMod2 = MixedModels.fit(MixedModel,f ,datSim;contrasts=contrasts)
	return simMod2,datSim
end


# ╔═╡ add34216-cf95-436a-b030-3f04cd9ba2b8
mres,data = sim_model(3)

# ╔═╡ d6591147-1895-465a-a06d-2de8fe2b41de
p = MixedModelsPermutations.permutation(100,mres);

# ╔═╡ 127b7bf3-0ebd-4648-a1af-652d023721c9
let
	d = DataFrame(p.allpars)
AlgebraOfGraphics.data(d) * mapping(:value,color=:names	)*mapping(row=:type,col=:names)*visual(Hist) |> x->draw(x,facet = (; linkxaxes = :none,linkyaxes=:minimal),legend=(position="bottom",))
end

# ╔═╡ 322b2a58-c47a-499b-8b96-f1afb5b34391
DataFrame(p.allpars).type

# ╔═╡ Cell order:
# ╠═add34216-cf95-436a-b030-3f04cd9ba2b8
# ╠═d6591147-1895-465a-a06d-2de8fe2b41de
# ╠═127b7bf3-0ebd-4648-a1af-652d023721c9
# ╠═322b2a58-c47a-499b-8b96-f1afb5b34391
# ╠═dece3491-2a55-43c4-82ca-df7869d42b78
# ╠═e8446d68-31bd-42d7-a937-f5dac6da0564
# ╠═94b41bed-7ce3-45eb-becb-f8c2f00e3a3e
# ╠═a1745cb4-5e31-4787-a999-27773800e20d
# ╠═b55babb0-eb98-4d91-ae3b-f966701e6919
# ╠═4a06e976-c55f-11ec-2835-058cd999729b

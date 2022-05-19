### A Pluto.jl notebook ###
# v0.19.2

using Markdown
using InteractiveUtils

# ╔═╡ 37e0c8ee-a9c3-11ec-1e10-e776f70176e2
begin
	using Pkg
	Pkg.activate("..")
	using Revise
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

# ╔═╡ 12ba6dcc-7f0a-4148-9658-4c3626a84116
include("../src/sim_utilities.jl");

# ╔═╡ 401f1e66-50e7-4090-892a-2fcb507c6319


# ╔═╡ 3a9d1713-2a3f-4477-a8cf-cbcd21099f5c
begin
		nsub = 28
		nitem = 34
		f =  @formula(dv ~ 1 + condition+stimType  + (1+condition+stimType|subj));
		contrasts = Dict(
				:age=>EffectsCoding, 
				:stimType=>EffectsCoding(), 
				:condition=>EffectsCoding()
				);
		dat = DataFrame(sim_model_getData(nsub,nitem));
end;

# ╔═╡ 44d45020-ab2b-42f0-a2db-1d94d60a90bc
dat

# ╔═╡ e18bd2bc-d987-431b-980b-ae92eb64beaf
begin
β_org = [0., 0.,0.]
σs_org = [1,4,2.,2.,2.,2.] # ful rank
	#σs_org = [1,4,0,0,0,0] # instable :S
end

# ╔═╡ 0f57c2f2-e0d0-46fe-a8f5-cc789b185cf6
begin

	
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
end;

# ╔═╡ 162c4305-4bf8-4df2-9483-fd469126d5fb
function inflation_method_cov(m::LinearMixedModel, blups=ranef(m), resids=residuals(m))
# FIXME I'm not sure this is correct
#       the nonparametric bootstrap underestimates variance components
#       compared to the parametricbootstrap

    σ = sdest(m)
    σres = std(resids; corrected=false)
  	inflation = map(zip(m.reterms, blups)) do (trm, re)
        # inflation
        λmle =  trm.λ * σ                              # L_R in CGR

		cov_emp = StatsBase.cov(re'; corrected=false)
				
        chol = cholesky(cov_emp, Val(true); check=false,tol=10^-5)

        #  ATTEMPT 2
         while chol.rank != size(cov_emp, 1)
			 #@info "rep"
            idx = chol.p[(chol.rank+1):end]
            cov_emp[idx, idx] .+= 1e-6
            chol = cholesky(cov_emp, Val(true); check=false,tol=10^-5)
        end
		
		L = chol.L[invperm(chol.p),:]
		cov_emp = L * L'
		cov_mle = λmle * λmle'
        
		return cov_mle / cov_emp
	end
    return [inflation; σ / σres]
end

# ╔═╡ b17cc188-89f6-4bfb-a126-03a192602d8b
morig,datSim = sim_model(5)

# ╔═╡ a53a5fb0-506f-4cfa-97b7-98be36742dd4
morig

# ╔═╡ aac7de5f-47d2-4e64-ab9c-ff9b74393469
MixedModels.rePCA(morig)

# ╔═╡ d61f4824-e516-4209-ba3c-1e8ccf825f74
begin
	
	    
	    
	    residual_permutation=:signflip
	    residual_method=MixedModelsPermutations.residuals
	    blup_method=MixedModelsPermutations.ranef
	    inflation_method=MixedModelsPermutations.inflation_factor
	
	    βsc, θsc = similar(morig.β), similar(morig.θ)
	    p, k = length(βsc), length(θsc)
	
	    β_names = (Symbol.(fixefnames(morig))..., )
	    rank = length(β_names)
	
	    blups = blup_method(morig)
	    resids = residual_method(morig, blups)
	    reterms = morig.reterms;
	   
end;

# ╔═╡ c2bf327c-7548-4c4a-8219-c37cd860e0ad
display(morig)

# ╔═╡ 3c8089f3-9eb4-4f08-875c-e4f8922b9dd6
morig

# ╔═╡ de83ac6d-e4a9-46d2-87ee-21675179ed5f
begin
		    rng = MersenneTwister(5)

	 scalings = inflation_method(morig, blups, resids)
	display(scalings[1] * scalings[1]')
		    # we need arrays of these for in-place operations to work across threads
	scalings = inflation_method_cov(morig,blups,resids)	    
	#scalings[1] = 1 ./scalings[1]
	println("\n\ncov-scaling")
	display(scalings[1] )
	#chol = cholesky(scalings[1], Val(true); check=false,tol=10^-5)
	#scalings[1] = chol.L[invperm(chol.p),:]
	#scalings[1] = I(3)
	#println("\n\nchol.L of cov-scaling")
	
	#display(scalings[1] )
	

	β=coef(morig)#zeros((coef(morig)))
	#β[2] = 0

	m_perm = MixedModelsPermutations.permute!(rng, deepcopy(morig);
				β=β, 
				blups=blups, 
				resids=resids,
				residual_permutation=residual_permutation, 
				scalings=scalings)
 	refit!(m_perm)
end

# ╔═╡ 9c89e1ce-b2f7-4fff-a239-110a518b75e2
coef(morig)

# ╔═╡ 1f51b98b-b2ba-4369-a34a-4f92f0f96b14
datSim

# ╔═╡ 0e56f07f-f4fe-4978-8275-cecf5a58a973
begin
datPlot = deepcopy(datSim)
	datPlot.p_residuals = residuals(m_perm)
	datPlot.p_response = response(m_perm)
	datPlot.response = response(morig)
	datPlot.residuals = residuals(morig)
	datPlot.row = 1:size(datPlot,1)
	#
	
	#d
	
	#data(datSim[datSim.subj.=="S25",:]) * mapping(:row,:dv,color=:subj)*visual(Scatter)|>draw
end

# ╔═╡ 2752b879-6178-4630-9106-953642c557e5

	data(datPlot) * mapping(:row,:p_response,color=:subj)*visual(Scatter)|>draw


# ╔═╡ fd43b181-bb30-4936-8788-434f536f67c1
data(datPlot) * mapping(:row,:response,color=:subj)*visual(Scatter)|>draw

# ╔═╡ d26b2916-b27e-40dc-8042-cb03a3148c06
data(datPlot) * mapping(:row,:p_response,color=:condition)*visual(Scatter)|>draw

# ╔═╡ 59a5048d-3423-4cf9-be4a-34d497f64903
	data(datPlot) * mapping(:p_response,:response,color=:subj)*visual(Scatter)|>draw


# ╔═╡ 38228fa5-9484-4fad-94aa-65837feaef2a
	data(datPlot) * mapping(:p_residuals,:residuals,color=:subj)*visual(Scatter)|>draw


# ╔═╡ Cell order:
# ╠═37e0c8ee-a9c3-11ec-1e10-e776f70176e2
# ╠═12ba6dcc-7f0a-4148-9658-4c3626a84116
# ╠═401f1e66-50e7-4090-892a-2fcb507c6319
# ╠═3a9d1713-2a3f-4477-a8cf-cbcd21099f5c
# ╠═44d45020-ab2b-42f0-a2db-1d94d60a90bc
# ╠═e18bd2bc-d987-431b-980b-ae92eb64beaf
# ╠═0f57c2f2-e0d0-46fe-a8f5-cc789b185cf6
# ╠═a53a5fb0-506f-4cfa-97b7-98be36742dd4
# ╠═aac7de5f-47d2-4e64-ab9c-ff9b74393469
# ╠═d61f4824-e516-4209-ba3c-1e8ccf825f74
# ╠═c2bf327c-7548-4c4a-8219-c37cd860e0ad
# ╠═162c4305-4bf8-4df2-9483-fd469126d5fb
# ╠═b17cc188-89f6-4bfb-a126-03a192602d8b
# ╠═3c8089f3-9eb4-4f08-875c-e4f8922b9dd6
# ╠═de83ac6d-e4a9-46d2-87ee-21675179ed5f
# ╠═9c89e1ce-b2f7-4fff-a239-110a518b75e2
# ╠═1f51b98b-b2ba-4369-a34a-4f92f0f96b14
# ╠═0e56f07f-f4fe-4978-8275-cecf5a58a973
# ╠═2752b879-6178-4630-9106-953642c557e5
# ╠═fd43b181-bb30-4936-8788-434f536f67c1
# ╠═d26b2916-b27e-40dc-8042-cb03a3148c06
# ╠═59a5048d-3423-4cf9-be4a-34d497f64903
# ╠═38228fa5-9484-4fad-94aa-65837feaef2a

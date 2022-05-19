### A Pluto.jl notebook ###
# v0.19.0

using Markdown
using InteractiveUtils

# This Pluto notebook uses @bind for interactivity. When running this notebook outside of Pluto, the following 'mock version' of @bind gives bound variables a default value (instead of an error).
macro bind(def, element)
    quote
        local iv = try Base.loaded_modules[Base.PkgId(Base.UUID("6e696c72-6542-2067-7265-42206c756150"), "AbstractPlutoDingetjes")].Bonds.initial_value catch; b -> missing; end
        local el = $(esc(element))
        global $(esc(def)) = Core.applicable(Base.get, el) ? Base.get(el) : iv(el)
        el
    end
end

# ╔═╡ b7ac3efc-8bda-41d9-b3d5-147591bac4b4
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
	using Plots
	using Random
	using LinearAlgebra
	using StaticArrays #
	using BlockDiagonals
	using ProgressMeter
	using StatsBase
	
end

# ╔═╡ 9374ae6b-f09b-471f-adbb-b95f5b6d19e0
using Statistics

# ╔═╡ 72f5e910-e40d-11eb-3725-f19a95c9b3c1
include("../src/sim_utilities.jl");

# ╔═╡ 3b2dee16-4010-4602-9556-d6308ea087fb


# ╔═╡ 0cbf776c-8e56-4d32-8a55-da22f8bf1940
f =  @formula(dv ~ 1 + condition  + (1+condition|subj));

# ╔═╡ 3f4afdc8-04af-45b6-aadf-a72ca1ff158c
#blup_method = olsranef;

# ╔═╡ 6917ddd6-6ef3-43a5-83f1-ef6583be54f3
residual_method = :signflip;

# ╔═╡ cbecd90f-b262-4bbd-8a55-177f3e40ae50
β = [0., 0.]

# ╔═╡ 55e97ef0-0f23-48a6-8229-7f1b6ca3d044
md"seed $(@bind seed_val Slider(1:100; default=1., show_value=true))"

# ╔═╡ 2141edd4-c8ed-4d1b-a5ec-9913b19a2d4a
	md"Subjects: $(@bind nsub Slider(2:2:50; default=30, show_value=true))"


# ╔═╡ bac32d2f-fb1f-4f02-b2f5-49df029b7b92
md"Items per Subject: $(@bind nitem Slider(2:2:50; default=30, show_value=true))"

# ╔═╡ 0eaf2824-dd65-4dfb-905e-96d8a1754e08
begin
	contrasts = Dict(
			:age=>EffectsCoding, 
			:stimType=>EffectsCoding(), 
			:condition=>EffectsCoding()
			);
	dat = DataFrame(sim_model_getData(nsub,nitem));
end;

# ╔═╡ eb276ed2-533a-4d40-a6df-bda77d627d3c
nrow(dat)/nsub # number of trials per subject

# ╔═╡ eccf3c09-aa58-4c7c-8e43-364309df6885
@bind bootPerm Radio([
		"0"=>"Bootstrap",
		"1"=>"Permutation",
		],"1")

# ╔═╡ 48fc0434-2a5c-4c61-9b37-354047aa15b8
@bind olsranef Radio([
		"1"=>"blups ranef",
		"2"=>"blups olsranef",
		#"3"=>"experimental"
		],"1")

# ╔═╡ e132bb53-9b6c-4d61-bb45-d5d9209ff4da
@bind reml Radio([
		"0"=>"ML",
		"1"=>"REML",
		#"3"=>"experimental"
		],"1")

# ╔═╡ acc913dd-b17c-4137-83f4-7df179bdc3a1
@bind res Radio([
		"1"=>"residuals(x)",
		"2"=>"residuals(x,olsranef)",
		#"3"=>"experimental"
		],"1")

# ╔═╡ 968e3f53-055c-4dc0-b4e3-1217e511b3b4
# radio button to function
begin
	if res == "1"
		residual_function = x->residuals(x)
	elseif res == "2"
		residual_function = x->residuals(x,olsranef(x))
	elseif res == "3"
		residual_function = x->residuals(x,ranef(x))
	end
end;

# ╔═╡ 2776e7cf-ee8f-421e-98ed-fa03ecdbee68
md"σ-intercept: $(@bind σ1 Slider(0:0.1:5; default=0., show_value=true))"

# ╔═╡ 629d660c-6a13-422c-8acc-6c3b8d37acaa
md"σ-test: $(@bind σ2 Slider(0:0.1:10; default=4., show_value=true))"

# ╔═╡ 21664a88-30fd-4829-a70d-e727a5f5b66b
σs = [σ1,σ2,0.2]

# ╔═╡ cfd225bb-1eec-41b1-b6c1-ebc036b7e1d2
begin
	function sim_model(seed_val,reml,contrasts)
	# generate simulated "y" vector
	simMod = fit(MixedModel, f, dat,contrasts=contrasts,REML=reml=="1")
	simMod = MixedModelsSim.update!(simMod,[create_re(x...) for x in σs]...)
	simMod = simulate!(MersenneTwister(Int(seed_val)),simMod, β = β, σ = 1.)
	
	# fit a LMM mod to the y (separate models to maybe have separate formulas later
	datSim = dat
	datSim.dv = simMod.y
	simMod2 = MixedModels.fit(MixedModel,f ,datSim,contrasts=contrasts,REML=reml=="1")
	return simMod2
end
end;

# ╔═╡ 3e6110d5-221f-4b8a-97da-10421eb70991
simMod2 = sim_model(seed_val,reml,contrasts)

# ╔═╡ 3fb9f0ef-af34-4624-8ce6-88d71711d005
function residuals2(model::LinearMixedModel{T}, blups::Vector{<:AbstractMatrix{T}}) where T
    # XXX This is kinda type piracy, if it weren't developed by one of the MixedModels.jl devs....

    y = response(model) # we are now modifying the model

    ŷ = zeros(T, length(y))
	scalings=MixedModelsPermutations.inflation_factor(model)

	 for (inflation, re, trm) in zip(scalings, blups, model.reterms)
        npreds, ngrps = size(re)
        # sign flipping
        newre = re * diagm(rand(MersenneTwister(1), (-1,1), ngrps))

        # our RE are actually already scaled, but this method (of unscaledre!)
        # isn't dependent on the scaling (only the RNG methods are)
        # this just multiplies the Z matrices by the BLUPs
        # and add that to y
        MixedModels.unscaledre!(y, trm, lmul!(inflation, newre))
        # XXX inflation is resampling invariant -- should we move it out?
    end
    # TODO: do this inplace to avoid an allocation
    return y .- ŷ
end

# ╔═╡ c64d7900-17ca-49a8-8d8a-bfbfe9a5b4c1
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
        

		#print("\ncov(mle):")
		#display(cov_mle)
		#print("\ncov(emp):")
		#display(cov_emp)
        # no transpose because the RE are transposed relativ to CGR
		#@info λmle
		#@info λemp
		#display(cov_mle ./ cov_emp)
        #chol = cholesky(cov_mle ./ cov_emp)
		#chol.L[invperm(chol.p),:]
		return cov_mle / cov_emp
	end
    return [inflation; σ / σres]
end

# ╔═╡ 58fa0d39-65e0-4ec7-aeb0-e8bde381379a
begin
	function calc_results(simMod2,residual_function,residual_method)
		H0 = coef(simMod2)
		H0[2] = 0.0
			
			morig = simMod2
			βsc, θsc = similar(morig.β), similar(morig.θ)
		    p, k = length(βsc), length(θsc)
		    m = deepcopy(morig)
		
		    β_names = (Symbol.(fixefnames(morig))..., )
		
		#	perm = permutation(Random.MersenneTwister(2),20, simMod2; β = H0,residual_method=residual_method,blup_method=blup_method,infla)
		perm = []
		for p = 1:20
		
		
				resids = residual_function(simMod2)
			if olsranef=="2"
				blups = MixedModelsPermutations.olsranef(simMod2)
				scalings = I(length(β))
			else
				blups = ranef(simMod2)
				scalings = MixedModelsPermutations.inflation_factor(morig, blups, resids)
				scalings = inflation_method_cov(morig,blups,resids)
			end
		
				
				if bootPerm == "0"
					# boot strap
					
					model = MixedModelsPermutations.resample!(MersenneTwister(p+1),deepcopy(morig);
		                  blups=ranef(simMod2),
		                  resids=residual_function(simMod2),
		                  scalings= scalings,
					)
				else
				#permutation
		
			model = MixedModelsPermutations.permute!(MersenneTwister(p+1),deepcopy(morig);
		                  β = H0,
		                  blups=blups,
		                  resids=resids,
					      residual_permutation = residual_method,
		                  scalings=scalings,
			)
			
		
				end
			refit!(model)
			res=	(
			 objective = model.objective,
			 σ = model.σ,
			 β = NamedTuple{β_names}(MixedModels.fixef!(βsc, model)),
			 #se = SVector{p,Float64}(MixedModels.stderror!(βsc, model)),
			 θ = SVector{k,Float64}(MixedModels.getθ!(θsc, model)),
			)
			append!(perm,[res])
		end
		return perm
	end
	
end

# ╔═╡ b95d6d67-a625-4007-9105-0b04764be51b
perm =  calc_results(simMod2,residual_function,residual_method);

# ╔═╡ 64d2509b-d94e-4e48-8f81-fa13d5f9e9be
let	
	Plots.plot(DataFrame(perm).σ,label="permutation")
	hline!([1.],label="theoretical σ")
	hline!([simMod2.σ],label="empirical σ")
	ylims!((1. *0.6,1. *1.2))
	ylabel!("residual σ")
	xlabel!("permutation")
end

# ╔═╡ 1a739946-82d2-4def-a655-db6dac6d737d
# ╠═╡ disabled = true
#=╠═╡

let
	rep = 50
	res = Array{Float64}(undef,rep)
for r = 1:rep
	simMod2 = sim_model(r,reml,contrasts)
	perm =  calc_results(simMod2,residual_function,residual_method)
	res[r] = mean(DataFrame(perm).σ)
end
	histogram(res)#,bins=0.82:0.01:100)
	@info mean(res)
	vline!([1.])
	vline!([mean(res)])
	
end
  ╠═╡ =#

# ╔═╡ Cell order:
# ╠═b7ac3efc-8bda-41d9-b3d5-147591bac4b4
# ╠═72f5e910-e40d-11eb-3725-f19a95c9b3c1
# ╠═0eaf2824-dd65-4dfb-905e-96d8a1754e08
# ╠═cfd225bb-1eec-41b1-b6c1-ebc036b7e1d2
# ╠═eb276ed2-533a-4d40-a6df-bda77d627d3c
# ╠═3e6110d5-221f-4b8a-97da-10421eb70991
# ╠═58fa0d39-65e0-4ec7-aeb0-e8bde381379a
# ╠═3b2dee16-4010-4602-9556-d6308ea087fb
# ╠═b95d6d67-a625-4007-9105-0b04764be51b
# ╠═968e3f53-055c-4dc0-b4e3-1217e511b3b4
# ╠═0cbf776c-8e56-4d32-8a55-da22f8bf1940
# ╠═3f4afdc8-04af-45b6-aadf-a72ca1ff158c
# ╠═6917ddd6-6ef3-43a5-83f1-ef6583be54f3
# ╠═cbecd90f-b262-4bbd-8a55-177f3e40ae50
# ╠═21664a88-30fd-4829-a70d-e727a5f5b66b
# ╟─55e97ef0-0f23-48a6-8229-7f1b6ca3d044
# ╟─2141edd4-c8ed-4d1b-a5ec-9913b19a2d4a
# ╟─bac32d2f-fb1f-4f02-b2f5-49df029b7b92
# ╟─eccf3c09-aa58-4c7c-8e43-364309df6885
# ╟─48fc0434-2a5c-4c61-9b37-354047aa15b8
# ╟─e132bb53-9b6c-4d61-bb45-d5d9209ff4da
# ╠═acc913dd-b17c-4137-83f4-7df179bdc3a1
# ╠═64d2509b-d94e-4e48-8f81-fa13d5f9e9be
# ╠═1a739946-82d2-4def-a655-db6dac6d737d
# ╠═9374ae6b-f09b-471f-adbb-b95f5b6d19e0
# ╟─2776e7cf-ee8f-421e-98ed-fa03ecdbee68
# ╟─629d660c-6a13-422c-8acc-6c3b8d37acaa
# ╠═3fb9f0ef-af34-4624-8ce6-88d71711d005
# ╠═c64d7900-17ca-49a8-8d8a-bfbfe9a5b4c1

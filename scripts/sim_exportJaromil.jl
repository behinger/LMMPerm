using Random,StatsModels,CSV
f3 =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))

dat = sim_model_getData()

simMod = sim_model(f3,simulationCoding = DummyCoding)

θ = [[1.], [4.]]
θ = [create_re(x...) for x in θ]
β = [0., 0.]
σ = 1.
simMod = MixedModelsSim.update!(simMod,θ...)
for k = 1:4000
simMod = simulate!(Random.MersenneTwister(k), simMod, β = β, σ = σ)
d = dat|>DataFrame
d.dv = simMod.y

   CSV.write(datadir("exportForJaromil","sim_"*string(k)*".csv"),d )
end
using DrWatson
@quickactivate "LMMPerm"

using Random,TimerOutputs
include(srcdir("sim_utilities.jl"))
include(srcdir("permutationtest_be.jl"))

convertDict = x-> [Symbol(d.first) => d.second for d in x]
f1 =  @formula(dv ~ 1 + condition  + (1|subj))
f2 =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))
f3 =  @formula(dv ~ 1 + condition  + (1+condition|subj))
f4 =  @formula(dv ~ 1 + condition  + (1+condition|subj) + (1+condition|item))

#---h0 tests
paramList = Dict(
    "f" => [f1,f3,f4],
    "σs" => [@onlyif("f"!= f4, [[1., 0.], [0.,0.]]),
             @onlyif("f"!= f4, [[1., 1.], [0.,0.]]),  
             @onlyif("f"!= f4, [[1., 4.], [0.,0.]]),
             @onlyif("f"!= f4, [[4., 1.], [0.,0.]]),

             @onlyif("f"== f4, [[1., 1.], [1., 0.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 4.]])],
    "σ" => 1.,
    "β" => [[0., 0.]],
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "inflationMethod" => [MixedModelsPermutations.inflation_factor,"noScaling"],
    "residualMethod" => [:signflip,:shuffle],#[:signflip,:shuffle],"
    "nRep" => 5000,
    "nPerm"=> 1000,

)

#----
# H1 test
paramList = Dict(
    "f" => [f1,f3,f4],
    "σs" => [@onlyif("f"== f1, [[1., 0.], [0.,0.]]),
             @onlyif("f"== f3, [[1., 1.], [0.,0.]]),  
             @onlyif("f"== f3, [[1., 4.], [0.,0.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
             ],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.1],[0., 1.]],
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "inflationMethod" => [MixedModelsPermutations.inflation_factor,"noScaling"],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "nRep" => 5000,
    "nPerm"=> 1000,

)

#----
# Power calculations
paramList = Dict(
    "statsMethod" => ["waldsT","pBoot","permutation"], # if this is "missing" we run permutation for backward compatibility
    "errorDistribution" => ["normal","tdist"],
    "f" => [f3],
    "σs" => [[[1., 1.], [0.,0.]]],
    "σ" => 1.,
    "β" => [[0., 0.],[0., 0.1],[0., 0.2],[0., .3],[0., 0.5]],
    "nRep" => 5000,
    "blupMethod" => [ranef,@onlyif("f"!=f4,olsranef)],
    "residualMethod" => [:shuffle],#[:signflip,:shuffle],"
    "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],
    "nPerm"=> 1000,
)
##---
include(srcdir("sim_utilities.jl"))



simMod = sim_model(f4)
dl = dict_list(paramList)[7]
dl["nPerm"] = 10
res = run_test(MersenneTwister(5),simMod; convertDict(dl)...)

##---
include(srcdir("sim_utilities.jl"))

@time begin
nWorkers=10
for dl = dict_list(paramList)
    println(dl)
    
    dl_save =deepcopy(dl)
    dl_save["f"]  = string(dl_save["f"].rhs)|>x->replace(x," "=>"") # rename formula


    if "residualMethod" ∈ keys(dl_save)
        dl_save["residualMethod"]  = string(dl_save["residualMethod"])
    end

    fnName = datadir("22-05_sim", savename("type1",dl_save, "jld2",allowedtypes=(Array,Float64,Integer,String,DataType,)))
    if isfile(fnName)
        # don't calculate again
        continue
    end

    simMod = sim_model(f4)

    t = @elapsed begin
        res = run_test_distributed(nWorkers,simMod;convertDict(dl)...)
    end
    #@warn " \beta is actual res[1]!!"
    dl_save["results"] = res
    dl_save["runtime"] = t
    @tagsave(fnName, dl_save)
end

end





##--- see nb_results.jl!!


##---- Load & Analze
c = collect_results(datadir("sim"))
c[!,"z<0.05"] = [sum(r.results[r.results.h1.=="1",:].z .<=0.05)/r.nRep for r in eachrow(c)]
c[!,"β<0.05"] = [sum(r.results[r.results.h1.=="1",:].β .<=0.05)/r.nRep for r in eachrow(c)]

c[!,["f","blupMethod","residualMethod","σs","β<0.05","z<0.05","analysisCoding","simulationCoding"]]




#d = DataFrame()

#for row in eachrow(c)
#     for col in filter(!=(:results), propertynames(row))
#        @show row[col]
#         row.results[!, col] .= row[col]
#    end
#    vcat(d, row.results)
#end


d = vcat(c.results...)
d[!,"blupMethod"] = vcat(repeat.([[x] for x in c.blupMethod],size.(c.results,1))...)
d[!,"residualMethod"] = vcat(repeat.([[x] for x in c.residualMethod],size.(c.results,1))...)
d[!,"σs"] = vcat(repeat.([[x] for x in c.σs],size.(c.results,1))...)
d[!,"nRep"] = vcat(repeat.([[x] for x in c.nRep],size.(c.results,1))...)
d[!,"nPerm"] = vcat(repeat.([[x] for x in c.nPerm],size.(c.results,1))...)


using DataFrames, AlgebraOfGraphics,Makie
using GLMakie

data(d[d.h1.=="1",:]) * mapping(:β,color=:σs,dodge=:σs,layout=:σs) * AlgebraOfGraphics.histogram(bins=0:0.01:1) |>draw

data(d[d.h1.=="1",:]|>x->stack(x,[Symbol("z"),Symbol("β")])) * mapping(:value,color=:σs,col=:σs,row=:variable) * AlgebraOfGraphics.histogram(bins=0:0.01:1) |>draw



data(d) * mapping(:β,color=:h1,layout_x=:blupMethod,layout_y=:residualMethod) * AlgebraOfGraphics.density() |>draw

data(d[d.h1.=="1",:]) * mapping(:z,color=:σs,layout_x=:blupMethod,layout_y=:residualMethod) * AlgebraOfGraphics.density() |>draw()



d = data(stack(c,[Symbol("z<0.05"),Symbol("β<0.05")])) * mapping(:σs,:value,color=:variable) *visual(Scatter) |>draw
hlines!(d.grid[1,1].axis,[0.05],color="black")
##---

p = data(c[(c.blupMethod.=="olsranefjf").&(c.residualMethod.=="signflip"),:]) * 
mapping(:nPerm,Symbol("<0.05"),
    color=:nRep,
    marker=:nRep,
    layout_y=:blupMethod,
    layout_x=:σs) *
visual(Scatter   ,) |>draw()
#ylims!(p.scene,(0.03, 0.06))

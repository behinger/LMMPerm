using DrWatson
@quickactivate "LMMPerm"

using Random,TimerOutputs
include(srcdir("sim_utilities.jl"))
include(srcdir("permutationtest_be.jl"))

f1 =  @formula(dv ~ 1 + condition  + (1|subj))
f2 =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))
f3 =  @formula(dv ~ 1 + condition  + (1+condition|subj))
f4 =  @formula(dv ~ 1 + condition  + (1+condition|subj) + (1+condition|item))
paramList = Dict(
    "f" => [f1,f3],
    "σs" => [@onlyif("f"!= f4, [[1., 0.], [0.,0.]]),
             @onlyif("f"!= f4, [[1., 1.], [0.,0.]]),  
             @onlyif("f"!= f4, [[1., 4.], [0.,0.]]),

             @onlyif("f"== f4, [[1., 1.], [1., 0.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
             @onlyif("f"== f4, [[1., 1.], [1., 4.]])],
    "σ" => 1.,
    "β" => [[0., 0.]],
    "blupMethod" => ["ranef","olsranef"],
    "residualMethod" => [:signflip],#[:signflip,:shuffle],
    "nRep" => 5000,
    "nPerm"=> 1000,
    "analysisCoding"=> DummyCoding,
    "simulationCoding" => DummyCoding
)



dict_list(paramList)
##---
include(srcdir("sim_utilities.jl"))

dl = dict_list(paramList)[1]
simMod = sim_model(f4,simulationCoding = dl["simulationCoding"])
res = run_permutationtest(MersenneTwister(5),simMod,
                dl["nPerm"],dl["β"],dl["σ"],[create_re(x...) for x in dl["σs"]],
                dl["residualMethod"], getfield(Main,Meta.parse(dl["blupMethod"])),dl["analysisCoding"],dl["f"])

##---
include(srcdir("sim_utilities.jl"))

@time begin
nWorkers=20
for dl = dict_list(paramList)
    println(dl)
    dl_save =deepcopy(dl)
    dl_save["f"]  = string(dl_save["f"].rhs)|>x->replace(x," "=>"") # rename formula
    dl_save["residualMethod"]  = string(dl_save["residualMethod"])

    fnName = datadir("sim", savename("type1",dl_save, "jld2",allowedtypes=(Array,Float64,Integer,String,DataType,)))
    if isfile(fnName)
        # don't calculate again
        continue
    end
    simMod = sim_model(f4,simulationCoding=dl["simulationCoding"],)
    t = @elapsed begin
    res = run_permutationtest_distributed(nWorkers,dl["nRep"],simMod,
        dl["nPerm"],dl["β"],dl["σ"],
        [create_re(x...) for x in dl["σs"]],
        dl["residualMethod"],
        getfield(Main,Meta.parse(dl["blupMethod"])),
        dl["analysisCoding"],dl["f"] )
    end
    df = DataFrame(:z=>res[1][:],:β=>res[2][:],:h1=>[repeat(["1"],size(res[1],1)); repeat(["0"],size(res[1],1))])
    dl_save["results"] = df
    dl_save["runtime"] = t
    @tagsave(fnName, dl_save)
end

end

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

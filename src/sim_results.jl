using DataFramesMeta
using SharedArrays
#using CategorialArrays

function read_results(folder)
    c = collect_results(folder)
    return read_results(c)

end
function read_results!(folder)
    c = collect_results!(folder; update=true)
    return read_results(c)
end
function read_results(c::DataFrame)



    # calculate p < 0.05
    c = c[.!ismissing.(c.results), :]
    for r = 1:nrow(c)
        x = c[r, :]
        if "side" ∉ names(x.results)
            x.results[!, "side"] .= "twosided"
        end
    end

    c = @rtransform(c, :p = combine(groupby(:results, ["coefname", "test", "side"]), Symbol("pval") => x -> mean(x .<= 0.05)))

    c = @rtransform(c, :r = combine(groupby(:results, ["coefname", "test", "side"]), Symbol("runtime") => (x -> quantile(x, 0.05)) => Symbol("r_low"),
        Symbol("runtime") => (x -> quantile(x, 0.5)) => Symbol("r_median"),
        Symbol("runtime") => (x -> mean(x)) => Symbol("r_mean"),
        Symbol("runtime") => (x -> mean(winsor(x, prop=0.2))) => Symbol("r_winmean"),
        Symbol("runtime") => (x -> quantile(x, 0.95)) => Symbol("r_high")))

    c = @rtransform(c, :w = combine(groupby(:results, ["coefname", "test", "side"]), Symbol("warnings") => (x -> sum(x[1])) => Symbol("maxtime"), Symbol("warnings") => (x -> sum(x[2])) => Symbol("NLoptRoundoff"),
        Symbol("warnings") => (x -> sum(x[3])) => Symbol("maxfval")))

    # move result table to own columns

    c = @rtransform(c, :coefname = :p.coefname, :test = :p.test, :pval = :p.pval_function, :side = :p.side,
        :warn_maxtime = :w.maxtime, :warn_nlopt = :w.NLoptRoundoff, :warn_maxfval = :w.maxfval,
        :runtime_median = :r.r_median, :runtime_mean = :r.r_mean, :runtime_winmean = :r.r_winmean, :runtime_low = :r.r_low, :runtime_high = :r.r_high)

    c = flatten(c, [:coefname, :test, :pval, :side, :warn_maxfval, :warn_nlopt, :warn_maxtime, :runtime_median, :runtime_low, :runtime_high, :runtime_mean, :runtime_winmean])[:, Not([:r, :w, :p, :results])]




    # move result table to own columns

    # replace missing statsmethod with permutation
    c.statsMethod[ismissing.(c.statsMethod)] .= "permutation"
    #disallowmissing!(c.statsMethod)
    c.pval[c.statsMethod.=="pBoot"] = 1 .- c.pval[c.statsMethod.=="pBoot"]

    # rename formulas
    @rtransform!(c, :f_simple = simpleFormula(:f))
    @transform!(c, :f_simple = categorical(:f_simple, levels=["1|s", "1+a||s", "1+a|s", "1+a|s + 1+a|i"]))


    # rename σs
    @rtransform!(c, :σs_simple = simpleσs(:σs))
    @transform!(c, :σs_simple = categorical(:σs_simple, levels=["1|s", "1+1*a|s", "4+1*a|s", "1+4*a|s", "1+1*a|s+1|i", "1+1*a|s+1+1*a|i", "1+1*a|s+1+4*a|i"]))


    # replace weird JLD names
    jldreplace = x -> replace(string(x),
        "JLD2.ReconstructedTypes" => "",
        "##" => "",
        "var" => "",
        "." => "",
        "MixedModels" => "",
        "Permutations" => "",
        "#" => "",
        r"[0-9\(\)]*" => "",)

    c.blupMethod = jldreplace.(c.blupMethod)
    c.inflationMethod = jldreplace.(c.inflationMethod)
    return c
end


simpleFormula(f::FormulaTerm) = simpleFormula(string(f.rhs) |> x -> replace(x, " " => ""))
function simpleFormula(f::AbstractString)
    return f == "1+condition+:(1|subj)" ? "1|s" :
           f == "1+condition+:((1+condition)|subj)" ? "1+a|s" :
           f == "1+condition+:(zerocorr((1+condition)|subj))" ? "1+a||s" :
           f == "1+condition+:((1+condition)|subj)+:((1+condition)|item)" ? "1+a|s + 1+a|i" : "can't parse $f"
end


function simpleσs(σs)
    return σs == [[1.0, 0], [0.0, 0]] ? "1|s" :
           σs == [[1.0, 1], [0.0, 0]] ? "1+1*a|s" :
           σs == [[1.0, 1]] ? "1+1*a|s" :
           σs == [[4.0, 1], [0.0, 0]] ? "4+1*a|s" :
           σs == [[1.0, 1], [1.0, 0]] ? "1+1*a|s+1|i" :
           σs == [[1.0, 1], [1.0, 4]] ? "1+1*a|s+1+4*a|i" :
           σs == [[1.0, 1], [1.0, 1]] ? "1+1*a|s+1+1*a|i" :
           σs == [[1.0, 4], [0.0, 0]] ? "1+4*a|s" : "undefined"

end

function simpleDefaultParameters()

    def = DataFrame(Ref(dict_list(defaultParameters())[1]))
    def["reml"] = false
    @transform!(def, @byrow(:f_simple = simpleFormula(:f)),
        @byrow(:σs_simple = simpleσs(:σs)))

    @rtransform!(def, :residualMethod = string(:residualMethod), :blupMethod = string(:blupMethod), :inflationMethod = string(:inflationMethod); renamecols=false)

    return def
end


function subselectDF(a; skip=[], def=simpleDefaultParameters(), debug=false, clip=0.1)
    def = deepcopy(def)
    def.coefname .= "condition: B"
    def.test .= "z"
    def.side .= :twosided

    skipAll = hcat(skip..., ["test", "σs", "f"]...)
    debug ? @show(skipAll) : ""
    onstring = setdiff(names(def), skipAll)

    #ix = Vector{Bool}(true,nrow(c))
    ix = fill(true, nrow(a))

    for k in onstring
        v = def[1, k]
        ix_l = isequal.(a[!, k], Ref(v)) .|| isequal.(a[!, k], "missing") .|| isequal.(a[!, k], missing)

        ix = ix .&& ix_l
        debug ? println("$k - $v | $(sum(ix_l)) | remaining: $(sum(ix))") : ""
        @assert sum(ix_l) > 0 "$k,$v - $(unique(a[:,k]))"
    end

    debug ? @show(sum(ix)) : ""

    a = select(a[ix, :], Not(:f))

    #@show unique(a.test)
    if !("test" ∈ skip)
        @rsubset!(a, :test !== "β")
    end
    #@show unique(a.test)
    a = @transform(a, @byrow :pval = ((:pval > clip) ? clip : :pval))
    return a
end
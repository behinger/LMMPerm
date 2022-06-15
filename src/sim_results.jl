function read_results(folder)
    c = collect_results(folder);
    return read_results(c)
    
end
function read_results!(folder)
    c = collect_results!(folder);
    return read_results(c);
end
    function read_results(c::DataFrame)

    

	# calculate p < 0.05

    for r = 1:nrow(c)
        x = c[r,:]     
        if "side"  ∉ names(x.results)
            x.results[!,"side"] .= "twosided"
        end
    end

c = @rtransform(c,:p = combine(groupby(:results,["coefname","test", "side"]),Symbol("pval") => x->mean(x .<= .05)))

	# move result table to own columns
c = @rtransform(c,:coefname = :p.coefname,:test=:p.test,:pval=:p.pval_function,:side=:p.side)

c = flatten(c,[:coefname,:test,:pval,:side])[:,Not([:p,:results])]

	# replace missing statsmethod with permutation
	c.statsMethod[ismissing.(c.statsMethod)] .= "permutation"
	#disallowmissing!(c.statsMethod)
	c.pval[c.statsMethod .== "pBoot"] = 1 .-c.pval[c.statsMethod .== "pBoot"]

	# rename formulas
	@transform!(c,@byrow :f_simple = 
		:f == "1+condition+:(1|subj)" ? "1|s" : 
		:f == "1+condition+:((1+condition)|subj)" ? "1+a|s" : 
		:f == "1+condition+:((1+condition)|subj)+:((1+condition)|item)" ? "1+a|s + 1+a|i" : "")
	
	# rename σs
	@transform!(c,@byrow :σs_simple = 
		:σs == [[1., 0],[0., 0]] ? "1|s" : 
		:σs == [[1., 1],[0., 0]] ? "1+1*a|s" : 
        :σs == [[1., 1]] ? "1+1*a|s" : 
		:σs == [[4., 1],[0., 0]] ? "4+1*a|s" : 
		:σs == [[1., 1],[1., 0]] ? "1+1*a|s+1|i" : 
		:σs == [[1., 1],[1., 4]] ? "1+1*a|s+1+4*a|i" : 
		:σs == [[1., 1],[1., 1]] ? "1+1*a|s+1+1*a|i" : 
		:σs == [[1., 4],[0., 0]] ? "1+4*a|s" : "undefined")


	# replace weird JLD names
	jldreplace = x->replace(string(x),
	"JLD2.ReconstructedTypes.var\"##MixedModels.#"=>"",
	"JLD2.ReconstructedTypes.var\"##MixedModelsPermutations.#"=>"",
	r"[\d]*\"()"=>"")

	c.blupMethod = jldreplace.(c.blupMethod)
	c.inflationMethod = jldreplace.(c.inflationMethod)
return c
end
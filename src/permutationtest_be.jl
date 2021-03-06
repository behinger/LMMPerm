using Tables
permutationtest_be(perm::MixedModelPermutation, model::LinearMixedModel) = permutationtest_be(perm::MixedModelPermutation, model, :twosided)

"""
    permutationtest(perm::MixedModelPermutation, model, type=:greater)

Perform a permutation using the already computed permutation and given the observed values.

The `type` parameter specifies use of a two-sided test (`:twosided`) or the directionality of a one-sided test
(either `:lesser` or `:greater`, depending on the hypothesized difference to the null hypothesis).

See also [`permutation`](@ref).

To account for finite permutations, we implemented the conservative method from Phipson & Smyth 2010:
 Permutation P-values Should Never Be Zero:Calculating Exact P-values When Permutations Are Randomly Drawn
 http://www.statsci.org/webguide/smyth/pubs/permp.pdf 

"""
function permutationtest_be(perm::MixedModelPermutation, model; type::Symbol=:twosided,β::AbstractVector=zeros(length(coef(model))), statistic=:z)
    #@warn """This method is known not to be fully correct.
    #         The interface for this functionality will likely change drastically in the near future."""
    # removed due to distributed run

    if type == :greater || type  == :twosided
        comp = >=
    elseif type == :lesser
        comp = <=
    else
        throw(ArgumentError("Comparison type $(type) unsupported"))
    end
    if statistic == :z
        x = coeftable(model)
        ests = Dict(Symbol(k) => v for (k,v) in zip(coefnames(model), x.cols[x.teststatcol]))
    elseif statistic == :β
        ests = Dict(Symbol(k) => v for (k,v) in zip(coefnames(model), coef(model)))
    else
        error("statistic not implemented yet")
    end

    perms = columntable(perm.coefpvalues)

    dd = Dict{Symbol, Vector}()

    for (ix,k) in enumerate(Symbol.(coefnames(model)))
        dd[k] = perms[statistic][perms.coefname .== k]

        
        push!(dd[k],ests[k]) # simplest approximation to ensure p is never 0 (impossible for permutation test)
        if type == :twosided
            # in case of testing the betas, H0 might be not β==0, therefore we have to remove it here first before we can abs
            # the "z's" are already symmetric around 0 regardless of hypothesis.
            if statistic == :β
                #println(β[ix])
                dd[k]  .= dd[k]  .- β[ix]
                ests[k] = ests[k] - β[ix]
            end

              dd[k]  .= abs.(dd[k])
              ests[k] = abs(ests[k])
        end
 
        
    end

    # short way to calculate: 
    # b = sum.(abs.(permDist).>=abs.(testValue)); (twosided)
    # Includes the conservative correction for approximate permutation tests
    # p_t = (b+1)/(nperm+1); 

    # (with comp being <=) Note that sum(<=(ests),v) does the same as  sum(v .<=ests) (thus "reversed" arguments in the first bracket)
    results = (; (k=> sum(comp(ests[k]),v)/length(v) for (k,v) in dd)...)
    #results = (; (k => (1+sum(comp(ests[k]),v))/(1+length(v)) for (k,v) in dd)...)
    
    return results
end

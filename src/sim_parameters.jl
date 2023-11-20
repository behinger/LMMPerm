using DrWatson
using StatsModels # for @formula
using MixedModels # for zercorr
function defaultFormulas()
    f1 =  @formula(dv ~ 1 + condition  + (1|subj))
    f2 =  @formula(dv ~ 1 + condition  + zerocorr(1+condition|subj))
    f3 =  @formula(dv ~ 1 + condition  + (1+condition|subj))
    f4 =  @formula(dv ~ 1 + condition  + (1+condition|subj) + (1+condition|item))
    return f1,f2,f3,f4
end

function defaultParameters()
    _,_,f3,_ = defaultFormulas()
    return Dict(
        "nRep" => 5000,
        "statsMethod" => ["permutation"], #"waldsT","pBoot","LRT","KenwardRoger"
        "f" => [f3],#f1,f2,f4 #int-only, zerocorr, max, max+item
        "σs" => [[[1., 1.],[0.,0.]]],
        "β" => [[0., 0.]],
        "errorDistribution" => ["normal",],#"tdist","skewed"
        "imbalance" => [nothing],#"subject","trial",
        "nSubject" => [30],
        "nItemsPerCondition" => [30],
        "nPerm"=> [@onlyif(("statsMethod"=="permutation")|("statsMethod"=="pBoot"),1000)],
        "blupMethod" => [@onlyif("statsMethod"=="permutation",ranef)],#olsranef,
        "residualMethod" => [@onlyif("statsMethod"=="permutation",:shuffle)],# @onlyif("statsMethod"=="permutation",:signflip)
        "inflationMethod" => [@onlyif("statsMethod" == "permutation",MixedModelsPermutations.inflation_factor)],#, "noScaling"],
        "σ" => 1.,
    )
end
function getParamList(task)
    f1,f2,f3,f4 = defaultFormulas()

   default = defaultParameters()
    if task == 1
        paramList = Dict(
            "f" => [f1,f2,f3,f4],
            "σs" => [@onlyif("f"!= f4, [[1., 0.],[0.,0.]]),
                     @onlyif("f"!= f4, [[1., 1.],[0.,0.]]),  
                     @onlyif("f"!= f4, [[1., 4.],[0.,0.]]),
                     @onlyif("f"!= f4, [[4., 1.],[0.,0.]]),
        
                     @onlyif("f"== f4, [[1., 1.], [1., 0.]]),
                     @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
                     @onlyif("f"== f4, [[1., 1.], [1., 4.]])],

            "blupMethod" => [ranef,@onlyif("f"!= f4,olsranef)],
            "inflationMethod" => [MixedModelsPermutations.inflation_factor,"noScaling"],
            "residualMethod" => [:signflip,:shuffle],            
        )
        elseif task == 2
        #----
        # H1 test
        # test some special cases of \sigmas / f1 
        paramList = Dict(
            "f" => [f1,f3,f4],
            "σs" => [@onlyif("f"== f1, [[1., 0.], [0.,0.]]),
                     @onlyif("f"== f3, [[1., 1.], [0.,0.]]),  
                     @onlyif("f"== f3, [[1., 4.], [0.,0.]]),
                     @onlyif("f"== f4, [[1., 1.], [1., 1.]]),
                     ],
            "β" =>[[0., 0.],[0., 0.1],[0., 0.2],[0., .3],[0., 0.5]],
        )
        
        elseif task == 3
        #----
        # Power calculations
        paramList = Dict(
            "statsMethod" => ["waldsT","pBoot","permutation","LRT","KenwardRoger"], # if this is "missing" we run permutation for backward compatibility
            "β" => [[0., 0.],[0., 0.1],[0., 0.2],[0., .3],[0., 0.5]]        
        )
        
        
        elseif task == 4
        #-----
        # Varying N
        paramList = Dict(
            "statsMethod" => ["waldsT","pBoot","permutation","LRT","KenwardRoger"], # if this is "missing" we run permutation for backward compatibility
            "errorDistribution" => ["normal","tdist","skewed","skewed_40"],
            "β" => [[0., 0.],[0., 0.3]],
            
            "blupMethod" => [@onlyif("statsMethod"=="permutation",ranef),
                             @onlyif("statsMethod"=="permutation",olsranef)],
            "nSubject" => [4,10,30],    
            "nItemsPerCondition" => [2,10,30,50],
        )
        
        elseif task == 5
            #-----
            # Errordistributions + balancing
            paramList = Dict(
                "statsMethod" => ["waldsT","pBoot","permutation","LRT","KenwardRoger"], # if this is "missing" we run permutation for backward compatibility
                "errorDistribution" => ["normal","tdist","skewed"],
                "imbalance" => ["subject","trial"],
                "σs" => [[[1., 1.],[0.,0.]]],
                "residualMethod" => [@onlyif("statsMethod"=="permutation",:shuffle),@onlyif("statsMethod"=="permutation",:signflip)],
                "nSubject" => [10,30],
            )
        end
        return merge(default,paramList)
end

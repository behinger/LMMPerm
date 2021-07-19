using MixedModels
using MixedModelsSim
n_subj = 40
n_item = 80
subj_btwn = Dict(:age => ["old", "young"])
item_btwn = Dict(:frequency => ["high", "low"])
both_win = Dict(:context => ["matched", "unmatched"])

rng = MersenneTwister(42)  # specify our random number generator for reproducibility
design = simdat_crossed(rng, n_subj, n_item;
                        subj_btwn = subj_btwn,
                        item_btwn = item_btwn,
                        both_win = both_win)


contrasts = Dict(:age => EffectsCoding(base="young"),
                 :frequency => EffectsCoding(base="high"),
                 :context => EffectsCoding(base="matched"))
form = @formula(dv ~ 1 + context +                    (1 + context | subj) )
m0 = fit(MixedModel, form, design; contrasts=contrasts)

re_subj = create_re(1.5, 0.5)

MixedModelsSim.update!(m0, re_subj)
permutation(10,m0)
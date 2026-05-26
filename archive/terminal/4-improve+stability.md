
----------
# python improve.py --mode train --csv  devset_videolist_GT.csv --feat features/ --llm  llm_scalar_cache_v2.json
Detected LLM keys: ['emotional_valence', 'human_presence', 'message_simplicity', 'novelty_surprise', 'narrative_arc', 'brand_prominence', 'repetition_hooks', 'direct_memorability']

Spearman ρ — LLM scalars vs targets:
  dimension                  ρ_video   ρ_brand
  emotional_valence          +0.132     +0.068
  human_presence             +0.144     +0.094
  message_simplicity         +0.121     +0.100
  novelty_surprise           +0.154     +0.089
  narrative_arc              +0.140     +0.106
  brand_prominence           -0.108     +0.049
  repetition_hooks           +0.083     +0.091
  direct_memorability        +0.153     +0.118

══════════════════════════════════════════════════
  TARGET: memorability_score
══════════════════════════════════════════════════
  meta+llm                        ρ = 0.2380
  meta+llm+cross                  ρ = 0.4047
  meta+face+llm                   ρ = 0.2364
  meta+face+llm+cross             ρ = 0.3980

══════════════════════════════════════════════════
  TARGET: brand_memorability
══════════════════════════════════════════════════
  meta+llm                        ρ = 0.2448
  meta+llm+cross                  ρ = 0.4036
  meta+face+llm                   ρ = 0.2493
  meta+face+llm+cross             ρ = 0.3961



----------
# python stability.py --csv   devset_videolist_GT.csv --feat  features/ --llm   llm_scalar_cache_v2.json --seeds 20

Running CV across 20 seeds...

  seed   ρ_video   ρ_brand
     0   0.2793    0.2392
     1   0.2792    0.2244
     2   0.3344    0.2336
     3   0.2901    0.2084
     4   0.3174    0.2342
     5   0.2546    0.1788
     6   0.2872    0.1450
     7   0.3692    0.3142
     8   0.2865    0.2319
     9   0.3085    0.2087
    10   0.2825    0.1732
    11   0.2436    0.1720
    12   0.2729    0.2044
    13   0.2769    0.2296
    14   0.3020    0.2235
    15   0.3038    0.2106
    16   0.2471    0.1747
    17   0.3130    0.2430
    18   0.4131    0.3342
    19   0.3083    0.2035

════════════════════════════════════════
  video_mem  : mean=0.2985  std=0.0389  min=0.2436  max=0.4131
  brand_mem  : mean=0.2193  std=0.0437  min=0.1450  max=0.3342




----------
# The 0.40 from seed=42 was an outlier. The honest numbers are:

Video memorability: mean ρ = 0.30 ± 0.04
Brand memorability: mean ρ = 0.22 ± 0.04
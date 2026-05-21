
----------
>> python improve.py --mode train --csv  devset_videolist_GT.csv --feat features/ --llm  llm_scalar_cache_v2.json
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
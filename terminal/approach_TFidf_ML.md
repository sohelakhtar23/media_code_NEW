## TF-IDF approach

`python approach_tfidf.py   --mode train   --train-csv devset_videolist_GT.csv   --stt       devset-stt/   --llm llm_scalar_cache_v2.json`                                                                                                               
Training set: 339 videos         

  memorability_score
    CV ρ = 0.2697
    config: n_comp=50  threshold=0.07  use_llm=True

  brand_memorability
    CV ρ = 0.1801
    config: n_comp=100  threshold=0.01  use_llm=False
    

`python approach_tfidf.py   --mode test   --train-csv devset_videolist_GT.csv   --test-csv  predict/testset_videolist_.csv   --stt       devset-stt/   --stt-test  predict/testset-stt/   --llm  llm_scalar_cache_v2.json   --llm-test  predict/llm_scalar_cache_test.json --out predict/`
Train: 339 videos  |  Test: 85 videos
  memorability_score: min=0.1875  max=1.0000
  brand_memorability: min=0.0833  max=0.8667

Saved → predict\predictions_tfidf.csv


## ML approach
`python approach_ml.py   --mode train   --train-csv devset_videolist_GT.csv   --feat features/   --llm llm_scalar_cache_v2.json`
Feature matrix: (339, 12)  (3d engagement + 8d LLM + 1d face_rate)

  memorability_score
    best CV ρ = 0.1271
    model     = Pipeline(steps=[('sc', StandardScaler()), ('m', Ridge(alpha=500))])

  brand_memorability
    best CV ρ = 0.0800
    model     = Pipeline(steps=[('sc', StandardScaler()), ('m', SVR(C=0.5))])


`python approach_ml.py   --mode test   --train-csv devset_videolist_GT.csv   --test-csv  predict/testset_videolist_.csv   --feat features   --llm llm_scalar_cache_v2.json   --llm-test  predict/llm_scalar_cache_test.json   --out predict/`

  memorability_score  CV ρ=0.1271

  brand_memorability  CV ρ=0.0800

Saved → predict\predictions_ml.csv

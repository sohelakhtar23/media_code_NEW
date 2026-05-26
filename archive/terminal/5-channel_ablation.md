# python channel_ablation.py --csv  devset_videolist_GT.csv --feat features/ --llm  llm_scalar_cache_v2.json

  combo                                ρ_mean   per-fold

  TARGET: memorability_score
  channel-aware (meta+llm)             0.2613   ['0.197', '0.347', '0.408', '0.153', '0.201']
  channel-agnostic (no ch enc)         0.0672   ['0.105', '0.003', '0.030', '0.300', '-0.103']
  best_model_aware                     Pipeline(steps=[('sc', StandardScaler()), ('m', Ridge(alpha=0.1))])
  best_model_agnostic                  XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             feature_weights=None, gamma=None, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=0.03, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=2,
             max_leaves=None, min_child_weight=None, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=100,
             n_jobs=None, num_parallel_tree=None, ...)

  TARGET: brand_memorability
  channel-aware (meta+llm)             0.2018   ['-0.010', '0.120', '0.477', '0.267', '0.155']
  channel-agnostic (no ch enc)         0.0257   ['0.156', '-0.017', '-0.046', '0.116', '-0.081']
  best_model_aware                     XGBRegressor(base_score=None, booster=None, callbacks=None,
             colsample_bylevel=None, colsample_bynode=None,
             colsample_bytree=None, device=None, early_stopping_rounds=None,
             enable_categorical=False, eval_metric=None, feature_types=None,
             feature_weights=None, gamma=None, grow_policy=None,
             importance_type=None, interaction_constraints=None,
             learning_rate=0.03, max_bin=None, max_cat_threshold=None,
             max_cat_to_onehot=None, max_delta_step=None, max_depth=2,
             max_leaves=None, min_child_weight=None, missing=nan,
             monotone_constraints=None, multi_strategy=None, n_estimators=100,
             n_jobs=None, num_parallel_tree=None, ...)
  best_model_agnostic                  Pipeline(steps=[('sc', StandardScaler()), ('m', SVR())])
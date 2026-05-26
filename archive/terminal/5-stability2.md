# python stability2.py --csv  devset_videolist_GT.csv --feat features/ --llm  llm_scalar_cache_v2.json

Channel-stratified fold composition:
  Fold composition:
    fold 1:  79 videos — ['Goldman Sachs']
    fold 2:  65 videos — ['UBS', 'Legal & General', 'Blackstone', 'Aon Assessment Solutions', 'Baillie Gifford UK', 'Santander Asset Management UK', 'M&G Investments']
    fold 3:  65 videos — ['jpmorgan', 'Legal & General Investment Management - LGIM', 'Invesco', 'Life at Capital Group', 'T. Rowe Price', 'Aviva UK', 'Baillie Gifford']
    fold 4:  65 videos — ['Aon', 'Allianz', 'HSBC UK', 'BMO Global Asset Management - EMEA', 'BNP Paribas', 'Aegon', 'BNP Paribas Leasing Solutions UK', 'AXA Investment Managers']
    fold 5:  65 videos — ['Credit Suisse', 'Vanguard', 'Janus Henderson Investment Trusts', 'Amundi', 'Aviva', 'Columbia Threadneedle Investments EMEA APAC', 'Natixis Investment Managers US', 'State Street Global Advisors', 'Capital Group']

  meta+llm (no stacking):
    video_mem  : mean=0.2530  folds=['0.198', '0.298', '0.363', '0.158', '0.248']
    brand_mem  : mean=0.1723  folds=['-0.116', '0.129', '0.467', '0.119', '0.262']

  meta+llm+cross (with stacking):
    video_mem  : mean=0.2769  folds=['0.204', '0.330', '0.370', '0.187', '0.293']
    brand_mem  : mean=0.1822  folds=['-0.116', '0.145', '0.489', '0.124', '0.269']

══════════════════════════════════════════════════
  Summary (stratified CV, no seed variance):
  combo                           ρ_video   ρ_brand
  meta+llm                        0.2530    0.1723
  meta+llm+cross                  0.2769    0.1822
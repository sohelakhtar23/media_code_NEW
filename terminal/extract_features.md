# python extract_frame_stats.py 
frame stats: 100%|██████████████████████████████████████████████████████████| 339/339 [01:38<00:00,  3.44it/s]
Frame stats shape: (339, 10)

Spearman ρ with targets:
  feature                    ρ_video   ρ_brand
  brightness_mean            -0.021     +0.000
  brightness_std             +0.005     -0.019
  saturation_mean            +0.036     -0.051
  saturation_std             -0.047     -0.039
  colorfulness_mean          +0.011     -0.021
  colorfulness_std           -0.016     -0.033
  face_rate                  +0.115     +0.020
  motion_mean                +0.030     +0.027
  motion_std                 -0.034     -0.044
  shot_cut_rate              +0.018     +0.022

Saved to features/frame_stats.npy


# python extract_visual.py
Using device: cuda

100%|████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 339/339 [12:24<00:00,  2.20s/it]
Visual feature matrix shape: (339, 3072)
Saved to features/
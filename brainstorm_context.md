# Brainstorming: Commercial Video Memorability Prediction

## The Problem

We are participating in **MediaEval 2026 — Predicting Commercial Memorability, Subtask 2**. The task is to predict two scores for each commercial video:

1. **Video memorability score** — how well viewers remember having seen the video
2. **Brand memorability score** — how well viewers remember which brand the video was for

Both scores are continuous values in [0, 1], derived from recall experiments
with human annotators. The task requires jointly predicting both scores for
85 unseen test videos, trained on 339 labelled videos.

---

## The Dataset — VIDEM

- **339 training videos**, **85 test videos** (no test labels)
- All videos are **commercial / institutional content** from financial brands
  (investment banks, asset managers, insurance companies)
- 32 unique channels in training, 9 in test
- **Goldman Sachs dominates**: 79 videos = 23% of training data
- Video durations range from 7 seconds to 94 minutes
- Memorability scores and brand memorability scores are strongly correlated
  (Spearman r ≈ 0.71)
- Number of annotators per video ranges from 12 to 51 (mean ≈ 17)

### Critical annotation protocol detail
Annotators were only allowed to watch **at most 1 minute** of each video,
regardless of its actual duration. This means:
- 85.8% of videos are longer than 60 seconds
- Median video is ~5 minutes but only 1 minute was watched


---

## Available Data

### 1. CSV metadata (`devset_videolist_GT.csv`)
Columns available for each video:
- `id` — YouTube video ID (e.g. `CbfCUSPN7KI`)
- `video_id` — integer index
- `channelName` — brand/channel name
- `title` — video title
- `description` — YouTube description (can be long)
- `tags` — comma-separated tags
- `durationSeconds` — full video duration
- `categoryName` —  e.g., News & Politics, Education, Science & Technology
- `viewsCount`, `likesCount`, `commentsCount`, `engagementRate`
- `nb_annotations` — number of human annotators for this video
- `memorability_score` — target 1
- `brand_memorability` — target 2
- `url` — YouTube video link 

### 2. STT transcripts (`devset-stt/{id}.txt`)
- Plain text, no timestamps
- Raw speech-to-text, no punctuation in most cases
- Some videos have empty transcripts (3 out of 339)
- Long videos can have very long transcripts

### 3. Video frames (`frames/{id}/*.jpg`)
- **Exactly 1 frame per second**, named in ascending order
- Frame filenames are sortable — first N files = first N seconds
- Frame resolution varies by video but is consistent within a video
- For a 90-minute video there are ~5400 frames
- For our purposes: first 60 files = first 60 seconds = what annotator watched

### 4. Raw video files available
- can access on request
---


# My pc configuration:
I have a NVIDIA RTX 4000 Ada Generation GPU (20 GB VRAM)
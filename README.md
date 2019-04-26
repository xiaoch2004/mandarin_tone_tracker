## Mandarin Tone Tracker

The mandarin_tone_tracker plots mandarin tone contour 
The code use [crepe](https://github.com/marl/crepe) to extract pitch. It also adds these post-processing methods to adapt mandarin frequency characteristics:
- Eliminate frames with too-low frequencies (default threshold 100.0Hz)
- Eliminate frames with too-low confidence (default threshold 0.40)
- Eliminate frames with too-low energy (default threshold 1e-2)

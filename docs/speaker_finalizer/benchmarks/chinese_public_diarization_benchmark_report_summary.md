# 中文 Diarization Benchmark Summary

- Speed ranking: nemo_speakernet (0.0641) > nemo_titanet_small (0.0948) > 3dspeaker_eres2net (0.2738)
- DER ranking: 3dspeaker_eres2net (0.2187) > nemo_speakernet (0.2825) > nemo_titanet_small (0.3682)

## Combined speed

| backend            |   successful_files |   failed_files |   total_audio_hours |   mean_RTF |   median_RTF |   p95_RTF |
|:-------------------|-------------------:|---------------:|--------------------:|-----------:|-------------:|----------:|
| 3dspeaker_eres2net |                 48 |              0 |             27.7069 |     0.2738 |       0.2891 |    0.3202 |
| nemo_speakernet    |                 48 |              0 |             27.7069 |     0.0641 |       0.064  |    0.0713 |
| nemo_titanet_small |                 48 |              0 |             27.7069 |     0.0948 |       0.0972 |    0.1042 |


## Combined DER

| backend            |   mean_DER_relaxed |   median_DER_relaxed |   p95_DER_relaxed |   mean_miss |   mean_fa |   mean_confusion |   mean_JER |
|:-------------------|-------------------:|---------------------:|------------------:|------------:|----------:|-----------------:|-----------:|
| 3dspeaker_eres2net |             0.2187 |               0.173  |            0.4849 |      0.0084 |    0.0424 |           0.168  |     0.4362 |
| nemo_speakernet    |             0.2825 |               0.3042 |            0.4629 |      0.0084 |    0.0424 |           0.2317 |     0.5105 |
| nemo_titanet_small |             0.3682 |               0.3796 |            0.593  |      0.0083 |    0.042  |           0.318  |     0.6116 |

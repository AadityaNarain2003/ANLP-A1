### Issues
- Not having a GPU ( have Intel Iris)
- Not having Ada access
- Collab/Kaggle Runtime always disconnecting

### Perplexity of Q1 ( done for all sentences )

Epoch 1/10
Training Loss: 6.7000
Validation Loss: 6.2619
Perplexity: 524.21
Epoch 2/10
Training Loss: 6.1773
Validation Loss: 6.0255
Perplexity: 413.86
Epoch 3/10
Training Loss: 5.9814
Validation Loss: 5.8923
Perplexity: 362.25
Epoch 4/10
Training Loss: 5.8518
Validation Loss: 5.8031
Perplexity: 331.32
Epoch 5/10
Training Loss: 5.7537
Validation Loss: 5.7388
Perplexity: 310.68
Epoch 6/10
Training Loss: 5.6786
Validation Loss: 5.6895
Perplexity: 295.73
Epoch 7/10
Training Loss: 5.6141
Validation Loss: 5.6520
Perplexity: 284.87
Epoch 8/10
Training Loss: 5.5587
Validation Loss: 5.6204
Perplexity: 275.99
Epoch 9/10
Training Loss: 5.5134
Validation Loss: 5.5960
Perplexity: 269.34
Epoch 10/10
Training Loss: 5.4700
Validation Loss: 5.5776
Perplexity: 264.44

Test Loss: 5.4202
Perplexity: 225.92

### Perplexity of Q2 ( done for 2.5k sentences- the maximum my laptop was able to handle )

Epoch 1/10
Train Loss: 7.0429, Train Perplexity: 1144.7266
Val Loss: 6.6670, Val Perplexity: 786.0185
Epoch 2/10
Train Loss: 6.3765, Train Perplexity: 587.8642
Val Loss: 6.4607, Val Perplexity: 639.5339
Epoch 3/10
Train Loss: 6.1075, Train Perplexity: 449.2144
Val Loss: 6.2878, Val Perplexity: 537.9717
Epoch 4/10
Train Loss: 5.8728, Train Perplexity: 355.2478
Val Loss: 6.1726, Val Perplexity: 479.4434
Epoch 5/10
Train Loss: 5.6856, Train Perplexity: 294.5991
Val Loss: 6.1002, Val Perplexity: 445.9290
Epoch 6/10
Train Loss: 5.5127, Train Perplexity: 247.8285
Val Loss: 6.0449, Val Perplexity: 421.9399
Epoch 7/10
Train Loss: 5.3665, Train Perplexity: 214.1172
Val Loss: 6.0102, Val Perplexity: 407.5451
Epoch 8/10
Train Loss: 5.2337, Train Perplexity: 187.4777
Val Loss: 5.9895, Val Perplexity: 399.2059
Epoch 9/10
Train Loss: 5.0996, Train Perplexity: 163.9554
Val Loss: 5.9806, Val Perplexity: 395.6974
Epoch 10/10
Train Loss: 4.9732, Train Perplexity: 144.4873
Val Loss: 5.9798, Val Perplexity: 395.3797

Test Loss: 5.9196, Test Perplexity: 372.2476

### Perplexity of Q3 ( done for 2.5k sentences- the maximum my laptop was able to handle )
Epoch 1/10
Train Loss: 5.3261, Train Perplexity: 205.6439
Val Loss: 3.6416, Val Perplexity: 38.1539
Epoch 2/10
Train Loss: 3.5238, Train Perplexity: 33.9127
Val Loss: 3.2924, Val Perplexity: 26.9086
Epoch 3/10
Train Loss: 3.2228, Train Perplexity: 25.0977
Val Loss: 3.5519, Val Perplexity: 34.8791
Epoch 4/10
Train Loss: 2.8473, Train Perplexity: 17.2415
Val Loss: 2.7358, Val Perplexity: 15.4226
Epoch 5/10
Train Loss: 2.6608, Train Perplexity: 14.3074
Val Loss: 2.6996, Val Perplexity: 14.8742
Epoch 6/10
Train Loss: 2.6508, Train Perplexity: 14.1657
Val Loss: 2.9868, Val Perplexity: 19.8223
Epoch 7/10
Train Loss: 2.7535, Train Perplexity: 15.6979
Val Loss: 2.9452, Val Perplexity: 19.0154
Epoch 8/10
Train Loss: 2.7321, Train Perplexity: 15.3650
Val Loss: 2.5997, Val Perplexity: 13.4592
Epoch 9/10
Train Loss: 2.5019, Train Perplexity: 12.2058
Val Loss: 2.5917, Val Perplexity: 13.3530
Epoch 10/10
Train Loss: 2.4930, Train Perplexity: 12.0977
Val Loss: 2.5592, Val Perplexity: 12.9255

Test Loss: 2.7938

### For Comparitive Analysis - Q1 on first 2.5k sentences

Epoch 1/10
Training Loss: 7.3588
Validation Loss: 6.8443
Perplexity: 938.49
Epoch 2/10
Training Loss: 6.8544
Validation Loss: 6.6204
Perplexity: 750.24
Epoch 3/10
Training Loss: 6.5517
Validation Loss: 6.5208
Perplexity: 679.10
Epoch 4/10
Training Loss: 6.4171
Validation Loss: 6.4464
Perplexity: 630.42
Epoch 5/10
Training Loss: 6.3403
Validation Loss: 6.4322
Perplexity: 621.54
Epoch 6/10
Training Loss: 6.2908
Validation Loss: 6.4213
Perplexity: 614.81
Epoch 7/10
Training Loss: 6.2677
Validation Loss: 6.3963
Perplexity: 599.65
Epoch 8/10
Training Loss: 6.2254
Validation Loss: 6.3779
Perplexity: 588.69
Epoch 9/10
Training Loss: 6.1994
Validation Loss: 6.4432
Perplexity: 628.41
Epoch 10/10
Training Loss: 6.2048
Validation Loss: 6.3793
Perplexity: 589.51

Test Loss: 6.2995

#### Model Strength

1. Tranformer- 2.7938(Loss)
2. LSTM- 5.9196(Loss)
3. FFN- 6.2995(Loss)


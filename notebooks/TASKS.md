## Accuracy

- #### Does using different scales for features improve accuracy?

Using one, concatenating them, etc.

- #### Can rstsf be made into a feature version?

RSTSF computes similarities with the target. Can that be delayed and pushed into an ML model?

- #### What other TSC models to add?

- #### Training one model n times. Is it better to have n columns or average the probabilities?


## Training Performance

- #### Reuse the spawned processes for stacking

- #### Long startup for processes

 ```
[98/112] Processing: Dataset=PowerCons, Fold=0, Model=loky-stacker-v7-soft-filter-ridge
[0.00s] Starting executor with 4 workers, run_dir=./tscglue/faec019a8201415b
[0.01s] Saved X and y to disk in 0.00s
[0.11s] Computed QUANT features (180, 1471) (2.02 MB) in 0.1079s
[0.11s] Starting repetition 0
[0.52s] Computed MultiRocket features (180, 49728) (68.29 MB) in 0.4098s
[0.77s] Computed Hydra features (180, 5120) (7.03 MB) in 0.2495s
[2.10s] Computed RDST features (180, 30000) (41.20 MB) in 1.3320s
[2.11s] Starting training with 4 workers for 40 models
[65.86s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.5021s for f-0/r-0 (2.60 MB) <---- Why this time gap?
[66.47s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.5423s for f-1/r-0 (2.60 MB)
[66.85s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.7291s for f-2/r-0 (2.60 MB)
[66.88s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.7262s for f-3/r-0 (2.60 MB)
[67.02s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.7094s for f-4/r-0 (2.60 MB)
[67.18s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.6607s for f-5/r-0 (2.60 MB)
[67.58s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.6765s for f-6/r-0 (2.60 MB)
[67.60s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.6749s for f-7/r-0 (2.60 MB)
[67.72s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.6570s for f-8/r-0 (2.60 MB)
[67.88s] Trained multirockethydra-bestk-p-ridgecv_r0 in 0.6411s for f-9/r-0 (2.60 MB)
[67.88s] Completed training for model multirockethydra-bestk-p-ridgecv_r0
[67.90s] OOF acc for model multirockethydra-bestk-p-ridgecv_r0: 0.9722222222222222
[68.06s] Trained quant-etc_r0 in 0.4296s for f-0/r-0 (0.35 MB)
[68.06s] Trained quant-etc_r0 in 0.4056s for f-1/r-0 (0.34 MB)
[68.17s] Trained quant-etc_r0 in 0.4114s for f-2/r-0 (0.34 MB)
[68.33s] Trained quant-etc_r0 in 0.4014s for f-3/r-0 (0.33 MB)
[68.52s] Trained quant-etc_r0 in 0.4190s for f-4/r-0 (0.34 MB)
[68.55s] Trained quant-etc_r0 in 0.4404s for f-5/r-0 (0.34 MB)
[68.64s] Trained quant-etc_r0 in 0.4248s for f-6/r-0 (0.34 MB)
[68.75s] Trained quant-etc_r0 in 0.3821s for f-7/r-0 (0.33 MB)
[68.93s] Trained rdst-p-ridgecv_r0 in 0.2558s for f-0/r-0 (0.92 MB)
[68.96s] Trained quant-etc_r0 in 0.3879s for f-8/r-0 (0.32 MB)
[68.99s] Trained quant-etc_r0 in 0.4070s for f-9/r-0 (0.33 MB)
[68.99s] Completed training for model quant-etc_r0
[69.01s] OOF acc for model quant-etc_r0: 0.9888888888888889
[69.15s] Trained rdst-p-ridgecv_r0 in 0.3498s for f-1/r-0 (0.92 MB)
[69.29s] Trained rdst-p-ridgecv_r0 in 0.2968s for f-2/r-0 (0.92 MB)
[69.31s] Trained rdst-p-ridgecv_r0 in 0.2613s for f-4/r-0 (0.92 MB)
[69.43s] Trained rdst-p-ridgecv_r0 in 0.4244s for f-3/r-0 (0.92 MB)
[69.55s] Trained rdst-p-ridgecv_r0 in 0.2342s for f-6/r-0 (0.92 MB)
[69.60s] Trained rdst-p-ridgecv_r0 in 0.2188s for f-7/r-0 (0.92 MB)
[69.85s] Trained rdst-p-ridgecv_r0 in 0.3281s for f-8/r-0 (0.92 MB)
[69.90s] Trained rdst-p-ridgecv_r0 in 0.3018s for f-9/r-0 (0.92 MB)
[70.03s] Trained rdst-p-ridgecv_r0 in 0.7863s for f-5/r-0 (0.92 MB)
[70.03s] Completed training for model rdst-p-ridgecv_r0
[70.07s] OOF acc for model rdst-p-ridgecv_r0: 0.9611111111111111
[72.38s] Trained rstsf_r0 in 2.6499s for f-0/r-0 (0.31 MB)
[74.45s] Trained rstsf_r0 in 2.0358s for f-4/r-0 (0.31 MB)
[74.65s] Trained rstsf_r0 in 4.3919s for f-3/r-0 (0.31 MB)
[75.15s] Trained rstsf_r0 in 5.1548s for f-2/r-0 (0.31 MB)
[75.94s] Trained rstsf_r0 in 5.9869s for f-1/r-0 (0.32 MB)
[76.49s] Trained rstsf_r0 in 2.0039s for f-5/r-0 (0.31 MB)
[78.57s] Trained rstsf_r0 in 2.0496s for f-9/r-0 (0.31 MB)
[79.32s] Trained rstsf_r0 in 4.0803s for f-7/r-0 (0.31 MB)
[79.61s] Trained rstsf_r0 in 4.9048s for f-6/r-0 (0.31 MB)
[80.00s] Trained rstsf_r0 in 3.9827s for f-8/r-0 (0.31 MB)
[80.00s] Completed training for model rstsf_r0
[80.03s] OOF acc for model rstsf_r0: 1.0
[82.01s] Completed repetition 0
[82.01s] Starting stacking model training (single pass)
[112.62s] Trained probability-ridgecv in 0.1174s for f-0 (0.00 MB) <---- Why this time gap?
[112.62s] Trained probability-ridgecv in 0.0051s for f-1 (0.00 MB)
[112.63s] Trained probability-ridgecv in 0.0056s for f-2 (0.00 MB)
[112.63s] Trained probability-ridgecv in 0.0048s for f-3 (0.00 MB)
[112.64s] Trained probability-ridgecv in 0.0048s for f-4 (0.00 MB)
[112.65s] Trained probability-ridgecv in 0.0047s for f-5 (0.00 MB)
[112.65s] Trained probability-ridgecv in 0.0057s for f-6 (0.00 MB)
[112.66s] Trained probability-ridgecv in 0.0051s for f-7 (0.00 MB)
[112.66s] Trained probability-ridgecv in 0.0045s for f-8 (0.00 MB)
[112.67s] Trained probability-ridgecv in 0.0048s for f-9 (0.00 MB)
[120.37s] OOF acc for model probability-ridgecv: 1.0
[120.37s] Completed all repetitions and stacking
[120.41s] Executor shutdown complete
[0.0000s] Starting prediction
Starting executor with 4 workers, run_dir=./tscglue/faec019a8201415b
[1.3090s] Computed features for prediction
[1.3913s] Feature arrays saved to mmap files
[1.3914s] Starting prediction with 4 workers for 40 first-level models
[43.8857s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.1522s <---- Why this time gap?
[44.0057s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.2337s
[44.0654s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.2954s
[44.2427s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.3062s
[44.3037s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.1697s
[44.3287s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.2246s
[44.5143s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.1342s
[44.5949s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.1869s
[44.6782s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.2833s
[44.7301s] Predicted quant-etc_r0 in 0.0116s
[44.7974s] Predicted quant-etc_r0 in 0.0252s
[44.8029s] Predicted quant-etc_r0 in 0.0127s
[44.8636s] Predicted quant-etc_r0 in 0.0120s
[44.8806s] Predicted quant-etc_r0 in 0.0073s
[44.9267s] Predicted quant-etc_r0 in 0.0095s
[44.9282s] Predicted multirockethydra-bestk-p-ridgecv_r0 in 0.2151s
[44.9686s] Predicted quant-etc_r0 in 0.0223s
[44.9801s] Predicted quant-etc_r0 in 0.0095s
[45.0214s] Predicted quant-etc_r0 in 0.0088s
[45.0316s] Predicted quant-etc_r0 in 0.0092s
[45.1344s] Predicted rdst-p-ridgecv_r0 in 0.0670s
[45.1617s] Predicted rdst-p-ridgecv_r0 in 0.0798s
[45.2006s] Predicted rdst-p-ridgecv_r0 in 0.1654s
[45.3419s] Predicted rdst-p-ridgecv_r0 in 0.1115s
[45.3947s] Predicted rdst-p-ridgecv_r0 in 0.1458s
[45.4281s] Predicted rdst-p-ridgecv_r0 in 0.2243s
[45.5347s] Predicted rdst-p-ridgecv_r0 in 0.0968s
[45.5997s] Predicted rdst-p-ridgecv_r0 in 0.0847s
[45.6687s] Predicted rdst-p-ridgecv_r0 in 0.1697s
[45.7027s] Predicted rdst-p-ridgecv_r0 in 0.1024s
[55.2105s] Predicted rstsf_r0 in 9.4529s
[59.8299s] Predicted rstsf_r0 in 4.5692s
[60.0108s] Predicted rstsf_r0 in 0.1381s
[64.8196s] Predicted rstsf_r0 in 19.0620s
[65.3046s] Predicted rstsf_r0 in 17.7354s
[65.6864s] Predicted rstsf_r0 in 20.0440s
[65.9672s] Predicted rstsf_r0 in 0.2202s
[66.1910s] Predicted rstsf_r0 in 6.1420s
[69.7951s] Predicted rstsf_r0 in 4.9220s
[70.0634s] Predicted rstsf_r0 in 4.6935s
[72.5194s] Completed all first-level model predictions
[72.6261s] Starting prediction with 4 workers for 10 stacking models
[135.4643s] Predicted probability-ridgecv in 0.0016s <---- Why this time gap?
[135.4667s] Predicted probability-ridgecv in 0.0013s
[135.4692s] Predicted probability-ridgecv in 0.0012s
[135.4717s] Predicted probability-ridgecv in 0.0012s
[135.4742s] Predicted probability-ridgecv in 0.0013s
[135.4771s] Predicted probability-ridgecv in 0.0013s
[135.4797s] Predicted probability-ridgecv in 0.0012s
[135.4824s] Predicted probability-ridgecv in 0.0013s
[135.4854s] Predicted probability-ridgecv in 0.0015s
[135.4879s] Predicted probability-ridgecv in 0.0012s
[140.1060s] Completed all stacking model predictions
Executor shutdown complete
```


## Other

- #### Proper cleanup on exit or kill

Script should clean up its data if killed or if it exits.

- #### Split code on tsc-glue and bechmarking 

- #### Handle what happens if one model/model raises error. 

- #### Precheck what models can even be ussed and what not (due to dependencies)
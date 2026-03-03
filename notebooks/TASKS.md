## Accuracy

- #### Does using different scales for features improve accuracy?

Using one, concatenating them, etc.

- #### Can rstsf be made into a feature version?

RSTSF computes similarities with the target. Can that be delayed and pushed into an ML model?

- #### What other TSC models to add?

- #### Training one model n times. Is it better to have n columns or average the probabilities?


## Other

- #### Proper cleanup on exit or kill

Script should clean up its data if killed or if it exits.

- #### Split code on tsc-glue and bechmarking 

- #### Handle what happens if one model/model raises error. 

- #### Precheck what models can even be ussed and what not (due to dependencies)
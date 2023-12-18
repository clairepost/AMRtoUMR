
# AMR to UMR split-role conversion
Model as described in the paper:


 - baseline.py : Uses only animacy inforamtion and rules to predict the UMR roles
 - base_nn.py: Uses only role_constraints to predict the UMR roles
 - nn_with_rules_weights.py: Uses animacy-informed rules + NN to predict the UMR roles

Each model can be rerun individually. Predictions are stored in /output. 
Results can be viewed and replicated in results.ipynb


A-Comparative-Evaluation-of-Active-Learning-Methods-in-Deep-Recommendation
==========================================================================


# Example Commands
### Data Split

For fine-tuning, to split ML1M dataset use
```
python3 getmovielens.py --implicit
```
For fine-tuning, to split Yelp dataset use
```
python3 getyelp.py --enable_implicit --name yelp/yelp_academic_dataset_review.json
```

For active learning, to split ML1M dataset use
```
python3 getmovielens.py --implicit --disable-validation
```
For active learning, to split Yelp dataset use
```
python3 getyelp.py --enable_implicit --name yelp/yelp_academic_dataset_review.json --disable_validation
```

### Single Run
For ML1M,
```
python3 main.py --path data/ --active_model Greedy --active_iteration 50
```
For Yelp,
```
python3 main.py --path data/ --active_model Greedy --epoch 300 --lamb 0.001 --rank 200 --active_iteration 50
```

### Other Run Examples
Please refer to `reproduce_ml1m_final_result.sh` and `reproduce_yelp_final_result.sh`.

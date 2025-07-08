# NCL-SR item test

## Requirements

For our running environment see requirements.txt

## Datasets
- We use the public datasets that can be downloaded from the url mentioned in the paper.
- One can use the true url of each dataset for the url function defined in the datasets.py files (e.g., beauty.py) to automatically download and process the files.
- One need to change PATH in config.py
## Scripts.
- Stage 1: train LRURec and E5 (Hybrid Retrieval)
   - Example
       ```
       python src/train_stage_1.py --mce_gamma_a  --mce_gamma_u  --num_replace  --dataset_code --lr --train_batch_size --val_batch_size  --print_freq  --epochs 5;
       python src/train_stage_2.py --mce_gamma_a  --mce_gamma_u  --num_replace  --dataset_code --lr --train_batch_size --val_batch_size  --print_freq  --epochs 3;
       python src/eval_e5.py       --mce_gamma_a  --mce_gamma_u  --num_replace  --dataset_code --test_batch_size;
       ```
   - Hyperparameters
      ```
      --mce_gamma_u             # \lambda_1 for uniformity in paper
      --mce_gamma_a             # \lambda_1 for alignment in paper
      --num_replace             # number of to-be-replaced items within each user profile
      --dataset_code            # select from 'beauty', 'games', 'toys_new', 'auto', 'office', 'sports'
      --lr                      # learning rate
      --train_batch_size        # training batch size
      --val_batch_size          # validation batch size
      --test_batch_size         # test batch size
      --print_freq              # frequency for printing validation results
      ```
    - After executing the scripts, the trained models and test scores will be automatically saved to a new folder `experiments`.

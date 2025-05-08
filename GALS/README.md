## GALS

### Repository Structure

The folder structure is primarily based on the original GALS repository. Below, we outline only the files and directories that have been modified to align with our framework. 
```
GALS/
├── approaches/
│   ├── base.py
|   ├── generic_cnn.py
│   └── coco_gender.py  
├── configs/
│   ├── coco_gals.yaml  (modify the attention folder based on the approach: median or GALS)
│   ├── coco_gals_mean.yaml  
│   ├── coco_gals_test.yaml
|   ├── food_attention.yaml
|   ├── food_attention_meat.yaml
|   ├── food_egals_mean.yaml
|   ├── food_egals_meat_mean.yaml
|   ├── food_egals_meat_median.yaml
|   ├── food_egals_median.yaml
|   ├── food_gals.yaml
|   ├── food_gals_meat.yaml
|   ├── food_gals_val.yaml
|   ├── food_meat_gals_val.yaml
|   ├── waterbirds_95_attention.yaml
|   ├── waterbirds_95_gals.yaml
|   ├── waterbirds_95_median.yaml
|   ├── waterbirds_95_gals_val.yaml
│   ├── waterbirds_100_gals.yaml  (modify the attention folder based on the approach: median or GALS)
│   ├── waterbirds_100_gals_mean.yaml
│   └── waterbirds_100_gals_val.yaml
├── data/  (created as stated in the main README file of the repo)
├── datasets/
│   ├── coco.py
|   ├── food.py (modify to work for red meat subset)
|   ├── food_gals.py (created to work for the meat experiment)
│   └── waterbirds.py  
├── notebooks/
│   ├── COCO_biased_balanced_set.ipynb  (used to align COCO img ids between ProbVLM and GALS framework)
│   └── Image_vis.ipynb  
├── utils/
│   ├── attention_utils.py  
│   └── general_util.py  (modify in init_wandb to adapt to your entity and project name)
└── wandb/  (needs to be created to save checkpoints of Weights & Biases)

```

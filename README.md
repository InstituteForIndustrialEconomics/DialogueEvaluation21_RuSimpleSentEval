# Solution for RuSimpleSentEval Competition

Create environment: `conda env create -f environment.yml`

Additional installations:
EASSE as described at organizers [github](https://github.com/dialogue-evaluation/RuSimpleSentEval#%D0%BE%D1%86%D0%B5%D0%BD%D0%BA%D0%B0-%D0%BA%D0%B0%D1%87%D0%B5%D1%81%D1%82%D0%B2%D0%B0)

`data` folder should contain csvs from https://github.com/dialogue-evaluation/RuSimpleSentEval and csvs from [WikiSimple-translated](https://drive.google.com/drive/folders/1jfij3KuiRbO_XoLiquSBP2mZafzPhrsL):

```
├── data
│   ├── WikiSimple-translated
│   │   ├── wiki_dev_cleaned_translated_sd.csv
|   |   ├── wiki_test_cleaned_translated_sd.csv
|   |   └── wiki_train_cleaned_translated_sd.csv
│   ├── dev_sents.csv
│   ├── public_test_only.csv
│   └── hidden_test_only.csv
...
```

`private_submission_check_100.csv` - sample of private submission annotated for loss of meaning of the original sentence 

## Pipeline

1. Dataset filtering
```
python prepare_datasets.py
```

2. Finetuning ruGPT3
```
python finetune_rugpt3.py --dataset wiki_part.csv \
                          --model sberbank-ai/rugpt3medium_based_on_gpt2 \
                          --output_dir result_rugpt3medium \
                          --log_dir logs_rugpt3medium \
                          --epochs 3 \
                          --batch_size 4 \
                          --gradient_accumulation_steps 2 \
                          --learning_rate 5e-5 \
                          --warmup_steps 500 \
                          --fp16 \
                          --seed 19
```

2. Get 5 candidates for output simplified sentence
```
python predict_5.py --input_test_path data/public_test_only.csv \
                    --input_dev_path data/dev_sents.csv \
                    --model_path result_rugpt3medium/checkpoint-45282 \
                    --predict_folder predictions \
                    --device cuda:0 \
                    --seed 19
```

2. Choose one prediction from generated candidates
```
python choose_prediction.py --input_test_path data/public_test_only.csv \
                            --input_dev_metrics_path data/prepared_data/dev_df_metrics.csv \
                            --test_predictions_path predictions/test_answers_5.csv \
                            --dev_predictions_path predictions/dev_answers_5.csv \
                            --submission_folder submissions \
                            --seed 19
```
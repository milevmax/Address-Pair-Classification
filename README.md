# Ukrainian Address Pair Classification

## Project Description
This project focuses on the classification of Ukrainian address pairs, determining whether two given addresses are the same or different.
For example:\
"ЧЕРКАСЬКА, РУСЬКА ПОЛЯНА, вул. 40 РОКІВ ПЕРЕМОГИ будинок 12" should match "ЧЕРКАСЬКА ОБЛ., СЕЛО РУСЬКА ПОЛЯНА ВУЛИЦЯ 40 РОКІВ ПЕРЕМОГИ БУД. 12"\
While "325000, Херсонська, Херсон, вул.Ілюши Кулика, 135А, кв.80" - "ХМЕЛЬНИЦЬКА ОБЛ., МІСТО ХМЕЛЬНИЦЬКИЙ ВУЛИЦЯ ВОЛОДИМИРСЬКА БУД. 105 КВ. 38" is different.

## Modules
- `data_augmentation.py`: Module for generating synthetic data using templates from `templates.py`.
- `templates.py`: Contains templates for address data generation.
- `model_development.ipynb`: Jupyter notebook detailing the biLSTM model development with a siamese architecture.
- `model_evaluation.ipynb`: Jupyter notebook presenting the evaluation of the model and proposing ways to enhance model performance.

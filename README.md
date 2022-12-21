# Master thesis
Code for my master thesis project related to information retrieval and semantic search.

## Install
To install dependencies run:

```console
$ pipenv install
```

## Run notebooks
You can run the notebooks on **Google Colab** by simply uploading them to your drive and opening them in Colab or you can run the locally with jupyter-lab:

``` console
$ pipenv run jupyter-lab
```

## Run scripts
To run scripts you should first enter the pipenv shell:

```console
$ pipenv shell
```

Then you can run any script like this:

```console
$ python3 <script_name>.py
```

## Structure
* [Dataset prep](/dataset_prep) - Data exploration, prep and train/dev/test split
* [Baselines](/baselines) - TF-IDF & dummy baselines
* [Fine-tuning embeddings (triplet loss)](/triplet_loss) - Prepare triplets & fine-tuning embeddings directly with triplet loss 
* [Emb + Classifier](/emb_classifier) - Random forest & ANN classifier on top of fine-tuned embeddings
* [End2End](/end2end) - BERT/RoBERTa/DeBERTaV3 End2End classifiers
* [End2End (Emb only)](/end2end_emb_only) - Eval emb from the End2End models
* [End2End Extra data](/end2end_extra_data) - End2End models with extra input features
* [Analysis steps prep and eval](/model_analysis) - Analysis experiments prep and eval

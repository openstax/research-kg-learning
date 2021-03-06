# TOKN Pytorch Relation Extraction Models 

The code in this folder is adapted from an open source [pytorch deep learning template](https://github.com/victoresque/pytorch-template). Refer to
the documentation provided at the link for more detailed information on the structure of this repo.

## Folder Structure
  ```
  re-pytorch-models/
  │
  ├── train.py - main script to start training
  ├── test.py - evaluation of trained model
  ├── predict.py - predict set of relations from input text and input list of terms
  │
  ├── config - folder that holds json configuration files 
  ├── parse_config.py - class to handle config file and cli options
  │
  ├── model/ - models, losses, and metrics
  │   ├── model.py: Contains pytorch model architectures
  │   ├── data_loaders.py: Contains data loader classes for reading in batches of data
  │   ├── metric.py: Contains metric functions used to evaluate the model
  │   ├── loss.py: Contains loss functions used to train the model
  │   └── trainer.py: Contains a trainer class used to train the model
  │
  ├── saved/
  │   ├── models/ - trained models are saved here
  │   └── log/ - default logdir for tensorboard and logging output
  │
  ├── logger/ - module for tensorboard visualization and logging
  │   ├── visualization.py
  │   ├── logger.py
  │   └── logger_config.json
  │  
  └── utils/ - Contains important data preprocessing functions for the predict script
  ```

## Installation & Environment

### Local

Install the required Python libraries:

`pip install -r requirements.txt`

Download the Stanford NLP models to enable the text preprocessing pipeline for prediction:

`import stanfordnlp; stanfordnlp.download('en')`

### AWS 

Training was done on a virtual AWS machine with GPU. We used the [AWS Deep Learning AMI (Ubuntu)](https://aws.amazon.com/blogs/machine-learning/get-started-with-deep-learning-using-the-aws-deep-learning-ami/) image on a p2.xlarge instance with a single GPU. The image comes with a bunch of different pre-configured conda environments. We used the pytorch_p36 environment (conda activate pytorch_p36) and then pip installed additional Python packages as needed.

## Training

The train.py script is responsible for training new models. It requires a json configuration file in order to run: 

`python train.py -c config/{name_of_config}.json`

See below for a description of the config file, but this specifies all the hyperparameters and the model architecture to be used. As part of the training process, tensorboard visualization information and a log of the terminal output will be written to the saved/logs folder. To visualize the loss and various metrics during or after training in tensorboard run:

`tensorboard --logdir saved/log` 

As part of the training process, a version of the model will be saved to disk each epoch if its performance on some validation metric is the best seen so far. These models get saved as follows:

`saved/models/{config_name}/{run_timestamp}/model_best.pth`

Additionally, a copy of the config file used to specify training will also be stored in this same directory. For testing and predicting, one can specify the path to the model_best.pth file for the model of choice to load in the pre-trained weights.

Besides making changes in the config file, one can also change the data splits to train and validate on by changing the argument to the data loader constructions on line 28 and 29 of train.py.

## Evaluating 

The test.py script is responsible for evaluating trained models. It requires a pre-trained model to evaluate as well as a data split to evaluate on as follows:

`python test.py -r saved/models/{config_name}/{run_timestamp}/model_best.pth -s {data_split}`

Valid data splits currently include debug (small data set for debugging purposes), train (70% of Life Biology KB relations), validation (15% of Life Biology KB relations), test (15% of Life Biology KB relations). 

This script will load the model and then run the given split through it generating predictions which will then be compared to the labels for evaluation. The script will output three files to the same directory where the saved model lives:

`saved/models/{config_name}/{run_timestamp}/{data_split}-model_best-metrics.json`

This is a json file containing the metrics on the provided data split. These are just common classification metrics including accuracy, recall, precision, and F1 both averaged across classes and within classes. 

`saved/models/{config_name}/{run_timestamp}/{data_split}-model_best-word-pair-errors.json`

This second file contains false_positives, true_positives, and false_negatives word pairs comparing the model's output to the provided list of relations for each relation type.

`saved/models/{config_name}/{run_timestamp}/{data_split}-model_best-word-pair-errors.json`

This third file contains predictions for each input word-pair. For each word pair, it provides a list of relations representing the predicted relation from most confidence to least confident along with the associated confidence scores (softmax probabilities). 

## Predicting

The predict.py script takes in a text file and term file and outputs predicted relations for all term pairs that share at least one sentence in the input text file (supported term input file formats are json in the output format from accompanying term extraction models in the original parent directory of this project or simple text file with one term per line). Here is the outline for the call to the predict.py script. 

`python predict.py 
     -r saved/models/{config_name}/{run_date_name}/{model_name}.pth 
     -t {term_file}.[json|txt]
     -i {input_text_file} 
     -o {output_dir}`


This script outputs a single file:
  - {input_filename}_{config_name}-{run_timestamp}_predicted_relations.json: This is a json file with an entry for every term pair that shares at least one sentence in the input text. Each term pair has an ordered list of relation types/classes and their associated confidences ordered from most confident to least confident (same format as test script output).

Predictions are made for both directions for each term pair. The predict script postprocesses the predicted relations by only keeping the direction for a term pair that has the highest confidence in the case that the same prediction is made for both directions.
  
## Config file format

Config files are in `.json` format:
```javascript
{
  "name": "entity-entity_class",        // training session name (folder name given in saved directory) 
  "n_gpu": 1,                           // number of GPUs to use for training.
  
  "arch": {
    "type": "BagAttentionBert",         // name of model architecture to train
    "args": {
      "dropout_rate": 0.3,   
      "num_classes": 4
    }                
  },
  "data_loader": {
    "type": "RelationDataLoader",         // selecting data loader
    "args":{
      "data_dir": "../../data/term_extraction",             // dataset path
      "batch_size": 4,                   // batch size
      "shuffle": true,                   // shuffle training data before splitting
      "relations": ["taxonomy", "meronym", "spatial"], // relation types or classes to predict 
      "num_workers": 0,                  // number of cpu processes to be used for data loading
      "embedding_type": "Bert",          // Type of embedding to use
      "max_sent_length": 256,            // Maximum number of tokens allowed for a sentence
      "max_sentences": 4                 // Maximum number of sentences allowed in each bag 
    }
  },
  "optimizer": {
    "type": "AdamW",
    "args":{
      "lr": 3e-5,                     // learning rate
      "weight_decay": 0.01            // (optional) weight decay
    }
  },
  "loss": "nll_loss",                  // loss
  "metrics": [
      "accuracy", "micro_f1", "macro_f1", "micro_recall", "macro_recall", "micro_precision", "macro_precision"
  ],
  "lr_scheduler": {
    "type": "StepLR",                  // learning rate scheduler
    "args":{
      "step_size": 50,          
      "gamma": 0.1
    }
  },
  "trainer": {
    "epochs": 10,                     // number of training epochs
    "save_dir": "saved/",             // checkpoints are saved in save_dir/models/name
    "save_freq": 1,                   // save checkpoints every save_freq epochs
    "verbosity": 2,                   // 0: quiet, 1: per epoch, 2: full
  
    "monitor": "max val_macro_f1"     // mode and metric for model performance monitoring. set 'off' to disable.
    "early_stop": 3	                  // number of epochs to wait before early stop. set 0 to disable.
  
    "tensorboard": true,              // enable tensorboard visualization
  }
}
```



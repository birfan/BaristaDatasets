# Barista Datasets and Data-Driven Dialogue Models in Generic and Personalised Task-Oriented Dialogue

The datasets and the evaluations for the adapted data-driven dialogue models are described in the paper "Coffee with a Hint of Data: Towards Using Data-Driven Approaches in Personalised Long-Term Interactions" (under review at the Frontiers in Robotics and AI journal) by Bahar Irfan and Tony Belpaeme.

##  Barista Datasets 

Barista Dataset is designed to model a real-world barista who: (1) greets and requests the drink order, (2) size, and (3) snack, (4) confirms the order, (5) changes the order if necessary, (6) takes the customer's name, (7) notes the order pick up location, (8) says goodbye. Typically, a customer can ask for the order in one sentence, removing the need of (2) and (3), however, we separated these steps to reduce the errors in rule-based (e.g., template matching) or data-driven approaches, and to aid speech recognition (for the robot).

Personalised Barista Dataset recognises customers with provided user recognition information (as if it is a human-robot interaction), and recalls their most common (or most recent) order to suggest to the customers. The dataset also contains failures that can arise in real-world interactions, such as incorrect recognition and recalls, and changes to the preferences, to train methods to overcome these failures.

Personalised Barista with Preferences Information Dataset includes the preferences of the customer, in addition to the customer identity, to simulate a knowledge-base extraction.

For detailed descriptions of the datasets, see the README files in the corresponding folders.

## Data-Driven Dialogue Models

The Barista datasets were evaluated with the state-of-the-art data-driven dialogue models: Supervised Embeddings (Dodge et al., 2015; Bordes et al., 2016), Sequence-to-Sequence (Sutskever et al., 2015), End-to-End Memory Networks (Sukhbaatar et al., 2016), Generative Profile Memory Networks (Zhang et al., 2018), Key-Value Memory Networks (Miller et al., 2016; Zhang et al., 2018) and Split Memory Networks (Joshi et al., 2017). We adapted the code from ParlAI (https://github.com/facebookresearch/ParlAI) for Sequence-to-Sequence, Generative Profile Memory Networks and Key-Value Memory Networks models and the code from Joshi et al., 2017 (https://github.com/chaitjo/personalized-dialog) for End-to-End Memory Networks, Split Memory Networks and Supervised Embeddings. The adapted code is provided here for reproducibility of the evaluations on the Barista Datasets.

Clone the repository.

    $ git clone https://github.com/birfan/BaristaDatasets.git ~/BaristaDatasets
    $ cd ~/BaristaDatasets
    $ mkdir results

Use Docker to build the container for evaluations. Note that NVIDIA CUDA 10.0 is used in the Docker container, you may need to adapt to your version.

    $ docker build -t barista_experiments .

Look up the corresponding docker image id.

    $ docker image ls

Run the contain interactively and mount the /project/ directory to the results folder to obtain the trained models and the test results:

    $ docker container run -v ~/BaristaDatasets/results:/project/ -it [docker_image_id] /bin/bash

### Variables

dataset_name: 

* "barista" for Barista Dataset
* "barista-personalised" for Personalised Barista Dataset
* "barista-personalised-order" for Personalised Barista with Preferences Information Dataset

task_name:

* "Task1k" for 1000 dialogues dataset
* "Task10k" for 10000 dialogues dataset
* "SecondInteraction" (only in Personalised Barista Datasets) for Second Interaction dataset

hops: in End-to-End Memory Networks, Split Memory Networks, Key-Value Memory Networks

* 1
* 2
* 3

Within the bash shell, run the following commands to train and test the baselines. All tasks for the dataset will be trained/tested (i.e., 1 to 7 for Barista Dataset, and 0 to 8 for Personalised Barista Datasets).

### End-to-End Memory Networks

Train:

    $ cd /app/hrinlp/baselines/MemN2N/;bash bin/train_all.sh [dataset_name] [task_name] [hops]

Test:

    $ cd /app/hrinlp/baselines/MemN2N/;bash bin/test_all.sh [dataset_name] [task_name] [hops]

### Split Memory Networks

Train:

    $ cd /app/hrinlp/baselines/MemN2N-split-memory/;bash bin/train_all.sh [dataset_name] [task_name] [hops]

Test:

    $ cd /app/hrinlp/baselines/MemN2N-split-memory/;bash bin/test_all.sh [dataset_name] [task_name] [hops]


### Supervised Embeddings

Train:

    $ cd /app/hrinlp/baselines/supervised-embedding/;bash bin/train_all.sh [dataset_name] [task_name]

Test:

    $ cd /app/hrinlp/baselines/supervised-embedding/;bash bin/test_all.sh [dataset_name] [task_name]

### Sequence-to-Sequence

Train:

    $ cd /app/ParlAI/;bash parlai_internal/agents/seq2seq/train_all.sh [dataset_name] [task_name]

Test: (Used 1 for batch size and number of threads in the paper)

    $ cd /app/ParlAI/;bash parlai_internal/agents/seq2seq/test_all.sh [dataset_name] [task_name] [batchsize] [numthreads]

### Generative Profile Memory Networks

Train:

    $ cd /app/ParlAI/;bash parlai_internal/agents/profilememory/train_all.sh [dataset_name] [task_name]

Test: (Used 1 for batch size and number of threads in the paper)

    $ cd /app/ParlAI/;bash parlai_internal/agents/profilememory/test_all.sh [dataset_name] [task_name] [batchsize] [numthreads]

### Key-Value Memory Networks:

Train: (18 threads were used in the experiments)

    $ cd /app/ParlAI/;bash parlai_internal/agents/kvmemnn/train_all.sh [dataset_name] [task_name] [hops] [numthreads]

Test: (18 threads were used in the experiments)

    $ cd /app/ParlAI/;bash parlai_internal/agents/kvmemnn/test_all.sh [dataset_name] [task_name] [hops] [numthreads]

More options are available for the models in train.py and eval.py scripts of ParlAI codes; single_dialog.py in Memory Networks or Split Memory; train.py and test.py scripts in Supervised Embeddings.

Models with other hyperparameters (as described in the Irfan and Belpaeme paper) are available under the "baselines" and "agents" folders. The above mentioned methods have the best performing hyperparameters as described in the paper.

## License

The Barista datasets are released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. A copy of this license is included with the data. The adapted codes are released with original licenses under the corresponding folders.

## Contact

For more details on the dataset and baselines, see the paper "Coffee with a Hint of Data: Towards Using Data-Driven Approaches in Personalised Long-Term Interactions" by Bahar Irfan and Tony Belpaeme, under review at the Frontiers in Robotics and AI journal. For any information, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.


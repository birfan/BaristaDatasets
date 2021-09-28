# Barista Datasets and Data-Driven Dialogue Models in Generic and Personalised Task-Oriented Dialogue

The datasets and the evaluations for the adapted data-driven dialogue models are described in detail in the papers:

 * Bahar Irfan, Mehdi Hellou, Alexandre Mazel, Tony Belpaeme (2020), "Challenges of a Real-World HRI Study with Non-Native English Speakers: Can Personalisation Save the Day?", Companion of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI), [DOI: 10.1145/3371382.3378278](https://dl.acm.org/doi/10.1145/3371382.3378278).

 * Bahar Irfan, Mehdi Hellou, Tony Belpaeme (2021), "Coffee with a Hint of Data: Towards Using Data-Driven Approaches in Personalised Long-Term Interactions", Frontiers in Robotics and AI, [DOI: 10.3389/frobt.2021.676814](https://www.frontiersin.org/article/10.3389/frobt.2021.676814).

Please cite both papers if you are using the datasets; cite both papers for generic and personalised barista robots based on the datasets; cite the second paper if you are referring to the data-driven dialogue evaluations on the datasets.

##  Barista Datasets

For obtaining only the datasets (excluding the data-driven models), download the *barista-datasets* release.

Barista Dataset (in *barista* folder under *data*) is designed to model a real-world barista who: (1) greets and requests the drink order, (2) size, and (3) snack, (4) confirms the order, (5) changes the order if necessary, (6) takes the customer's name, (7) notes the order pick up location, (8) says goodbye. Typically, a customer can ask for the order in one sentence, removing the need of (2) and (3), however, we separated these steps to reduce the errors in rule-based (e.g., template matching) or data-driven approaches, and to aid speech recognition (for the robot).

Personalised Barista Dataset (*barista-personalised*) recognises customers with provided user recognition information (as if it is a human-robot interaction), and recalls their most common (or most recent) order to suggest to the customers. The dataset also contains failures that can arise in real-world interactions, such as incorrect recognition and recalls, and changes to the preferences, to train methods to overcome these failures.

Personalised Barista with Preferences Information Dataset (*barista-personalised-order*) includes the preferences of the customer, in addition to the customer identity, to simulate a knowledge-base extraction.

*all_labels* folder contains datasets with all possible correst responses (each response is separated with "|") for each customer phrase, exactly corresponding (in terms of customer orders, phrases and names in order) to the Barista Datasets. 

*ParlAI_format* folder contains the Barista Datasets in ParlAI (Miller et al., 2017) [format](https://parl.ai/docs/tutorial_task.html#quickstart-adding-a-new-dataset). The original Barista Datasets are in FbDialogTeacher format (the previous format of datasets in ParlAI).

*info* folder contains the proportion of phrases containing customer name or preferences (*personal(ised)*), phrases containing order item (*order*), remaining (*other*) phrases, and phrases from the Barista Dataset (*Only Barista* (B7)) within all the utterances (*Utterance count*) of the barista (bot) for each task and dataset.

*templates* folder contains the template phrases used in generating the datasets. The phrases in *barista_templates.csv* are used in both the Barista and Personalised Barista Datasets, whereas *personalised_barista_templates.csv* contains phrases specific to the Personalised Barista Datasets. The templates for order items, preferences or customer name are written in brackets in capitals, e.g., *[RECALLED_PREF_DRINK]* refers to the recalled most preferred drink of the customer, *[RECOG_CUS_NAME] [RECOG_CUS_SURNAME]* refers to the first name and surname of the recognised customer, *PREV_CHOSEN* and *NEW_CHOSEN* refer to the previous and new chosen order item, respectively. The first column denotes whether the response is from the customer or bot, and the second column denotes the type of the phrase (e.g. *Greeting*).

For detailed descriptions of the datasets, see the README files in the corresponding folders.

## Data-Driven Dialogue Models

The Barista datasets were evaluated with the state-of-the-art data-driven dialogue models: Supervised Embeddings (Dodge et al., 2015; Bordes et al., 2016), Sequence-to-Sequence (Sutskever et al., 2015), End-to-End Memory Networks (Sukhbaatar et al., 2016; Bordes et al., 2016), Generative Profile Memory Networks (Zhang et al., 2018), Key-Value Memory Networks (Miller et al., 2016; Zhang et al., 2018) and Split Memory Networks (Joshi et al., 2017). We adapted the implementations from [ParlAI](https://github.com/facebookresearch/ParlAI) for Sequence-to-Sequence, Generative Profile Memory Networks and Key-Value Memory Networks models and the implementation from [Joshi et al., 2017](https://github.com/chaitjo/personalized-dialog) for End-to-End Memory Networks, Split Memory Networks and Supervised Embeddings. The adapted code is provided here (in *parlai_internal* for former methods, in *baselines* for latter methods) for reproducibility of the evaluations on the Barista Datasets.

Clone the repository (or download the *barista-datasets-and-models* release).

    $ git clone https://github.com/birfan/BaristaDatasets.git ~/BaristaDatasets
    $ cd ~/BaristaDatasets
    $ mkdir results

Use Docker to build the container for evaluations. Note that NVIDIA CUDA 10.0 is used in the Docker container, you may need to adapt to your version.

    $ docker build -t barista_experiments .

Look up the corresponding docker image id.

    $ docker image ls

Run the contain interactively and mount the /project/ directory to the results folder to obtain the trained models and the test results:

    $ docker container run -v ~/BaristaDatasets/results:/project/ -it [docker_image_id] /bin/bash

If you are not using Docker, download [ParlAI repository](https://github.com/facebookresearch/ParlAI.git). Copy *data* and *parlai_internal* folders in *BaristaDatasets* under *ParlAI* main folder, and replace *params.py* file in *ParlAI/parlai/core* with the file in *parlai_internal/scripts*. For all other dependencies (and how to download them), see *Dockerfile*.

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

Below code is for training and evaluating models in Docker container, adapt accordingly if running in local.

### End-to-End Memory Networks: Best performing model for Personalised Task-Oriented Dialogue

Adapted implementation from: https://github.com/chaitjo/personalized-dialog

Train:

    $ cd /app/hrinlp/baselines/MemN2N/;bash bin/train_all.sh [dataset_name] [task_name] [hops]

Test:

    $ cd /app/hrinlp/baselines/MemN2N/;bash bin/test_all.sh [dataset_name] [task_name] [hops]

### Split Memory Networks

Adapted implementation from: https://github.com/chaitjo/personalized-dialog

Train:

    $ cd /app/hrinlp/baselines/MemN2N-split-memory/;bash bin/train_all.sh [dataset_name] [task_name] [hops]

Test:

    $ cd /app/hrinlp/baselines/MemN2N-split-memory/;bash bin/test_all.sh [dataset_name] [task_name] [hops]


### Supervised Embeddings

Adapted implementation from: https://github.com/chaitjo/personalized-dialog

Train:

    $ cd /app/hrinlp/baselines/supervised-embedding/;bash bin/train_all.sh [dataset_name] [task_name]

Test:

    $ cd /app/hrinlp/baselines/supervised-embedding/;bash bin/test_all.sh [dataset_name] [task_name]

### Sequence-to-Sequence: Best performing model for Generic Task-Oriented Dialogue

Adapted implementation from: https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2/baselines/seq2seq

Train:

    $ cd /app/ParlAI/;bash parlai_internal/agents/seq2seq/train_all.sh [dataset_name] [task_name]

Test: (Used 1 for batch size and number of threads in the paper)

    $ cd /app/ParlAI/;bash parlai_internal/agents/seq2seq/test_all.sh [dataset_name] [task_name] [batchsize] [numthreads]

### Generative Profile Memory Networks

Adapted implementation from: https://github.com/facebookresearch/ParlAI/tree/6a76a555ea84b06e2914cdea4c56a46a5f495821/projects/personachat

Train:

    $ cd /app/ParlAI/;bash parlai_internal/agents/profilememory/train_all.sh [dataset_name] [task_name]

Test: (Used 1 for batch size and number of threads in the paper)

    $ cd /app/ParlAI/;bash parlai_internal/agents/profilememory/test_all.sh [dataset_name] [task_name] [batchsize] [numthreads]

### Key-Value Memory Networks

Adapted implementation from: https://github.com/facebookresearch/ParlAI/tree/master/projects/convai2/baselines/kvmemnn

Train: (18 threads were used in the experiments)

    $ cd /app/ParlAI/;bash parlai_internal/agents/kvmemnn/train_all.sh [dataset_name] [task_name] [hops] [numthreads]

Test: (18 threads were used in the experiments)

    $ cd /app/ParlAI/;bash parlai_internal/agents/kvmemnn/test_all.sh [dataset_name] [task_name] [hops] [numthreads]

More options are available for the models in train.py and eval.py scripts of ParlAI codes; single_dialog.py in Memory Networks or Split Memory; train.py and test.py scripts in Supervised Embeddings.

Models with other hyperparameters (as described in the Irfan et al., 2021) are available under the *baselines* and *agents* folders. The above mentioned methods have the best performing hyperparameters as described in the paper.

## Trained Models

*barista-datasets-and-trained-models* release contains the Barista Datasets, adapted code for models and the best trained models corresponding to the reported results (in Irfan et al., 2021) for each baseline on the Barista Datasets. Profile Memory and Seq2Seq models have only the last task of the datasets (i.e., task 7 in Barista or task 8 in Personalised Barista Datasets), because the trained models are very large in size. Key-Value Memory Network was only trained with one hop, due to the vast amount of time required for training. *performance_results* contains the per-response accuracy of each trained model on the Barista Datasets.

## License

The Barista datasets are released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. In other words, these datasets are made available for academic research purpose only. A copy of this license is included with the data. The adapted codes are released with original licenses under the corresponding folders.

## Contact

For any information or for requesting Barista Datasets with different order items, customer names, or larger dataset size, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk (the most recent contact information is available at [personal website](https://www.baharirfan.com)).

## References

 * Alexander H. Miller, Will Feng, Adam Fisch, Jiasen Lu, Dhruv Batra, Antoine Bordes, Devi Parikh, Jason Weston (2017), "ParlAI: A Dialog Research Software Platform", [arXiv:1705.06476](https://arxiv.org/abs/1705.06476)
 * Jesse Dodge, Andreea Gane, Xiang Zhang, Antoine Bordes, Sumit Chopra, Alexander Miller, Arthur Szlam, Jason Weston (2015), "Evaluating prerequisite qualities for learning end-to-end dialog systems", [arXiv:1511.06931](https://arxiv.org/abs/1511.06931) 
 * Antoine Bordes, Y-Lan Boureau, Jason Weston (2016), "Learning End-to-End Goal-Oriented Dialog", [arXiv:1605.07683](https://arxiv.org/abs/1605.07683)
 * Ilya Sutskever, Oriol Vinyals, Quoc V. Le (2015), "Sequence to Sequence Learning with Neural Networks", [arXiv:1409.3215](https://arxiv.org/abs/1409.3215)
 * Sainbayar Sukhbaatar, Arthur Szlam, Jason Weston, Rob Fergus (2015), "End-To-End Memory Networks", [arXiv:1503.08895](https://arxiv.org/abs/1503.08895)
 * Saizheng Zhang, Emily Dinan, Jack Urbanek, Arthur Szlam, Douwe Kiela, Jason Weston (2018), "Personalizing Dialogue Agents: I have a dog, do you have pets too?", [arXiv:1801.07243](https://arxiv.org/abs/1801.07243)
 * Alexander Miller, Adam Fisch, Jesse Dodge, Amir-Hossein Karimi, Antoine Bordes, Jason Weston (2016), "Key-Value Memory Networks for Directly Reading Documents", [arXiv:1606.03126](https://arxiv.org/abs/1606.03126)
 * Chaitanya K. Joshi, Fei Mi, Boi Faltings (2017), "Personalization in Goal-Oriented Dialog", [arXiv:1706.07503](https://arxiv.org/abs/1706.07503)



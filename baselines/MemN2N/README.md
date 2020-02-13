# End-to-end Memory Networks for Dialog
Implementation of Memory Networks described in [Learning End-to-End Goal-Oriented Dialog](https://arxiv.org/abs/1605.07683) in Tensorflow. Tested on the bAbI dialog dataset and the personalized dialog dataset. 

Experimental logs and best performing models for each task can be downloaded using the `build_experiments.py` script.

Adapted from [vyraun's implementation](https://github.com/vyraun/chatbot-MemN2N-tensorflow).

## Usage

Train the model
```
python single_dialog.py --train True --task_id 1
```

Test the trained model
```
python single_dialog.py --train False --task_id 1
```

## Requirements

* tensorflow 0.12.1
* scikit-learn
* six
* scipy

## Results

### bAbI Dialog tasks

Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy	 |  Testing Accuracy(OOV)
------|---------------------|-----------------------|--------------------|-----------------------
1     |  99.9	            |  99.1		            |  99.3				 |	76.3
2     |  100                |  100		            |  99.9				 |	78.9
3     |  96.1               |  71.0		            |  71.1				 |	64.8
4     |  99.9               |  56.7		            |  57.2				 |	57.0
5     |  99.9               |  98.4		            |  98.5				 |	64.9
6     |  73.1               |  49.3		            |  40.6				 |	---

### Personalized Dialog tasks (full set)

Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy	 |  Testing Accuracy(OOV)
------|---------------------|-----------------------|--------------------|-----------------------
1	  |  99.75				|  99.69				|  99.83  			 |  75.57
2	  |  100				|  99.99				|  99.99			 |  78.85
3     |  87.52				|  57.17				|  58.94			 |  ---
4	  |  100				|  56.67				|  57.17			 |  ---
5	  |  94.38				|  81.11				|  82.60 (85.33)	 |  ---

### Personalized Dialog tasks (small set)

Task  |  Training Accuracy  |  Validation Accuracy  |  Testing Accuracy	 |  Testing Accuracy(OOV)
------|---------------------|-----------------------|--------------------|-----------------------
1	  |  99.43				|  97.75				|  98.87  			 |  71.24
2	  |  100				|  99.91				|  99.93			 |  78.85
3     |  96.54				|  57.97				|  58.71			 |  ---
4	  |  100				|  56.67				|  57.17			 |  ---
5	  |  99.58				|  77.59				|  77.74			 |  ---

### Multi-task learning experiments

Profile				|  Training Accuracy  |	 Validation Accuracy  |	 Testing Accuracy  |  Testing Accuracy (multi-profile model) [trained on small set]
--------------------|---------------------|-----------------------|--------------------|--------------------------------
male, young			|		99.66		  |			79.66		  |		80.38	  	   |		85.04 [77.70]	
female, young		|		99.51		  |			79.96		  |		80.15		   |		84.91 [77.14]
male, middle-aged	|		99.68		  |			80.07		  |		80.29		   |		84.71 [77.59]		
female, middle-aged	|		99.60		  |			79.88		  |		80.21	  	   |		85.14 [77.80]	
male, elderly		|		99.61		  |			80.04		  |		80.57	  	   |		85.46 [77.82]	
female, elderly		|		99.40		  |			79.57		  |		80.41	  	   |		84.95 [77.52]	

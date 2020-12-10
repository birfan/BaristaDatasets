# Barista Dataset

This directory contains a set of 7 tasks for evaluating end-to-end dialogue systems for goal-oriented interactions in ordering a drink and a snack at a coffee shop from the barista.

The datasets and the evaluations for the adapted data-driven dialogue models are described in detail in the papers:

    Bahar Irfan, Mehdi Hellou, Alexandre Mazel, and Tony Belpaeme (2020), "Challenges of a Real-World HRI Study with Non-Native English Speakers: Can Personalisation Save the Day?", Companion of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI), DOI: 10.1145/3371382.3378278.

    Bahar Irfan and Tony Belpaeme (under review), "Coffee with a Hint of Data: Towards Using Data-Driven Approaches in Personalised Long-Term Interactions", Frontiers in Robotics and AI.

Please cite both papers if you are using the datasets; cite the first paper for generic and personalised barista robots based on the datasets; cite the second paper if you are referring to the data-driven dialogue evaluations on the datasets.

## Data

The format of the dataset is in ParlAI format (https://parl.ai/docs/tutorial_task.html#quickstart-adding-a-new-dataset). Tasks are under folders *train* (training set), *test* (test set), *dev* (development set), *test-OOV* (out of vocabulary set - includes names and drink, size and snack choices which are not present in training, test or development sets). *Task1k* contains 1000 dialogues and *Task10k* contains 10000 dialogues.

The data is structured as follows in training set:

```
text:user_utterance [tab] labels:bot_utterance
...
text:user_utterance [tab] labels:bot_utterance [tab] episode_done:True

```

The goal of the tasks is to predict the bot utterances, which are sentences.

In test, development and OOV sets, *eval_labels* is used instead of *labels* to prevent the model from training on the labels. Along with the train, dev and test sets, we also include a file containing the candidates to select the answer from (barista-candidates.txt) for tasks 1-7, that is simply made of all the unique bot utterances in the train, dev, test for these tasks.

## Tasks

### Task 1: Greetings

This task evaluates greeting the customer. No order is made.

**Example 1.1:**
```
text:Hi!	labels:Hello! What can I get for you?
text:That is correct.	labels:Perfect! What would be the name?
text:It is Kaley Cuoco.	labels:We will get that ready for you at the next counter, Kaley.
text:Cheers!	labels:Enjoy your day!  episode_done:True
```

### Task 2: Order drink without greetings

This task evaluates ordering a drink from the barista. No extras are ordered. Greetings are not included.

**Example 2.1:**
```
text:I would like a cappuccino, please. labels:Which size for the cup?
text:Small, please. labels:Can I offer you anything to go with your drink?
text:That would be it.	labels:Your order is a small cappuccino. Is that right?
text:Definitely.    labels:Your order will be ready at the next counter.    episode_done:True
```

### Task 3: Order drink with changes without greetings 

This task evaluates ordering a drink and changing the order (0-2 changes) during the interaction. No extras are ordered. Greetings are not included.

**Example 3.1:**
```
text:Could I have an Earl Grey, please?	labels:Which size for the cup?
text:Small.	labels:Would you like a delicious snack with that?
text:No, thank you.	labels:Your order is a small Earl Grey. Did I get that right?
text:On second thought, I would have a mocha instead.	labels:Of course! So that would be a small mocha. Did I get that right?
text:Yeap.	labels:Your order will be ready at the next counter.    episode_done:True
```

### Task 4: Order drink and extras without greetings

This task evaluates ordering a drink and an extra (snack). Greetings are not included. The order is not changed.

**Example 4.1:**
```
text:Could I have a salted caramel hot chocolate, please?	labels:Which size?
text:Large would do.	labels:Anything else?
text:Ah yes, a gingerbread cookie, please.	labels:So that would be a large salted caramel hot chocolate and a gingerbread cookie. Is that right?
text:Definitely.	labels:We will get that ready for you at the next counter.  episode_done:True
```

### Task 5: Order drink and extras with changes without greetings

This task evaluates ordering a drink and an extra and changing the order (0-2 changes) during the interaction. No extras are ordered. Greetings are not included.

**Example 5.1:**
```
text:A salted caramel hot chocolate, please.	labels:Which size?
text:Large, thank you!	labels:Anything else?
text:Ah yes, a double chocolate muffin, please.	labels:So that would be a large salted caramel hot chocolate and a double chocolate muffin. Is that right?
text:I think I will switch salted caramel hot chocolate with a peppermint hot chocolate.	labels:Of course! Your order is a large peppermint hot chocolate and a double chocolate muffin. Is that right?
text:Erm, I would like small actually.	labels:Absolutely! Your order is a small peppermint hot chocolate and a double chocolate muffin. Did I get that right?
text:That is correct.	labels:Your order will be ready at the next counter.    episode_done:True
```

### Task 6: Order drink and extras with greetings

This task is the combination of tasks 1 and 4.

**Example 6.1:**
```
text:Hey!	labels:Welcome to the coffee heaven! What can I get for you today?
text:A caramel macchiato, please.	labels:Which size would you like?
text:Large cup.	labels:Would you like anything else with that?
text:Could I get a blueberry oatmeal, please?	labels:So that would be a large caramel macchiato and a blueberry oatmeal. Did I get that right?
text:Yes!	labels:Great, and what is your name?
text:It is Jesse Tyler Ferguson.	labels:Your order will be ready at the next counter, Jesse.
text:Thank you! labels:Have a great day!    episode_done:True
```

### Task 7: Order drink and extras with changes and greetings

This task is the combination of tasks 1 and 5.

**Example 7.1:**
```
text:Hey!	labels:Welcome to Botista! What would you like to have?
text:An espresso, please.	labels:Which size for the cup?
text:Large would do.	labels:Would you like anything else with that?
text:I would like a plain bagel, please.	labels:So that would be a large espresso and a plain bagel. Is that right?
text:Erm, I would like small actually.	labels:Sure! Your order is a small espresso and a plain bagel. Is that right?
text:Sorry, I would like to change that to a croissant.	labels:Sure! Your order is a small espresso and a croissant. Is that right?
text:That sounds right!	labels:May I take your name, please?
text:Matt Bomer.	labels:You can pick up your order at the next counter, Matt.
text:Thanks!	labels:Have a great day!    episode_done:True
```

## License

The Barista datasets are released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. In other words, these datasets are made available for academic research purpose only. A copy of this license is included with the data.

## Contact

For any information or for requesting Barista Datasets with different order items, customer names, or larger dataset size, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.

# Barista Dataset

This directory contains a set of 7 tasks for evaluating end-to-end dialogue systems for goal-oriented interactions in ordering a drink and a snack at a coffee shop from the barista.

The datasets and the evaluations for the adapted data-driven dialogue models are described in detail in the papers:

    Bahar Irfan, Mehdi Hellou, Alexandre Mazel, and Tony Belpaeme (2020), "Challenges of a Real-World HRI Study with Non-Native English Speakers: Can Personalisation Save the Day?", Companion of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI), DOI: 10.1145/3371382.3378278.

    Bahar Irfan and Tony Belpaeme (under review), "Coffee with a Hint of Data: Towards Using Data-Driven Approaches in Personalised Long-Term Interactions", Frontiers in Robotics and AI.

Please cite both papers if you are using the datasets; cite the first paper for generic and personalised barista robots based on the datasets; cite the second paper if you are referring to the data-driven dialogue evaluations on the datasets.

## Data

The format of the dataset is in *FB* in FbDialogTeacher format for ParlAI. Tasks are under folders *train* (training set), *test* (test set), *dev* (development set), *test-OOV* (out of vocabulary set - includes names and drink, size and snack choices which are not present in training, test or development sets). *Task1k* contains 1000 dialogues and *Task10k* contains 10000 dialogues.

The data is structured as follows:

```
ID user_utterance [tab] bot_utterance
...
```

The *ID* for a given dialogue start at 1 and increase within each dialogue. The goal of the tasks is to predict the bot utterances, which are sentences.

Along with the train, dev and test sets, we also include a file containing the candidates to select the answer from (*barista-candidates.txt*) for tasks 1-7, that is simply made of all the unique bot utterances in the train, dev, test for these tasks.

## Tasks

### Task 1: Greetings

This task evaluates greeting the customer. No order is made.

**Example 1.1:**
```
1 Hi!	Hello! What can I get for you?
2 That is correct.	Perfect! What would be the name?
3 It is Kaley Cuoco.	We will get that ready for you at the next counter, Kaley.
4 Cheers!	Enjoy your day!
```

### Task 2: Order drink without greetings

This task evaluates ordering a drink from the barista. No extras are ordered. Greetings are not included.

**Example 2.1:**
```
1 I would like a cappuccino, please.	Which size for the cup?
2 Small, please.	Can I offer you anything to go with your drink?
3 That would be it.	Your order is a small cappuccino. Is that right?
4 Definitely.	Your order will be ready at the next counter.
```

### Task 3: Order drink with changes without greetings 

This task evaluates ordering a drink and changing the order (0-2 changes) during the interaction. No extras are ordered. Greetings are not included.

**Example 3.1:**
```
1 Could I have an Earl Grey, please?	Which size for the cup?
2 Small.	Would you like a delicious snack with that?
3 No, thank you.	Your order is a small Earl Grey. Did I get that right?
4 On second thought, I would have a mocha instead.	Of course! So that would be a small mocha. Did I get that right?
5 Yeap.	Your order will be ready at the next counter.
```

### Task 4: Order drink and extras without greetings

This task evaluates ordering a drink and an extra (snack). Greetings are not included. The order is not changed.

**Example 4.1:**
```
1 Could I have a salted caramel hot chocolate, please?	Which size?
2 Large would do.	Anything else?
3 Ah yes, a gingerbread cookie, please.	So that would be a large salted caramel hot chocolate and a gingerbread cookie. Is that right?
4 Definitely.	We will get that ready for you at the next counter.
```

### Task 5: Order drink and extras with changes without greetings

This task evaluates ordering a drink and an extra and changing the order (0-2 changes) during the interaction. No extras are ordered. Greetings are not included.

**Example 5.1:**
```
1 A salted caramel hot chocolate, please.	Which size?
2 Large, thank you!	Anything else?
3 Ah yes, a double chocolate muffin, please.	So that would be a large salted caramel hot chocolate and a double chocolate muffin. Is that right?
4 I think I will switch salted caramel hot chocolate with a peppermint hot chocolate.	Of course! Your order is a large peppermint hot chocolate and a double chocolate muffin. Is that right?
5 Erm, I would like small actually.	Absolutely! Your order is a small peppermint hot chocolate and a double chocolate muffin. Did I get that right?
6 That is correct.	Your order will be ready at the next counter.
```

### Task 6: Order drink and extras with greetings

This task is the combination of tasks 1 and 4.

**Example 6.1:**
```
1 Hey!	Welcome to the coffee heaven! What can I get for you today?
2 A caramel macchiato, please.	Which size would you like?
3 Large cup.	Would you like anything else with that?
4 Could I get a blueberry oatmeal, please?	So that would be a large caramel macchiato and a blueberry oatmeal. Did I get that right?
5 Yes!	Great, and what is your name?
6 It is Jesse Tyler Ferguson.	Your order will be ready at the next counter, Jesse.
7 Thank you!	Have a great day!
```

### Task 7: Order drink and extras with changes and greetings

This task is the combination of tasks 1 and 5.

**Example 7.1:**
```
1 Hey!	Welcome to Botista! What would you like to have?
2 An espresso, please.	Which size for the cup?
3 Large would do.	Would you like anything else with that?
4 I would like a plain bagel, please.	So that would be a large espresso and a plain bagel. Is that right?
5 Erm, I would like small actually.	Sure! Your order is a small espresso and a plain bagel. Is that right?
6 Sorry, I would like to change that to a croissant.	Sure! Your order is a small espresso and a croissant. Is that right?
7 That sounds right!	May I take your name, please?
8 Matt Bomer.	You can pick up your order at the next counter, Matt.
9 Thanks!	Have a great day!
```

## License

The Barista datasets are released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. In other words, these datasets are made available for academic research purpose only. A copy of this license is included with the data.

## Contact

For any information or for requesting Barista Datasets with different order items, customer names, or larger dataset size, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.

# Personalised Barista Dataset

This directory contains a set of 9 tasks for evaluating end-to-end dialogue systems for personalisation in goal-oriented long-term interactions, which is an extension of the non-personalised barista-dataset (of task 7 - ordering drink and extras with changes and greetings). The scenario is based on ordering a drink and a snack (extra) at a coffee shop from the barista. The barista recognises the customers and recalls the previous orders. 

The datasets and the evaluations for the adapted data-driven dialogue models are described in detail in the papers:

    Bahar Irfan, Mehdi Hellou, Alexandre Mazel, and Tony Belpaeme (2020), "Challenges of a Real-World HRI Study with Non-Native English Speakers: Can Personalisation Save the Day?", Companion of the 2020 ACM/IEEE International Conference on Human-Robot Interaction (HRI), DOI: 10.1145/3371382.3378278.

    Bahar Irfan and Tony Belpaeme (under review), "Coffee with a Hint of Data: Towards Using Data-Driven Approaches in Personalised Long-Term Interactions", Frontiers in Robotics and AI.

Please cite both papers if you are using the datasets; cite the first paper for generic and personalised barista robots based on the datasets; cite the second paper if you are referring to the data-driven dialogue evaluations on the datasets.


## Data

The format of the dataset is in ParlAI format (https://parl.ai/docs/tutorial_task.html#quickstart-adding-a-new-dataset). Tasks are under folders *train* (training set), *test* (test set), *dev* (development set), *test-OOV* (out of vocabulary set - includes names and drink, size and snack choices which are not present in training, test or development sets). *SecondInteraction* folder contains dialogues that focus on first, second and third interaction scenarios. *Task1k* contains 1000 dialogues and *Task10k* contains 10000 dialogues.

The data is structured as follows:

```
recognised_isCusKnown , recognised_cusID , recognised_cusName 
text:user_utterance [tab] labels:bot_utterance
...
text:It is David Boreanaz.	labels:You can pick up your order at the next counter, David.
isCusKnown , cusID , cusName
text:Thanks!	labels:I wish you an amazing day!   episode_done:True
```

In test, development and OOV sets, *eval_labels* is used instead of *labels* to prevent the model from training on the labels.

*recognised\_isCusKnown* is True if the customer is known, False otherwise. *recognised\_cusID* refers to the recognised identity number of the customer, and recognised\_cusName is the corresponding customer name. Each customer is initially unknown, denoted by the *cusID* 0. When the customer is first met, a new *cusID* is given to the customer, which corresponds to the number of customers known by the bot (e.g. *cusID* 1 for the first customer, 2 for the second customer, etc.). Both of this information can be incorrect when there are recognition errors (Tasks 2, 5, 6, 8). If the customer name is requested in the dialogue, that is, if the customer is not recognised or not recognised correctly, then the recognition module records the correct values of the *isCusKnown* and *cusID*, along with the *cusName*. If the customer is not previously known, then a new *cusID* is assigned at this stage. If the user is recognised as unknown at the beginning, then a non-personalised dialogue is used for all the tasks. The most common drink/ extra or the most recent drink/ extra (in case of ties for the most common order) of the customer is remembered.

"<SILENCE>" token is used to fill in for the missing user utterance if the bot speaks two turns in a row during reconsulting of the recognition module in the case of recognition errors.

For example (for task 2):
```
True , 88 , Chris Colfer
text:Hello!    labels:Welcome back Chris Colfer! Would you like to have a large espresso macchiato again?
text:Sorry, I am not Chris Colfer.    labels:I am terribly sorry! What was the name?
text:Patrick Dempsey.   labels: Sorry about that Patrick. Let me see if I have any previous orders from you.
True , 64 , Patrick Dempsey
text:<SILENCE>    labels:Ah, of course, Patrick! Can I get you a small caramel macchiato and a chocolate chip muffin again?
text:Yes, thank you!    labels:We will get that ready for you at the next counter, Patrick.
text:Cheers!    labels:Enjoy your day!  episode_done:True
```

The goal of the tasks is to predict the bot utterances, which are sentences, and recall the previous orders of the customer to personalise the dialogue.

Along with the train, dev and test sets, we also include a file containing the candidates to select the answer from (*barista-personalised-candidates.txt*) for tasks 0-8, that is simply made of all the unique bot utterances in the train, dev, test for these tasks.

## Tasks

For each task, customer orders file (*customer-orders.csv*) is provided to keep track of orders in the interaction which contains interaction number (*IN*), *cusID*, *cusName*, *drink*, *size* (of drink) and *extra* (can be empty). Customer preferences file (*customer-preferences.txt*) show the orders of each customer in increasing *cusID* in a list of order list [drink, size, extra]. In all the tasks, the customer orders and the customer list from training (*train*) is remembered in *test*, *dev* and *OOV* sets. OOV contains new choices for orders, whereas, test and development sets have the same set of order choices as training.


### Task 0: Unknown customers

This task evaluates learning unknown customers and personalising the dialogue in the second interaction. This task assumes perfect recognition, recall and no changes are made by the customers to their previous preference.

For task 0: In the training set, 100 customers from database A have 2 interactions each with the bot. In test and development sets, 100 customers from database B have 2 interactions each. In OOV set, 100 customers from database C have 2 interactions each.

Total number of dialogues in *SecondInteraction* for task 0: 
*train*: 200 = 100 (number of customers in database A) * 2 (number of interactions per customer in database A)
*test*: 200 = 100 (number of customers in database B) * 2 (number of interactions per customer in database B)
*dev*: 200 (same as test)
*OOV*: 200  = 100 (number of customers in database C) * 2 (number of interactions per customer in database C)

**Example 0.1:** First interaction 
```
False , 0 
text:Hi!	labels:Welcome to the coffee heaven! What can I get for you?
text:I would like a flat white, please.	labels:Which size would you like?
text:Small, please.	labels:Can I offer you anything to go with your drink?
text:Ah yes, a pain au chocolat, please.	labels:So that would be a small flat white and a pain au chocolat. Is that right?
text:Sorry, I would like to change that to a cordusio.	labels:Sure! So that would be a small cordusio and a pain au chocolat. Is that right?
text:Yes!	labels:Perfect! What would be the name?
text:Zoe Saldana.	labels:Your order will be ready at the next counter, Zoe.
False , 6 , Zoe Saldana
text:Thanks!	labels:Have a nice day! Hope to see you again at Botista!   episode_done:True
```

**Example 0.2:** Second interaction
```
True , 6 , Zoe Saldana
text:Hey!	labels:Nice to see you, Zoe Saldana! Would you like to have a small cordusio and a pain au chocolat again?
text:Absolutely!	labels:Your order will be ready at the next counter, Zoe.
text:Thank you!	labels:Enjoy your day!  episode_done:True
```

### Task 1: Unknown + known customers

This task evaluates learning unknown customers and remembering the most common preference of the customer to personalise the dialogue. This task assumes perfect recognition, recall and no changes are made by the customers to their previous preference.

For tasks 1-8: In the training set, 100 customers from database A have 3 interactions each with the bot. The first two interactions evaluate learning the new customer from two different types of interactions (non-personalised for the first interaction and personalised for the second). The third interaction has the same structure as the second interaction, and it allows to evaluate if the dialogue system recalled the most common order. In test and development sets, 100 customers from database A have 2 interactions each with the bot that for evaluating the fourth and fifth interaction. AND 100 customers from database B have 2 interactions, for evaluating the first and second interactions with the customer. In OOV set, 100 customers from database C have 2 interactions each.

Total number of dialogues in *SecondInteraction* for tasks 1-8:
*train*: 300 = 100 (number of customers in database A) * 3 (number of interactions per customer in database A)
*test*: 400 = 100 (number of customers in database A) * 2 (number of interactions per customer in database A) + 100 (number of customers in database B) * 2 (number of interactions per customer in database B)
*dev*: 400 (same as test)
*OOV*: 200 = 100 (number of customers in database C) * 2 (number of interactions per customer in database C)


### Task 2: Recognition error 

This task trains (and evaluates) the bot to correct itself after incorrect recognitions. This type of interaction is categorised as *open-world recognition* (A. Bendale and T. Boult, "Towards Open World Recognition", 2015) in which new users are incrementally added to the recognition system. We use two metrics to evaluate the recognition: Detection and Identification Rate (DIR) and False Alarm Rate (FAR). DIR is the fraction of correctly classified probes within the probes of the enrolled users. FAR is the fraction of incorrectly classified probes within the probes of the unknown users. For this the dataset, we used DIR = 0.9 and FAR = 0.1. This task contains incorrect recognitions as well as correct ones.

There are three scenarios of incorrect recognitions as shown below in the examples.

**Example 2.1:** Customer is not known, but incorrectly recognised as a known person (FAR error)
```
True , 10 , Jennifer Lawrence
text:Hey!	labels:Hello, Jennifer Lawrence! Can I get you a large cordusio and a blueberry muffin again?
text:Do I have a twin that I don't know?	labels:Maybe it is time for me to go to the optician! What was the name?
text:It is Melissa Joan Hart.	labels:Sorry about that Melissa. Let me see if I have any previous orders from you.
False , 11 , Melissa Joan Hart
text:Sure.	labels:It seems to be your first time here! What would you like to have, Melissa?
text:Could I get a flat white, please?	labels:Which size would you like?
text:Medium would do.	labels:Would you like a delicious snack with that?
text:Could I get a double chocolate muffin, please? labels:Your order is a medium flat white and a double chocolate muffin. Did I get that right?
text:That is correct.	labels:You can pick up your order at the next counter, Melissa.
text:Cheers!	labels:Enjoy your day!  episode_done:True
```

**Example 2.2:** Customer is known, but was not recognised (DIR error)
```
False , 0 
text:Hey!	labels:Welcome to Botista! What would you like today?
text:Could I get a latte, please?	labels:Which size would you like?
text:Small cup.	labels:Can I offer you anything to go with your drink?
text:No, thank you.	labels:Your order is a small latte. Is that right?
text:Actually, I would like large instead.	labels:Absolutely! Your order is a large latte. Did I get that right?
text:On second thought, I would have medium instead.	labels:Sure! So that would be a medium latte. Is that right?
text:That sounds right!	labels:Perfect! What would be the name?
text:It is Ty Burrell.	labels:We will get that ready for you at the next counter, Ty.
True , 53 , Ty Burrell
text:Thanks!	labels:I wish you an amazing day!   episode_done:True
```

**Example 2.3:** Customer is known, but was incorrectly recognised (DIR error)
```
True , 53 , Ty Burrell
text:Hello!	labels:Welcome back, Ty Burrell! Can I get you a medium black tea and a gingerbread cookie again?
text:I think you are mixing me with someone.	labels:I am sorry! I must be in need of more coffee and I am at the right place! Can I take your name, please?
text:It is David Boreanaz.	labels:Let me see if I have any previous records of you, David.
True , 79 , David Boreanaz
text:Okay.	labels:Ah, of course, David! Can I get you a small mocha and a chocolate chip muffin again?
text:Definitely!	labels:Your order will be ready at the next counter, David.
text:Thanks!	labels:Have a nice day! episode_done:True
```

### Task 3: Incorrect recall

This task trains (and evaluates) the bot to correct itself after an incorrect recall of the preferences of the customer. An incorrect memory rate of 0.3 is used, that is 30% of the dialogues contain incorrect recalls (a random choice) of the preferences of known users.

**Example 3.1:** Previous order of customer 29: small espresso and a pain au chocolat
```
True , 29 , Anne Hathaway
text:Hi!	labels:Welcome back, Anne Hathaway! Would you like to have a large latte and a blueberry oatmeal again?
text:That is actually my least favourite!	labels:My memory is failing me! What would you like to have, Anne?
text:Can I have a black tea, please?	labels:Which size would you like?
text:Small, thank you!	labels:Would you like a delicious snack with that?
text:No, thanks.	labels:So that would be a small black tea. Did I get that right?
text:That sounds right!	labels:Your order will be ready at the next counter, Anne.
text:Cheers!	labels:Enjoy your day!  episode_done:True
```

### Task 4: Changes to preference

This task acknowledges that customers can change their preferences, that is for any interaction, the customer could want to have the same order as the previous times, or can ask for some other drink or snack.

**Example 4.1:** Previous order of customer 25: small black tea
```
True , 25 , Teri Hatcher
text:Hey!	labels:Welcome back, Teri Hatcher! Would you like to have a small black tea again?
text:I think I will change it this time! Could I get an Earl Grey, please?	labels:Which size would you like?
text:Medium, thank you!	labels:Anything else?
text:Just the drink, please.	labels:Your order is a medium Earl Grey. Is that right?
text:That sounds right!	labels:We will get that ready for you at the next counter, Teri.
text:Cheers!	labels:Have a great day!    episode_done:True
```

### Task 5: Recognition error + incorrect recall

This task is the combination of tasks 2 and 3.

### Task 6: Recognition error + changes to preference

This task is the combination of tasks 2 and 4.

### Task 7: Incorrect recall + changes to preference

This task is the combination of tasks 3 and 4.

### Task 8: All tasks (Recognition error + incorrect recall + changes to preference)

This task is the combination of tasks 2, 3 and 4.

## License

The Barista datasets are released under a Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License. In other words, these datasets are made available for academic research purpose only. A copy of this license is included with the data.

## Contact

For any information or for requesting Barista Datasets with different order items, customer names, or larger dataset size, contact Bahar Irfan: bahar.irfan (at) plymouth (dot) ac (dot) uk.

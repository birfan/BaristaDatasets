from __future__ import absolute_import
from __future__ import print_function

from data_utils import load_dialog_task, vectorize_data, load_candidates, vectorize_candidates, vectorize_candidates_sparse, tokenize, convert_word_index_to_sentence, log_predictions_to_file, write_result_to_csv
from sklearn import metrics
from memn2n import MemN2NDialog
from itertools import chain
from six.moves import range, reduce
import sys
import tensorflow as tf
import numpy as np
import os
import pickle
import time

tf.flags.DEFINE_float("learning_rate", 0.001, "Learning rate for Adam Optimizer.")
tf.flags.DEFINE_float("epsilon", 1e-8, "Epsilon value for Adam Optimizer.")
tf.flags.DEFINE_float("max_grad_norm", 40.0, "Clip gradients to this norm.")
tf.flags.DEFINE_integer("evaluation_interval", 1, "Evaluate and print results every x epochs") # this was 10
tf.flags.DEFINE_integer("batch_size", 32, "Batch size for training.")
tf.flags.DEFINE_integer("hops", 1, "Number of hops in the Memory Network.") # this was 3
tf.flags.DEFINE_integer("epochs", 100, "Number of epochs to train for.") # this was 200
tf.flags.DEFINE_integer("embedding_size", 20, "Embedding size for embedding matrices.")
tf.flags.DEFINE_integer("memory_size", 250, "Maximum size of memory.")
tf.flags.DEFINE_integer("task_id", 1, "task id, 0 <= id <= 8")
tf.flags.DEFINE_integer("random_state", None, "Random state.")
tf.flags.DEFINE_string("data_dir", "../../data/", "Directory containing datasets")
# tf.flags.DEFINE_string("data_dir", "../data/personalized-dialog-dataset/full", "Directory containing bAbI tasks")
tf.flags.DEFINE_string("model_dir", "checkpoints/", "Directory containing memn2n model checkpoints")
tf.flags.DEFINE_boolean("train", True, "if True, begin to train")
tf.flags.DEFINE_boolean("OOV", False, "if True, use OOV test set")
tf.flags.DEFINE_boolean("save_vocab", True, "if True, saves vocabulary")
tf.flags.DEFINE_boolean("load_vocab", True, "if True, loads vocabulary instead of building it")

# EXTRA ADDED ARGUMENTS FOR BARISTA DATASETS
tf.flags.DEFINE_string("ds_name", "barista-personalised", "Dataset name")
tf.flags.DEFINE_string("task_size", "Task1k", "Folder name for task size")
tf.flags.DEFINE_string("ds_type", "spaced_api_call", "spaced_api_call: space before punctuations and includes api_call for getname")
tf.flags.DEFINE_boolean("order_info", False, "Is most common order info included in profile information, default: False")
tf.flags.DEFINE_string("ds_format", "FB", "Dataset formatting: FB or PAI")
tf.flags.DEFINE_string("response_type", "single_bot_response", 
                       "If single_bot_response, bot always replies the same way (e.g.`Hi` to `Hi`, `Hey`, `Hello`) \
                       or multiple_bot_responses: multiple correct responses are possible")

tf.flags.DEFINE_boolean("log_predictions", False, "Log predictions to json or txt file (depending on the extension of the dupi or dupc). Needs -dupi or -dupc to be filled.")
tf.flags.DEFINE_string("dupi", None,"Dump incorrect predictions into the specified path")
tf.flags.DEFINE_string("dupc", None,"Dump correct predictions into the specified path.")
tf.flags.DEFINE_string("result_file", "results_test.csv","CSV file for logging the results of test.")
tf.flags.DEFINE_string("results_dir", "","Results directory.")

FLAGS = tf.flags.FLAGS
print("Started Task:", FLAGS.task_id)


class chatBot(object):
    def __init__(self, data_dir, model_dir, vocab_dir, task_id,
                 OOV=False,
                 memory_size=250,
                 random_state=None,
                 batch_size=32,
                 learning_rate=0.001,
                 epsilon=1e-8,
                 max_grad_norm=40.0,
                 evaluation_interval=10,
                 hops=3,
                 epochs=200,
                 embedding_size=20,
                 save_vocab=False,
                 load_vocab=False,
                 ds_name="barista-personalised",
                 task_size="Task1k",
                 ds_type="spaced_api_call", 
                 order_info=False,
                 ds_format="FB",
                 response_type="single_bot_response",
                 log_predictions=False,
                 dupi=None,
                 dupc=None,
                 result_file="results_test.csv"
                 ):
        """Creates wrapper for training and testing a chatbot model.

        Args:
            data_dir: Directory containing datasets.
            
            model_dir: Directory containing memn2n model checkpoints.

            task_id: Task id, 0 <= id <= 8. Defaults to `1`.

            OOV: If `True`, use OOV test set. Defaults to `False`

            memory_size: The max size of the memory. Defaults to `250`.

            random_state: Random state to set graph-level random seed. Defaults to `None`.

            batch_size: Size of the batch for training. Defaults to `32`.

            learning_rate: Learning rate for Adam Optimizer. Defaults to `0.001`.

            epsilon: Epsilon value for Adam Optimizer. Defaults to `1e-8`.

            max_gradient_norm: Maximum L2 norm clipping value. Defaults to `40.0`.

            evaluation_interval: Evaluate and print results every x epochs. 
            Defaults to `10`.

            hops: The number of hops over memory for responding. A hop consists 
            of reading and addressing a memory slot. Defaults to `3`.

            epochs: Number of training epochs. Defualts to `200`.

            embedding_size: The size of the word embedding. Defaults to `20`.

            save_vocab: If `True`, save vocabulary file. Defaults to `False`.

            load_vocab: If `True`, load vocabulary from file. Defaults to `False`.
            
            ds_name: Name of the dataset, `barista-personalised` (default) or `barista`.
            
            task_size: Folder containing the tasks. If there are 1k dialogues then the folder is named 
                      `Task1k` (default), other options: `Task100`, `Task10k` (for barista), 
                      `Task6k` and `SecondInteraction` (barista-personalised).
            
            ds_type: Dataset type. `spaced_api_call` (default) contains spaces before punctuations and
                     api_call for requesting the name.
            
            order_info: If profile information contains the most frequent/ recent order, this should be `True`.
                      Defaults to `False`.
            
            ds_format: Formatting of the dataset according to ParlAI: `FB` (default) or `PAI`.
            
            response_type: If `single_bot_response` (default), bot always replies the same way 
                    (e.g.`Hi` to `Hi`, `Hey`, `Hello`). If `multiple_bot_responses`, multiple correct responses are possible.
        """
        
        self.task_id = task_id
        self.model_dir = model_dir
        self.vocab_dir = vocab_dir
        self.OOV = OOV
        self.memory_size = memory_size
        self.random_state = random_state
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.max_grad_norm = max_grad_norm
        self.evaluation_interval = evaluation_interval
        self.hops = hops
        self.epochs = epochs
        self.embedding_size = embedding_size
        self.save_vocab = save_vocab
        self.load_vocab = load_vocab
        self.order_info = order_info
        self.id_example = 1
        self.log_predictions=log_predictions
        self.incorrect_pred_path = dupi
        self.correct_pred_path = dupc
        self.result_file = result_file

        self.task_dir = os.path.join(data_dir, ds_name, ds_name + '-dataset', response_type, ds_type, ds_format, task_size, 'task{num}'.format(num=self.task_id))
        # self.task_dir = data_dir

        candidates,self.candid2indx = load_candidates(self.task_dir, self.task_id, ds_name)
        self.n_cand = len(candidates)
        print("Candidate Size", self.n_cand)
        self.indx2candid = dict((self.candid2indx[key],key) 
                                for key in self.candid2indx)
        
        # Task data
        self.trainData, self.testData, self.valData = load_dialog_task(
            self.task_dir, self.task_id, self.candid2indx, self.OOV, ds_name)
        data = self.trainData + self.testData + self.valData
        
        self.build_vocab(data, candidates, self.save_vocab, self.load_vocab)
        
        self.candidates_vec = vectorize_candidates(
            candidates,self.word_idx,self.candidate_sentence_size)
        
        optimizer = tf.train.AdamOptimizer(
            learning_rate=self.learning_rate, epsilon=self.epsilon)
        
        self.sess = tf.Session()
        
        self.model = MemN2NDialog(self.batch_size, self.vocab_size, self.n_cand, 
                                  self.sentence_size, self.embedding_size, 
                                  self.candidates_vec, session=self.sess,
                                  hops=self.hops, max_grad_norm=self.max_grad_norm, 
                                  optimizer=optimizer, name='MemN2N', task_id=task_id)
        
        self.saver = tf.train.Saver(max_to_keep=50)
        
    def build_vocab(self,data,candidates,save=False,load=False):
        """Build vocabulary of words from all dialog data and candidates."""
        
        vocab_path = self.vocab_dir + 'vocab-task'+ str(self.task_id)+'.obj'
        
        if os.path.isfile(vocab_path) and load:
            # Load from vocabulary file
            vocab_file = open(vocab_path, 'rb')
            vocab = pickle.load(vocab_file)
        else:
            vocab = reduce(lambda x, y: x | y, 
                           (set(list(chain.from_iterable(s)) + q) 
                             for s, q, a in data))
            vocab |= reduce(lambda x,y: x|y, 
                            (set(candidate) for candidate in candidates) )
            vocab = sorted(vocab)
        
        self.vocab = vocab
        
        self.word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
        max_story_size = max(map(len, (s for s, _, _ in data)))
        mean_story_size = int(np.mean([ len(s) for s, _, _ in data ]))
        self.sentence_size = max(map(len, chain.from_iterable(s for s, _, _ in data)))
        self.candidate_sentence_size=max(map(len,candidates))
        query_size = max(map(len, (q for _, q, _ in data)))
        self.memory_size = min(self.memory_size, max_story_size)
        self.vocab_size = len(self.word_idx) + 1  # +1 for nil word
        self.sentence_size = max(query_size, self.sentence_size)  # for the position

        # Print parameters
        print("vocab size:", self.vocab_size)
        print("Longest sentence length", self.sentence_size)
        print("Longest candidate sentence length", self.candidate_sentence_size)
        print("Longest story length", max_story_size)
        print("Average story length", mean_story_size)

        # Save to vocabulary file
        if save:
            vocab_file = open(vocab_path, 'wb')
            pickle.dump(vocab, vocab_file)

    def train(self):
        """Runs the training algorithm over training set data.

        Performs validation at given evaluation intervals.
        """
        trainS, trainQ, trainA = vectorize_data(
            self.trainData, self.word_idx, self.sentence_size, 
            self.batch_size, self.n_cand, self.memory_size)
        valS, valQ, valA = vectorize_data(
            self.valData, self.word_idx, self.sentence_size, 
            self.batch_size, self.n_cand, self.memory_size)
        n_train = len(trainS)
        n_val = len(valS)
        print("Training Size", n_train)
        print("Validation Size", n_val)
        tf.set_random_seed(self.random_state)
        batches = zip(range(0, n_train-self.batch_size, self.batch_size), 
                      range(self.batch_size, n_train, self.batch_size))
        batches = [(start, end) for start, end in batches]
        best_validation_accuracy=0

#         summary_writer = tf.summary.FileWriter(
#             self.model_dir, self.model.graph_output.graph)

        train_writer = tf.summary.FileWriter(self.model_dir + "training", self.model.graph_output.graph)
        val_writer = tf.summary.FileWriter(self.model_dir + "validation", self.model.graph_output.graph)
        
        acc = tf.Variable(0.0)
        tf.summary.scalar("accuracy", acc)
        merged_summary = tf.summary.merge_all()

        epoch_start = 1
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
            epoch_start = int(os.path.basename(ckpt.model_checkpoint_path).split('-')[1]) + 1
            print("starting a checkpoint from epoch:", epoch_start)

        for t in range(epoch_start, self.epochs+1):                
        # Training loop
            print('Epoch', t)
            np.random.shuffle(batches)
            total_cost = 0.0
            for start, end in batches:
                s = trainS[start:end]
                q = trainQ[start:end]
                a = trainA[start:end]
                cost_t = self.model.batch_fit(s, q, a)
                total_cost += cost_t
            if t % self.evaluation_interval == 0:
                # Perform validation
                train_preds = self.batch_predict(trainS,trainQ,n_train)
                val_preds = self.batch_predict(valS,valQ,n_val)
                # accuracy score evaluates exact match between the predicted response and the correct response
                train_acc = metrics.accuracy_score(np.array(train_preds), trainA)
                val_acc = metrics.accuracy_score(val_preds, valA)
                print('-----------------------')
                print('Epoch', t)
                print('Total Cost:', total_cost)
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                print('-----------------------')

#                 # Write summary
#                 train_acc_summary = tf.summary.scalar(
#                     'task' + str(self.task_id) + '/' + 'train_acc', 
#                     tf.constant((train_acc), dtype=tf.float32))
#                 val_acc_summary = tf.summary.scalar(
#                     'task' + str(self.task_id) + '/' + 'val_acc', 
#                     tf.constant((val_acc), dtype=tf.float32))
#                 merged_summary = tf.summary.merge([train_acc_summary, val_acc_summary])
#                 summary_str = self.sess.run(merged_summary)
#                 summary_writer.add_summary(summary_str, t)
#                 summary_writer.flush()

                # for training
                summary = self.sess.run(merged_summary, {acc: train_acc})
                train_writer.add_summary(summary, t)
                train_writer.flush()
            
                # for validation
                summary = self.sess.run(merged_summary, {acc: val_acc})
                val_writer.add_summary(summary, t)
                val_writer.flush()
                
                if val_acc > best_validation_accuracy:
                    print('Saving checkpoint')
                    best_validation_accuracy = val_acc
                    self.saver.save(self.sess, self.model_dir+'model.ckpt',
                                    global_step=t)
                    
                if train_acc == 1.0 and val_acc == 1.0:
                    print("Reached 100% accuracy on training and validation sets.")
                    break
                    
    def test(self):
        """Runs testing on testing set data.

        Loads best performing model weights based on validation accuracy.
        """
        start_test = time.time()
        ckpt = tf.train.get_checkpoint_state(self.model_dir)
        if ckpt and ckpt.model_checkpoint_path:
            self.saver.restore(self.sess, ckpt.model_checkpoint_path)
        else:
            print("...no checkpoint found...")
        testS, testQ, testA = vectorize_data(
            self.testData, self.word_idx, self.sentence_size, 
            self.batch_size, self.n_cand, self.memory_size)
        n_test = len(testS)
        print("Testing Size", n_test)
        
        test_preds = self.batch_predict(testS, testQ, n_test)
        test_acc = metrics.accuracy_score(test_preds, testA)
        test_time = time.time() - start_test
        print("Testing Accuracy:", test_acc)

        # # Un-comment below to view correct responses and predictions 
        # for pred in test_preds:
        #    print(pred, self.indx2candid[pred])

        #BI: print results to csv file
        write_result_to_csv({'exs': n_test, 'accuracy': test_acc}, self.result_file, self.task_id, OOV=self.OOV, test_time=test_time)
   
        #BI: dump predictions to file
        for num_pred in range(len(test_preds)):
            answer = testA[num_pred].tolist()

            contex = ""
            for s in testS[num_pred]:
                stor = convert_word_index_to_sentence(s, self.vocab)
                if stor:
                    contex+=stor
            contex+=convert_word_index_to_sentence(testQ[num_pred], self.vocab) 
            if self.log_predictions:
                try:
                    self.id_example = log_predictions_to_file(contex, self.indx2candid[test_preds[num_pred]], self.indx2candid[answer], profile="", id_example=self.id_example, incorrect_pred_path=self.incorrect_pred_path, correct_pred_path=self.correct_pred_path)
                except:
                    pass

    def batch_predict(self,S,Q,n):
        """Predict answers over the passed data in batches.

        Args:
            S: Tensor (None, memory_size, sentence_size)
            Q: Tensor (None, sentence_size)
            n: int

        Returns:
            preds: Tensor (None, vocab_size)
        """
        preds = []
        for start in range(0, n, self.batch_size):
            end = start + self.batch_size
            s = S[start:end]
            q = Q[start:end]
            pred = self.model.predict(s, q)
            preds += list(pred)
        return preds

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    if FLAGS.order_info and "order" not in FLAGS.ds_name:
        results_dir = FLAGS.results_dir + FLAGS.ds_name + "-order/" + FLAGS.task_size + "/" + "hop" + str(FLAGS.hops) + "/"
    else:
        results_dir = FLAGS.results_dir + FLAGS.ds_name + "/" + FLAGS.task_size + "/" + "hop" + str(FLAGS.hops) + "/"
        
    model_dir = results_dir + FLAGS.model_dir + "task" + str(FLAGS.task_id) + "/"
    vocab_dir = results_dir + "data/"
    log_dir= results_dir + "log/" + "task" + str(FLAGS.task_id) + "/"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    if not os.path.exists(vocab_dir):
        os.makedirs(vocab_dir)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    if FLAGS.dupi and FLAGS.log_predictions:
        dupi=log_dir + FLAGS.dupi
    else:
        dupi=None

    if FLAGS.dupc and FLAGS.log_predictions:
        dupc=log_dir + FLAGS.dupc
    else:
        dupc=None
        
    result_file= results_dir + "log/" + FLAGS.result_file 
    
    chatbot = chatBot(FLAGS.data_dir, model_dir, vocab_dir, FLAGS.task_id, OOV=FLAGS.OOV,
                      batch_size=FLAGS.batch_size, memory_size=FLAGS.memory_size,
                      epochs=FLAGS.epochs, evaluation_interval=FLAGS.evaluation_interval,
                      hops=FLAGS.hops, save_vocab=FLAGS.save_vocab,
                      load_vocab=FLAGS.load_vocab, learning_rate=FLAGS.learning_rate,
                      embedding_size=FLAGS.embedding_size, 
                      ds_name=FLAGS.ds_name, task_size=FLAGS.task_size,
                      ds_type=FLAGS.ds_type, order_info=FLAGS.order_info, 
                      ds_format=FLAGS.ds_format, response_type=FLAGS.response_type,
                      log_predictions=FLAGS.log_predictions, dupi=dupi, dupc=dupc,
                      result_file=result_file)
    
    if FLAGS.train:
        chatbot.train()
    else:
        chatbot.test()
    
    chatbot.close_session()

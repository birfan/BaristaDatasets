from make_tensor import make_tensor, load_vocab
from model import Model
from sys import argv
from utils import batch_iter, convert_BOW_sentence, log_predictions_to_file, write_result_to_csv
from tqdm import tqdm
import numpy as np
import tensorflow as tf
import argparse
import os
import time


def main(test_tensor, candidates_tensor, model, checkpoint_dir, 
         vocab_inverse=None, task_id=1, result_file=None, OOV=False,
         incorrect_pred_path=None, correct_pred_path=None):
    start_test = time.time()
    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())
        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        saver.restore(sess, ckpt.model_checkpoint_path)
        print("Candidate Size", candidates_tensor.shape[0])
        print("vocab size:", candidates_tensor.shape[2])
        print("Testing Size", test_tensor.shape[0])
        (pos, neg, accuracy) = evaluate(test_tensor, candidates_tensor, sess, model, vocab_inverse,
                                        incorrect_pred_path=incorrect_pred_path, correct_pred_path=correct_pred_path)
        test_time = time.time() - start_test
        print("Testing Accuracy:", accuracy)
        #BI: print results to csv file
        write_result_to_csv({'exs': test_tensor.shape[0], 'accuracy': accuracy}, result_file, task_id, OOV=OOV, test_time=test_time)
      
  
def evaluate(test_tensor, candidates_tensor, sess, model, 
             vocab_inverse=None, incorrect_pred_path=None, correct_pred_path=None):
    neg = 0
    pos = 0
    id_example=1
    for row in tqdm(test_tensor):
        true_context = [row[0]]
        test_score = sess.run(
            model.f_pos,
            feed_dict={model.context_batch: true_context,
                       model.response_batch: [row[1]],
                       model.neg_response_batch: [row[1]]}
        )
        test_score = test_score[0]
        is_pos, predicted = evaluate_one_row(candidates_tensor, true_context, sess, model, test_score, row[1])

        if vocab_inverse:
            try:
                id_example = log_predictions_to_file(row[0], predicted,
                                                     row[1], profile="", is_pos=is_pos,
                                                     id_example=id_example, 
                                                     incorrect_pred_path=incorrect_pred_path, 
                                                     correct_pred_path=correct_pred_path, vocab=vocab_inverse)
            except:
                pass

        if is_pos:
            pos += 1
        else:
            neg += 1
    return (pos, neg, pos / (pos+neg))


def evaluate_one_row(candidates_tensor, true_context, sess, model, test_score, true_response):
    for batch in batch_iter(candidates_tensor, 512):
        candidate_responses = batch[:, 0, :]
        context_batch = np.repeat(true_context, candidate_responses.shape[0], axis=0)

        scores = sess.run(
            model.f_pos,
            feed_dict={model.context_batch: context_batch,
                       model.response_batch: candidate_responses,
                       model.neg_response_batch: candidate_responses}
        )
        for ind, score in enumerate(scores):
            if score == float('Inf') or score == -float('Inf') or score == float('NaN'):
                print(score, ind, scores[ind])
                raise ValueError
            if score >= test_score and not np.array_equal(candidate_responses[ind], true_response):
                return False, candidate_responses[ind]
    return True, true_response


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--test', help='Path to test filename')
    parser.add_argument('--vocab', default='data/vocab.tsv')
    parser.add_argument('--candidates', default='data/candidates.tsv')
    parser.add_argument('--checkpoint_dir')
    parser.add_argument('--log_dir', default='')
    parser.add_argument('--emb_dim', type=int, default=32)
    parser.add_argument('--OOV', action='store_true',
                        help='OOV test set.')
    parser.add_argument('--result_file', default='results_test.csv')
    parser.add_argument('--task_id', type=int, default=1)
    parser.add_argument('--log_predictions', action='store_true', help="Log predictions to json or txt file (depending on the extension of the dupi or dupc). Needs -dupi or -dupc to be filled.")    
    parser.add_argument('--dupi', default=None, help="Dump incorrect predictions into the specified path")
    parser.add_argument('--dupc', default=None, help="Dump correct predictions into the specified path")

    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = _parse_args()
    vocab, vocab_inverse = load_vocab(args.vocab)
    test_tensor = make_tensor(args.test, vocab)
    candidates_tensor = make_tensor(args.candidates, vocab)
    model = Model(len(vocab), args.emb_dim)
    if args.log_predictions:
        task_log = args.log_dir + "/" + "task" + str(args.task_id) + "/"
        if args.dupi:
            dupi = task_log + args.dupi
        if args.dupc:
            dupc = task_log + args.dupc
        if not os.path.exists(task_log):
            os.makedirs(task_log)
    else:
        dupi = None
        dupc = None
    main(test_tensor, candidates_tensor, model, args.checkpoint_dir, vocab_inverse, 
         task_id=args.task_id, result_file=args.result_file, OOV=args.OOV,
         incorrect_pred_path=dupi, correct_pred_path=dupc)
      
    

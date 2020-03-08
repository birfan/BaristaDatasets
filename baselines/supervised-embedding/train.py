import tensorflow as tf
import numpy as np
import argparse
import logging
import sys
from tqdm import tqdm
from make_tensor import make_tensor, load_vocab
from model import Model
from sys import argv
from test import evaluate
from utils import batch_iter, neg_sampling_iter
import os

def _setup_logger():
    logging.basicConfig(
        format='[%(levelname)s] %(asctime)s: %(message)s (%(pathname)s:%(lineno)d)',
        datefmt="%Y-%m-%dT%H:%M:%S%z",
        stream=sys.stdout)
    logger = logging.getLogger('barista-dialog')
    logger.setLevel(logging.DEBUG)
    return logger


#logger = _setup_logger()


def _parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--train', help='Path to train filename')
    parser.add_argument('--dev', help='Path to dev filename')
    parser.add_argument('--vocab', default='data/vocab.tsv')
    parser.add_argument('--candidates', default='data/candidates.tsv')
    parser.add_argument('--emb_dim', default=32, type=int)
    parser.add_argument('--save_dir')
    parser.add_argument('--margin', type=float, default=0.01)
    parser.add_argument('--negative_cand', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--task', default="task1")
    parser.add_argument('--batchsize', type=int, default=32)
    
    args = parser.parse_args()

    return args


def _train(train_tensor, batch_size, neg_size, model, optimizer, sess):
    avg_loss = 0
    for batch in batch_iter(train_tensor, batch_size, True):
        for neg_batch in neg_sampling_iter(train_tensor, batch_size, neg_size):
            loss = sess.run(
                [model.loss, optimizer],
                feed_dict={model.context_batch: batch[:, 0, :],
                           model.response_batch: batch[:, 1, :],
                           model.neg_response_batch: neg_batch[:, 1, :]}
            )
            avg_loss += loss[0]
    avg_loss = avg_loss / (train_tensor.shape[0]*neg_size)
    return avg_loss


def _forward_all(dev_tensor, model, sess):
    avg_dev_loss = 0
    for batch in batch_iter(dev_tensor, 256):
        for neg_batch in neg_sampling_iter(dev_tensor, 256, 1, 42):
            loss = sess.run(
                [model.loss],
                feed_dict={model.context_batch: batch[:, 0, :],
                           model.response_batch: batch[:, 1, :],
                           model.neg_response_batch: neg_batch[:, 1, :]}
            )
            avg_dev_loss += loss[0]
    avg_dev_loss = avg_dev_loss / (dev_tensor.shape[0]*1)
    return avg_dev_loss


def main(train_tensor, dev_tensor, candidates_tensor, model, config, task_name):
    print("Run main with config:", config)
    #logger.info('Run main with config {}'.format(config))

    epochs = config['epochs']
    batch_size = config['batch_size']
    negative_cand = config['negative_cand']
    save_dir = config['save_dir']
    model_dir = os.path.dirname(os.path.abspath(save_dir))
    evaluation_interval = config['evaluation_interval']

    # TODO: Add LR decay
    optimizer = tf.train.AdamOptimizer(config['lr']).minimize(model.loss)

    prev_best_accuracy = 0

    saver = tf.train.Saver()
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    

    train_writer = tf.summary.FileWriter(model_dir + "/training", model.loss.graph)
    val_writer = tf.summary.FileWriter(model_dir + "/validation", model.loss.graph)
    
    acc = tf.Variable(0.0)
    tf.summary.scalar("accuracy", acc)
    merged_summary = tf.summary.merge_all()

    print("Started Task:", task_name.replace("task",""))
    print("Candidate Size", candidates_tensor.shape[0])
    print("vocab size:", candidates_tensor.shape[2])
    print("Training Size", train_tensor.shape[0])
    print("Validation Size", dev_tensor.shape[0])
    
    with tf.Session(config=config) as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(1, epochs+1):
            avg_loss = _train(train_tensor, batch_size, negative_cand, model, optimizer, sess)
            # TODO: Refine dev loss calculation
            avg_dev_loss = _forward_all(dev_tensor, model, sess)
            print('-----------------------')
            print('Epoch', epoch)
            print('Train loss', avg_loss)
            print('Dev loss', avg_dev_loss)
            # logger.info('Epoch: {}; Train loss: {}; Dev loss: {};'.format(epoch, avg_loss, avg_dev_loss))
                
            if epoch % evaluation_interval == 0:
                train_eval = evaluate(train_tensor, candidates_tensor, sess, model)
                dev_eval = evaluate(dev_tensor, candidates_tensor, sess, model)
                train_acc = train_eval[2]
                val_acc = dev_eval[2]
                print('Training Accuracy:', train_acc)
                print('Validation Accuracy:', val_acc)
                
                # Write summary
    
                # for training
                summary = sess.run(merged_summary, {acc: train_acc})
                train_writer.add_summary(summary, epoch)
                train_writer.flush()
            
                # for validation
                summary = sess.run(merged_summary, {acc: val_acc})
                val_writer.add_summary(summary, epoch)
                val_writer.flush()
            
                # logger.info('Evaluation: {}'.format(dev_eval))
                if val_acc > prev_best_accuracy:
#                     logger.debug('Saving checkpoint')
                    print('Saving checkpoint')
                    prev_best_accuracy = val_acc
                    saver.save(sess, save_dir, global_step=epoch)
                    
                if train_acc == 1.0 and val_acc == 1.0:
                    print("Reached 100% accuracy on training and validation sets.")
                    break
                
    summary_writer.close()


if __name__ == '__main__':
    args = _parse_args()
    vocab, _ = load_vocab(args.vocab)
    train_tensor = make_tensor(args.train, vocab)
    dev_tensor = make_tensor(args.dev, vocab)
    candidates_tensor = make_tensor(args.candidates, vocab)
    config = {'epochs': 25, #epochs was 15 - then I made 100 but it was too long
              'negative_cand': args.negative_cand, 'save_dir': args.save_dir,
              'lr': args.learning_rate, 'evaluation_interval': 1} #evaluation_interval used to be 2
    if args.batchsize:
        config['batch_size'] = args.batchsize #batchsize was 32 (increased to 128 use gpu better)
    else:
        config['batch_size'] = 32
    model = Model(len(vocab), emb_dim=args.emb_dim, margin=args.margin)
    main(train_tensor, dev_tensor, candidates_tensor, model, config, args.task)

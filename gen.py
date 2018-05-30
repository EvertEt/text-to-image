#! /usr/bin/python
# -*- coding: utf8 -*-
""" GAN-CLS """
import logging
import pickle
import time

import nltk

import model
from model import *
from utils import *

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str, default='102flowers', help='102flowers | birds')
    parser.add_argument('id', type=str, default='000', help='id')

    args = parser.parse_args()

    dataset = args.dataset
    id = args.id

    logger = logging.getLogger()

    ###======================== PREPARE DATA ====================================###

    print('Opening Vocab')
    with open('_vocab_' + dataset + '.pickle', 'rb') as f:
        print('Opened Vocab')
        vocab = pickle.load(f)
    print('Loaded Vocab')
    print('Loading Done')

    ni = int(np.ceil(np.sqrt(batch_size)))

    tl.files.exists_or_mkdir('gen/gan-cls_' + dataset + id)
    save_dir = 'checkpoint_' + dataset + id
    tl.files.exists_or_mkdir(save_dir)

    ###======================== DEFINE MODEL ===================================###
    t_real_pos = tf.placeholder('float32', [batch_size, 2], name='real_pos')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    ## training inference for txt2img
    generator_txt2img = model.generator_txt2img_resnet

    net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False)

    ## testing inference for txt2img
    net_g, _ = generator_txt2img(t_z, t_real_pos, rnn_embed(t_real_caption, is_train=False, reuse=True).outputs, is_train=False, reuse=False, batch_size=batch_size)

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    # load the latest checkpoints
    net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
    # net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    # net_d_name = os.path.join(save_dir, 'net_d.npz')

    if not load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn):
        exit(1)

    if not load_and_assign_npz(sess=sess, name=net_g_name, model=net_g):
        exit(1)

    ## seed for generation, z and sentence ids
    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

    n = int(sample_size / ni)
    if dataset == '102flowers':
        sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * n + \
                          ["this flower has petals that are yellow, white and purple and has dark lines"] * n + \
                          ["the petals on this flower are white with a yellow center"] * n + \
                          ["this flower has a lot of small round pink petals."] * n + \
                          ["this flower is orange in color, and has petals that are ruffled and rounded."] * n + \
                          ["the flower has yellow petals and the center of it is brown."] * n + \
                          ["this flower has petals that are blue and white."] * n + \
                          ["these white flowers have petals that start off white in color and end in a white towards the tips."] * n
    else:
        sample_sentence = ["this magnificent fellow is almost all black with a red crest, and white cheek patch."] * n + \
                          ["this small bird has a pink breast and crown, and black primaries and secondaries."] * n + \
                          ["an all black bird with a distinct thick, rounded bill."] * n + \
                          ["this small bird has a yellow breast, brown crown, and black superciliary"] * n + \
                          ["a tiny bird, with a tiny beak, tarsus and feet, a blue crown, blue coverts, and black cheek patch"] * n + \
                          ["this bird is different shades of brown all over with white and black spots on its head and back"] * n + \
                          ["the gray bird has a light grey head and grey webbed feet"] * n + \
                          ["the gray bird has a light grey head and grey webbed feet"] * n

    for i, sentence in enumerate(sample_sentence):
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]  # add END_ID
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    sample_pos = [[0.2, 0.2] for _ in range(sample_size)]

    img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
        t_real_caption: sample_sentence,
        t_real_pos: sample_pos,
        t_z: sample_seed})

    save_images(img_gen, [ni, ni], 'gen/gen_' + dataset + id + str(int(time.time())) + '.png')

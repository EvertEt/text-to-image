#! /usr/bin/python
# -*- coding: utf8 -*-
""" GAN-CLS """
import datetime
import logging
import pickle
import time

import nltk
from tensorlayer.cost import *

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


    def make_gif():
        import imageio
        filenames = tl.files.load_file_list('samples/step1_gan-cls_' + dataset + id, regx='^train_\d+0\.png', printable=False)
        with imageio.get_writer('train.gif', mode='I', fps=0.1) as writer:
            for filename in filenames:
                image = imageio.imread('samples/step1_gan-cls_' + dataset + '/' + filename)
                writer.append_data(image)


    ###======================== PREPARE DATA ====================================###

    print('Opening Vocab')
    with open('_vocab_' + dataset + '.pickle', 'rb') as f:
        print('Opened Vocab')
        vocab = pickle.load(f)
    print('Loaded Vocab')
    print('Opening Train')
    with open('_image_train_' + dataset + '.pickle', 'rb') as f:
        print('Opened Train')
        _, images_train = pickle.load(f)
    print('Loaded Train')
    print('Opening Test')
    with open('_image_test_' + dataset + '.pickle', 'rb') as f:
        print('Opened Test')
        _, images_test = pickle.load(f)
    print('Loaded Test')
    print('Opening n')
    with open('_n_' + dataset + '.pickle', 'rb') as f:
        print('Opened n')
        n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
    print('Loaded n')
    print('Opening Caption')
    with open('_caption_' + dataset + '.pickle', 'rb') as f:
        print('Opened Caption')
        captions_ids_train, captions_ids_test = pickle.load(f)
    print('Loaded Caption')
    if dataset == 'birds':
        print('Opening BB')
        with open('_bb_' + dataset + '.pickle', 'rb') as f:
            print('Opened BB')
            bb_train, bb_test = pickle.load(f)
        print('Loaded BB')
    else:
        bb_train, bb_test = [], []
    print('Loading Done')
    # images_train_256 = np.array(images_train_256)
    # images_test_256 = np.array(images_test_256)
    images_train = np.array(images_train)
    images_test = np.array(images_test)
    bb_train = np.array(bb_train)
    bb_test = np.array(bb_test)

    ni = int(np.ceil(np.sqrt(batch_size)))

    tl.files.exists_or_mkdir('samples/step1_gan-cls_' + dataset + id)
    save_dir = 'checkpoint_' + dataset + id
    tl.files.exists_or_mkdir(save_dir)

    ###======================== DEFINE MODEL ===================================###
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='wrong_image')
    t_real_pos = tf.placeholder('float32', [batch_size, 4], name='real_pos')
    t_wrong_pos = tf.placeholder('float32', [batch_size, 4], name='wrong_pos')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    ## training inference for text-to-image mapping
    net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
    x = net_cnn.outputs
    v = rnn_embed(t_real_caption, is_train=True, reuse=False).outputs
    x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs
    v_w = rnn_embed(t_wrong_caption, is_train=True, reuse=True).outputs

    alpha = 0.2  # margin alpha
    rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
               tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))
    summ_rnn_loss = tf.summary.scalar('rnn_loss', rnn_loss)

    ## training inference for txt2img
    generator_txt2img = model.generator_txt2img_resnet
    discriminator_txt2img = model.discriminator_txt2img_resnet

    net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=True)

    net_fake_image, _ = generator_txt2img(t_z, t_real_pos, net_rnn.outputs, is_train=True, reuse=False, batch_size=batch_size)

    net_d, disc_fake_image_logits = discriminator_txt2img(net_fake_image.outputs, t_real_pos, net_rnn.outputs, is_train=True, reuse=False)

    _, disc_real_image_logits = discriminator_txt2img(t_real_image, t_real_pos, net_rnn.outputs, is_train=True, reuse=True)

    _, disc_mismatch_logits = discriminator_txt2img(t_real_image, t_wrong_pos, rnn_embed(t_wrong_caption, is_train=False, reuse=True).outputs, is_train=True, reuse=True)
    # TODO mismatchen ook met pos en text apart testen

    ## testing inference for txt2img
    net_g, _ = generator_txt2img(t_z, t_real_pos, rnn_embed(t_real_caption, is_train=False, reuse=True).outputs, is_train=False, reuse=True, batch_size=batch_size)

    d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
    summ_d_loss1 = tf.summary.scalar('d_loss1', d_loss1)
    d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits, tf.zeros_like(disc_mismatch_logits), name='d2')
    summ_d_loss2 = tf.summary.scalar('d_loss2', d_loss2)
    d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
    summ_d_loss3 = tf.summary.scalar('d_loss3', d_loss3)
    d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
    summ_d_loss_tot = tf.summary.scalar('d_loss', d_loss)
    summ_d_loss = tf.summary.merge([summ_d_loss1, summ_d_loss2, summ_d_loss3, summ_d_loss_tot])

    g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')
    summ_g_loss = tf.summary.scalar('g_loss', g_loss)

    ####======================== DEFINE TRAIN OPTS ==============================###
    lr = 0.0002
    lr_decay = 0.5  # decay factor for adam, https://github.com/reedscot/icml2016/blob/master/main_cls_int.lua  https://github.com/reedscot/icml2016/blob/master/scripts/train_flowers.sh
    decay_every = 100  # https://github.com/reedscot/icml2016/blob/master/main_cls.lua
    beta1 = 0.5
    cnn_vars = tl.layers.get_variables_with_name('cnn', True)
    rnn_vars = tl.layers.get_variables_with_name('rnn', True)
    d_vars = tl.layers.get_variables_with_name('discriminator', True)
    g_vars = tl.layers.get_variables_with_name('generator', True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr, trainable=False)
        summ_lr_v = tf.summary.scalar('lr_v', lr_v)

    d_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(d_loss, var_list=d_vars)
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)
    # e_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(e_loss, var_list=e_vars + c_vars)
    grads, _ = tf.clip_by_global_norm(tf.gradients(rnn_loss, rnn_vars + cnn_vars), 10)
    optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1)  # optimizer = tf.train.GradientDescentOptimizer(lre)
    rnn_optim = optimizer.apply_gradients(zip(grads, rnn_vars + cnn_vars))

    # adam_vars = tl.layers.get_variables_with_name('Adam', False, True)

    ###============================ TRAINING ====================================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))
    tl.layers.initialize_global_variables(sess)

    tb_writer = tf.summary.FileWriter('tb_logs/' + dataset + id + '/', sess.graph)

    # load the latest checkpoints
    net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
    net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
    net_g_name = os.path.join(save_dir, 'net_g.npz')
    net_d_name = os.path.join(save_dir, 'net_d.npz')

    # load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
    # load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
    # load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
    # load_and_assign_npz(sess=sess, name=net_d_name, model=net_d)

    ## seed for generation, z and sentence ids
    sample_size = batch_size
    sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
    # sample_seed = np.random.uniform(low=-1, high=1, size=(sample_size, z_dim)).astype(np.float32)
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
        sample_sentence = ["this vibrant red bird has a pointed black beak."] * n + \
                          ["this bird is yellowish orange with black wings"] * n + \
                          ["the bright blue bird has a white colored belly"] * n + \
                          ["this small bird has a pink breast and crown, and black primaries and secondaries."] * n + \
                          ["This birds is completely blue."] * n + \
                          ["an all black bird with a distinct thick, rounded bill"] * n + \
                          ["the gray bird has a light grey head and grey webbed feet."] * n + \
                          ["This blue bird has white wings."] * n

    # sample_sentence = captions_ids_test[0:sample_size]
    for i, sentence in enumerate(sample_sentence):
        # print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]  # add END_ID
        # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
        # print(sample_sentence[i])
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    sample_pos = [[0.2, 0.2] for _ in range(sample_size)]

    n_epoch = 600
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)
    for epoch in range(0, n_epoch + 1):
        start_time = time.time()

        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            _, summ = sess.run([tf.assign(lr_v, lr * new_lr_decay), summ_lr_v])
            tb_writer.add_summary(summ, epoch)
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()
            ## get matched text
            idexs = get_random_int(0, n_captions_train - 1, batch_size)
            b_real_caption = captions_ids_train[idexs]
            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')

            ## get real image
            rounded_idexs = np.floor(np.asarray(idexs).astype('float') / n_captions_per_image).astype('int')
            b_real_images = images_train[rounded_idexs]

            ## get real bb
            b_real_pos = bb_train[rounded_idexs] if dataset == 'birds' else [[32, 32, 20, 20] for _ in range(len(rounded_idexs))]
            # b_real_pos = list(map(get_center, b_real_pos))

            ## get wrong caption
            idexs = get_random_int(0, n_captions_train - 1, batch_size)
            b_wrong_caption = captions_ids_train[idexs]
            b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')

            ## get wrong image
            idexs2 = get_random_int(0, n_images_train - 1, batch_size)
            b_wrong_images = images_train[idexs2]

            ## get wrong bb
            b_wrong_pos = bb_train[idexs2] if dataset == 'birds' else [[32, 32, 20, 20] for _ in range(len(idexs2))]
            # b_wrong_pos = list(map(get_center, b_wrong_pos))

            ## get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

            # [0, 255] --> [-1, 1] + augmentation
            b_real_images, b_real_pos = zip(*threading_data(list(zip(b_real_images, b_real_pos)), prepro_img, img_size=image_size))
            b_wrong_images, b_wrong_pos = zip(*threading_data(list(zip(b_wrong_images, b_wrong_pos)), prepro_img, img_size=image_size))

            ## updates text-to-image mapping
            if epoch < 50:
                errRNN, _, summ = sess.run([rnn_loss, rnn_optim, summ_rnn_loss], feed_dict={
                    t_real_image: b_real_images,
                    t_wrong_image: b_wrong_images,
                    t_real_caption: b_real_caption,
                    t_wrong_caption: b_wrong_caption})
                tb_writer.add_summary(summ, epoch)
            else:
                errRNN = 0

            ## updates D
            errD, _, summ = sess.run([d_loss, d_optim, summ_d_loss], feed_dict={
                t_real_image: b_real_images,
                # t_wrong_image : b_wrong_images,
                t_wrong_caption: b_wrong_caption,
                t_real_caption: b_real_caption,
                t_real_pos: b_real_pos,
                t_wrong_pos: b_wrong_pos,
                t_z: b_z})
            tb_writer.add_summary(summ, epoch)

            ## updates G
            errG, _, summ = sess.run([g_loss, g_optim, summ_g_loss], feed_dict={
                t_real_caption: b_real_caption,
                t_real_pos: b_real_pos,
                t_z: b_z})
            tb_writer.add_summary(summ, epoch)

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, rnn_loss: %.8f"
                  % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errRNN))

        if (epoch + 1) % print_freq == 0:
            print(" ** [%s] Epoch %d took %fs" % (datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), epoch, time.time() - start_time))
            img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
                t_real_caption: sample_sentence,
                t_real_pos: sample_pos,
                t_z: sample_seed})

            # img_gen = threading_data(img_gen, prepro_img, mode='rescale')
            save_images(img_gen, [ni, ni], 'samples/step1_gan-cls_' + dataset + id + '/train_{:02d}.png'.format(epoch))

        ## save model
        if (epoch != 0) and (epoch % 10) == 0:
            tl.files.save_npz(net_cnn.all_params, name=net_cnn_name, sess=sess)
            tl.files.save_npz(net_rnn.all_params, name=net_rnn_name, sess=sess)
            tl.files.save_npz(net_g.all_params, name=net_g_name, sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name, sess=sess)
            print("[*] Save checkpoints SUCCESS!")

        if (epoch != 0) and (epoch % 100) == 0:
            tl.files.save_npz(net_cnn.all_params, name=net_cnn_name + str(epoch), sess=sess)
            tl.files.save_npz(net_rnn.all_params, name=net_rnn_name + str(epoch), sess=sess)
            tl.files.save_npz(net_g.all_params, name=net_g_name + str(epoch), sess=sess)
            tl.files.save_npz(net_d.all_params, name=net_d_name + str(epoch), sess=sess)

        # if (epoch != 0) and (epoch % 200) == 0:
        #     sess.run(tf.initialize_variables(adam_vars))
        #     print("Re-initialize Adam")

    make_gif()

#! /usr/bin/python
# -*- coding: utf8 -*-
""" GAN-CLS """

import pickle

import nltk

import model
from model import *
from tensorlayer.cost import *
from utils import *

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument('dataset', type=str, default='102flowers', help='102flowers | birds')

    args = parser.parse_args()

    dataset = args.dataset


    def make_gif():
        import imageio
        filenames = tl.files.load_file_list('samples/step1_gan-cls_' + dataset, regx='^train_\d+0\.png', printable=False)
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
    print('Loading Done')
    # images_train_256 = np.array(images_train_256)
    # images_test_256 = np.array(images_test_256)
    images_train = np.array(images_train)
    images_test = np.array(images_test)

    # print(n_captions_train, n_captions_test)
    # exit()

    ni = int(np.ceil(np.sqrt(batch_size)))
    # os.system("mkdir samples")
    # os.system("mkdir samples/step1_gan-cls")
    # os.system("mkdir checkpoint")
    tl.files.exists_or_mkdir("samples/step1_gan-cls_" + dataset)
    # tl.files.exists_or_mkdir("samples/step_pretrain_encoder")
    tl.files.exists_or_mkdir("checkpoint_" + dataset)
    save_dir = "checkpoint_" + dataset

    ###======================== DEFINE MODEL ===================================###
    t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='real_image')
    t_wrong_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name='wrong_image')
    t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
    t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
    t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

    with tl.ops.suppress_stdout():
        ## training inference for text-to-image mapping
        net_cnn = cnn_encoder(t_real_image, is_train=True, reuse=False)
        x = net_cnn.outputs
        v = rnn_embed(t_real_caption, is_train=True, reuse=False).outputs
        x_w = cnn_encoder(t_wrong_image, is_train=True, reuse=True).outputs
        v_w = rnn_embed(t_wrong_caption, is_train=True, reuse=True).outputs

    alpha = 0.2  # margin alpha
    rnn_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x, v_w))) + \
               tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, v) + cosine_similarity(x_w, v)))

    ## training inference for txt2img
    generator_txt2img = model.generator_txt2img_resnet
    discriminator_txt2img = model.discriminator_txt2img_resnet

    with tl.ops.suppress_stdout():
        net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=True)
        net_fake_image, _ = generator_txt2img(t_z,
                                              net_rnn.outputs,
                                              is_train=True, reuse=False, batch_size=batch_size)
        # + tf.random_normal(shape=net_rnn.outputs.get_shape(), mean=0, stddev=0.02), # NOISE ON RNN
        net_d, disc_fake_image_logits = discriminator_txt2img(
            net_fake_image.outputs, net_rnn.outputs, is_train=True, reuse=False)
        _, disc_real_image_logits = discriminator_txt2img(
            t_real_image, net_rnn.outputs, is_train=True, reuse=True)
        _, disc_mismatch_logits = discriminator_txt2img(
            # t_wrong_image,
            t_real_image,
            # net_rnn.outputs,
            rnn_embed(t_wrong_caption, is_train=False, reuse=True).outputs,
            is_train=True, reuse=True)
        ## testing inference for txt2img
        net_g, _ = generator_txt2img(t_z,
                                     rnn_embed(t_real_caption, is_train=False, reuse=True).outputs,
                                     is_train=False, reuse=True, batch_size=batch_size)

        d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
        d_loss2 = tl.cost.sigmoid_cross_entropy(disc_mismatch_logits, tf.zeros_like(disc_mismatch_logits), name='d2')
        d_loss3 = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.zeros_like(disc_fake_image_logits), name='d3')
        d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5
        g_loss = tl.cost.sigmoid_cross_entropy(disc_fake_image_logits, tf.ones_like(disc_fake_image_logits), name='g')

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
    if dataset == '102flowers':
        sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size / ni) + \
                          ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size / ni) + \
                          ["the petals on this flower are white with a yellow center"] * int(sample_size / ni) + \
                          ["this flower has a lot of small round pink petals."] * int(sample_size / ni) + \
                          ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size / ni) + \
                          ["the flower has yellow petals and the center of it is brown."] * int(sample_size / ni) + \
                          ["this flower has petals that are blue and white."] * int(sample_size / ni) + \
                          ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size / ni)
    else:
        sample_sentence = ["this vibrant red bird has a pointed black beak."] * int(sample_size / ni) + \
                          ["this bird is yellowish orange with black wings"] * int(sample_size / ni) + \
                          ["the bright blue bird has a white colored belly"] * int(sample_size / ni) + \
                          ["this small bird has a pink breast and crown, and black primaries and secondaries."] * int(sample_size / ni) + \
                          ["This birds is completely blue."] * int(sample_size / ni) + \
                          ["an all black bird with a distinct thick, rounded bill"] * int(sample_size / ni) + \
                          ["the gray bird has a light grey head and grey webbed feet."] * int(sample_size / ni) + \
                          ["This blue bird has white wings."] * int(sample_size / ni)

    # sample_sentence = captions_ids_test[0:sample_size]
    for i, sentence in enumerate(sample_sentence):
        # print("seed: %s" % sentence)
        sentence = preprocess_caption(sentence)
        sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]  # add END_ID
        # sample_sentence[i] = [vocab.word_to_id(word) for word in sentence]
        # print(sample_sentence[i])
    sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

    n_epoch = 600
    print_freq = 1
    n_batch_epoch = int(n_images_train / batch_size)
    for epoch in range(0, n_epoch + 1):
        start_time = time.time()

        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay ** (epoch // decay_every)
            sess.run(tf.assign(lr_v, lr * new_lr_decay))
            log = " ** new learning rate: %f" % (lr * new_lr_decay)
            print(log)
        elif epoch == 0:
            log = " ** init lr: %f  decay_every_epoch: %d, lr_decay: %f" % (lr, decay_every, lr_decay)
            print(log)

        for step in range(n_batch_epoch):
            step_time = time.time()
            ## get matched text
            idexs = get_random_int(min=0, max=n_captions_train - 1, number=batch_size)
            b_real_caption = captions_ids_train[idexs]
            b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')

            ## get real image
            b_real_images = images_train[np.floor(np.asarray(idexs).astype('float') / n_captions_per_image).astype('int')]

            ## get wrong caption
            idexs = get_random_int(min=0, max=n_captions_train - 1, number=batch_size)
            b_wrong_caption = captions_ids_train[idexs]
            b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')

            ## get wrong image
            idexs2 = get_random_int(min=0, max=n_images_train - 1, number=batch_size)
            b_wrong_images = images_train[idexs2]

            ## get noise
            b_z = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

            # [0, 255] --> [-1, 1] + augmentation
            b_real_images = threading_data(b_real_images, prepro_img, mode='train')
            b_wrong_images = threading_data(b_wrong_images, prepro_img, mode='train')

            ## updates text-to-image mapping
            if epoch < 50:
                errRNN, _ = sess.run([rnn_loss, rnn_optim], feed_dict={
                    t_real_image: b_real_images,
                    t_wrong_image: b_wrong_images,
                    t_real_caption: b_real_caption,
                    t_wrong_caption: b_wrong_caption})
            else:
                errRNN = 0

            ## updates D
            errD, _ = sess.run([d_loss, d_optim], feed_dict={
                t_real_image: b_real_images,
                # t_wrong_image : b_wrong_images,
                t_wrong_caption: b_wrong_caption,
                t_real_caption: b_real_caption,
                t_z: b_z})
            ## updates G
            errG, _ = sess.run([g_loss, g_optim], feed_dict={
                t_real_caption: b_real_caption,
                t_z: b_z})

            print("Epoch: [%2d/%2d] [%4d/%4d] time: %4.4fs, d_loss: %.8f, g_loss: %.8f, rnn_loss: %.8f"
                  % (epoch, n_epoch, step, n_batch_epoch, time.time() - step_time, errD, errG, errRNN))

        if (epoch + 1) % print_freq == 0:
            print(" ** Epoch %d took %fs" % (epoch, time.time() - start_time))
            img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
                t_real_caption: sample_sentence,
                t_z: sample_seed})

            # img_gen = threading_data(img_gen, prepro_img, mode='rescale')
            save_images(img_gen, [ni, ni], 'samples/step1_gan-cls_' + dataset + '/train_{:02d}.png'.format(epoch))

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

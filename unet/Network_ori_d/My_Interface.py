'''
Created on Sat Nov 30 22:06:00 2017

@author: zhangjun

modified by Wujian wjtcw@hotmail.com
2018.12.12
'''
import os.path as osp
import pprint
from datetime import datetime
#from networks.resnet20_dense4 import *
from networks.U_net import *
from input.input_pipeline import inpute, process
import scipy.io as scio
from libs.utils import *

GDL_LOSS_BOOL = 0
step_view = 500    #number of steps to show the result
step_save = 5000

def train():
    print('MODEL_DIR: ',FLAGS.MODEL_DIR)
    print('BETA: ', FLAGS.BETA)
    print('MAX_INTER: ', FLAGS.MAX_INTER)
    print('BATCH_SIZE: ', FLAGS.BATCH_SIZE)
    print('LR: ', FLAGS.LR)
    print('FILTER_DIM: ', FLAGS.FILTER_DIM)
    print('CROP_X: ', FLAGS.CROP_X)

    # choose GPU automatically
    GPU_ID = 7#get_gpu_id(total_gpu=6)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    # used for gdl_loss
    if GDL_LOSS_BOOL:
        gdl_total_loss = 0.0
    train_tot_loss = 0.0
    test_tot_loss = 0.0

    # create some output directories
    log_dir = osp.join('logs', FLAGS.MODEL_DIR)
    model_dir = osp.join('models', FLAGS.MODEL_DIR)

    if not osp.exists(log_dir):
        os.makedirs(log_dir)
    if not osp.exists(model_dir):
        os.makedirs(model_dir)

    X = tf.placeholder("float", [None, None, None, FLAGS.INPUT_C])
    Y = tf.placeholder("float", [None, None, None, FLAGS.OUTPUT_C])

    # adjust the learning rate
    global_step = tf.Variable(0)
    lr = tf.train.exponential_decay(
        learning_rate=FLAGS.LR,
        global_step=global_step,
        decay_steps=20000,
        decay_rate=0.5,
        staircase=True
    )

    # Net Models
    net = model(X)
    num_params = get_num_params()
    print (num_params)
    # loss
    mse, loss_T2 = losses(net, Y)
    # optim
#    train_op = tf.train.MomentumOptimizer(learning_rate, 0.9).minimize(mse)
    # train_op = tf.train.RMSPropOptimizer(lr, momentum = 0.9)\
    #     .minimize(
    #         train_loss,
    #         var_list = train_vars,
    #         global_step = global_step
    #     )
    train_op = tf.train.AdamOptimizer(learning_rate=lr, beta1=0.9, beta2=0.999, epsilon=1e-8).minimize(mse,global_step =global_step)

    # tensorboard
    summary_op_train = tf.summary.merge([
        tf.summary.scalar('train_loss', mse),
        tf.summary.image('T2', tf.expand_dims(net[:,:,:,0],-1)),
        tf.summary.image('T2', tf.expand_dims(Y[:,:,:,0],-1)),
        tf.summary.scalar('lr', lr)
    ])
    summary_op_test = tf.summary.merge([
        tf.summary.scalar('test_loss', mse),
        tf.summary.image('test_output', tf.expand_dims(net[:,:,:,0],-1)),
        tf.summary.image('test_labels', tf.expand_dims(Y[:,:,:,0],-1))
    ])
    summary_op_brain = tf.summary.merge([
        tf.summary.image('brain_T2', tf.expand_dims(net[:,:,:,0],-1)),
    ])
    train_writer = tf.summary.FileWriter(log_dir)
    train_pipeline = {
        'lr': lr,
        'train_loss': mse,
        'T2_loss': loss_T2,
        'opt': train_op,
        'log': summary_op_train,
    }
    test_pipeline = {
        'test_loss': mse,
        'log': summary_op_test
    }
    brain_pipeline = {
        'log': summary_op_brain
    }
    saver = tf.train.Saver(max_to_keep = FLAGS.MAX_TO_KEEP)

# load data
    X_train_pre, Y_train_pre = inpute(FLAGS.DATA_DIR, 'train')
    X_test_pre, Y_test_pre = inpute(FLAGS.DATA_DIR, 'test')
    X_brain, Y_brain = inpute(FLAGS.DATA_DIR, 'brain')
# init
    config = tf.ConfigProto()
    #config.gpu_options.per_process_gpu_memory_fraction = 0.8  # 程序最多只能占用指定gpu50%的显存
    config.gpu_options.allow_growth = True  # 程序按需申请内存0
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    start = 1
    if FLAGS.LOAD_MODEL:
        print(' [*] Reading checkpoints...')
        ckpt = tf.train.get_checkpoint_state(model_dir)

        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = osp.basename(ckpt.model_checkpoint_path)
            saver.restore(sess, osp.join(model_dir, ckpt_name))
            step = ckpt.model_checkpoint_path.split(
                '/')[-1].split('-')[-1]
            print('Loading success, step is %s' % step)
            start = int(step)
        else:
            print(' [*] Failed to find a checkpoint')
            start = 1

    print('******* start with %d *******' % start)
    idx = start
    temp_loss_other = [0, 0, 0, 0, 0]
    while True:
        X_train, Y_train = process(X_train_pre, Y_train_pre)
        feed_dict = {
            X: X_train,
            Y: Y_train,
            }
        train_op = sess.run(train_pipeline, feed_dict=feed_dict)
        temp_loss = train_op['train_loss']
        train_tot_loss = train_tot_loss+temp_loss
        temp_loss_other[1] = temp_loss_other[1]+train_op['T2_loss']
        if (idx) % step_view == 0:
            train_tot_loss /= step_view
            for j in range(20):
                X_test, Y_test = process(X_test_pre, Y_test_pre)
                test_op = sess.run(test_pipeline, feed_dict={
                    X: X_test,
                    Y: Y_test
                })
                brain_op = sess.run(brain_pipeline, feed_dict={
                    X: X_brain,
                    Y: Y_brain
                })

                temp_loss = test_op['test_loss']
                test_tot_loss = test_tot_loss + temp_loss
            test_tot_loss = test_tot_loss / 20
            print('step: %d, train loss %.8f, validate loss: %.8f, loss_T2: %.8f, loss_T2star: %.8f, B0_loss: %.8f,learning rate: %.8f' % (
            idx, train_tot_loss, test_tot_loss, temp_loss_other[1]/step_view, temp_loss_other[2]/step_view, temp_loss_other[3]/step_view, train_op['lr']))
            train_tot_loss = 0
            test_tot_loss = 0
            temp_loss_other = [0,0,0,0,0]
            train_writer.add_summary(train_op['log'], idx)
            train_writer.add_summary(test_op['log'], idx)
            train_writer.add_summary(brain_op['log'], idx)
        if (idx - step_save) % step_save == 0:
            checkpoint_path = osp.join(model_dir, 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=idx)
            print('**********  model%d saved  **********' % idx)
        if FLAGS.MAX_INTER <= idx:
            sess.close()
            break
        idx += 1

def test():
    FLAGS.BATCH_SIZE = 1
    FLAGS.MAX_INTER = 1
    FLAGS.IS_TRAIN = False
    FLAGS.LOAD_MODEL = True

    model_dir = osp.join('models', FLAGS.MODEL_DIR)
    output_dir = osp.join(
        'test_output', FLAGS.MODEL_DIR, 'brain', 'step' + str(FLAGS.STEP)
    )
    test_log_dir = osp.join(
        'test_log_dir', FLAGS.MODEL_DIR, 'brain', 'step' + str(FLAGS.STEP)
    )
    if not os.path.exists(model_dir):
        print('Checkpoint not found, please check you path to model')
        os._exit(0)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)
    if not osp.exists(test_log_dir):
        os.makedirs(test_log_dir)
# load data
    X_test_pre, Y_test_pre = inpute(FLAGS.DATA_DIR, 'brain')
    # Placeholders
    X = tf.placeholder(dtype=tf.float32, shape=[None, None, None, FLAGS.INPUT_C])
    Y = tf.placeholder(dtype=tf.float32, shape=[None, None, None, FLAGS.OUTPUT_C])

    # Building Network
    output = model(X)
    # loss
    mse, loss_T2 = losses(output, Y)
    saver = tf.train.Saver(max_to_keep=None)

    init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    GPU_ID = 7#get_gpu_id(gpu_num=FLAGS.NUM_GPUS)
    os.environ['CUDA_VISIBLE_DEVICES'] = str(GPU_ID)
    config = tf.ConfigProto(allow_soft_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(init)
    print(' [*] Reading checkpoints...')
    ckpt = tf.train.get_checkpoint_state(model_dir)

    if ckpt and ckpt.model_checkpoint_path:
        ckpt_name = osp.basename(ckpt.model_checkpoint_path)
        if FLAGS.STEP is not None:
            step = int(FLAGS.STEP)
            model_path = osp.join(
                model_dir, ckpt_name.split('-')[0] + '-' + str(step)
            )
        else:
            model_path = osp.join(model_dir, ckpt_name)
            step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]

        saver.restore(sess, model_path)
        print('Loading success, step is %s' % step)

    else:
        print(' [*] Failed to find a checkpoint')
    idx = 0
    try:
        while True:
            summary_op_test = tf.summary.merge([tf.summary.scalar('test_loss', mse)])
            test_pipeline = {
                'result': output,
                'test_loss': mse,
                'log': summary_op_test
            }
            # test
            X_test, Y_test = X_test_pre[idx, :, :, :], Y_test_pre[idx, :, :, :]
            X_test, Y_test = np.expand_dims(X_test,0), np.expand_dims(Y_test,0)
            test_op = sess.run(test_pipeline, feed_dict={
                X: X_test,
                Y: Y_test
            })

            print('%d: test loss %.8f' % (idx,  test_op['test_loss'] ))

            input_image = X_test[0]
            input_image = (input_image[:, :, 0] ** 2 + input_image[:, :, 1] ** 2) ** 0.5
            label = Y_test[0]
            label = label[:, :, :]
            output_image = test_op['result'][0, :, :, :]

            scio.savemat(
                osp.join(output_dir, 'result' + str(idx+1) + '.mat'),
                {'input_image': input_image,
                 'output_image': output_image,
                 'test_label': label
                 })
            idx += 1
            if (idx == X_test_pre.shape[0]):
                break

    except tf.errors.OutOfRangeError:
        print('Testing done!')


if __name__ == '__main__':
    print('Current time: %s' % datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('The network initialization with learning rate %f ...' % FLAGS.LR)
    pprint.pprint(FLAGS.__flags)
    #train()
    test()

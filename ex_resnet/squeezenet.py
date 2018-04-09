import ex_resnet.layers as layer
import tensorflow as tf

def inference(input,is_train):
    with tf.variable_scope("squeezenet") as scope:
        conv_1 = layer.conv(input, 64, 7, 2, "SAME", is_train, 'conv_1')
        pool_1 = layer.maxpool(conv_1, 3, 2, "SAME", 'pool_1')

        fire_2 = fire(pool_1,16,64,'fire_2',is_train)
        fire_3 = fire(fire_2, 16, 64, 'fire_3', is_train)
        fire_3 = fire_2 + fire_3
        fire_4 = fire(fire_3, 32, 128, 'fire_4', is_train)

        pool_2= layer.maxpool(fire_4, 3, 2, "SAME", 'pool_2')

        fire_5 = fire(pool_2,32,128,'fire_5',is_train)
        fire_5 = pool_2+fire_5
        fire_6 = fire(fire_5, 48, 192, 'fire_6', is_train)
        fire_7 = fire(fire_6, 48, 192, 'fire_7', is_train)

        fire_7 = fire_6 + fire_7
        fire_8 = fire(fire_7, 48, 192, 'fire_8', is_train)
        pool_3 = layer.maxpool(fire_8, 3, 2, "SAME", 'pool_3')
        fire_9 = fire(pool_3, 64, 256, 'fire_9', is_train)

        avg_pool = tf.nn.avg_pool(fire_9, ksize=[1, 2, 2, 1],
                              strides=[1, 1, 1, 1],
                              padding="VALID")
        rst = layer.fc(avg_pool, 10, 'logits')
    return rst

def fire(input,inCh,outCh,name,is_train):
    squeeze = layer.conv(input, inCh, 1, 1, "SAME", is_train, name+'squeeze')
    extend_1 = layer.conv(squeeze, outCh, 1, 1, "SAME", is_train, name + 'extend_1')
    extend_3 = layer.conv(squeeze, outCh, 3, 1, "SAME", is_train, name + 'extend_3')
    concat = tf.concat([extend_1,extend_3],axis=3)

    return concat
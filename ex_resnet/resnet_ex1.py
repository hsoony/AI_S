import ex_resnet.layers as layer
import tensorflow as tf

def resnet(input,is_train):
    conv_1 = layer.conv(input, 64, 7, 2, "SAME", is_train, 'conv_1')
    pool_1 = layer.maxpool(conv_1, 3, 2, "SAME", 'pool_1')

    res_out = pool_1
    for i in range(3):
        res_1_0 = layer.conv_preAct(res_out, 64, 3, 1, "SAME", is_train, 'res_1_0' + str(i))
        res_1_1 = layer.conv_preAct(res_1_0, 64, 3, 1, "SAME", is_train, 'res_1_1' + str(i))
        res_out = res_out + res_1_1

    res_2_0 = layer.conv_preAct(res_out, 128, 3, 2, "SAME", is_train, 'res_2_0' )
    res_2_1 = layer.conv_preAct(res_2_0, 128, 3, 1, "SAME", is_train, 'res_2_1' )

    pool_2 = layer.conv_preAct(res_out, 128, 1, 2, "SAME", is_train, 'pool_2')
    res_out = pool_2 + res_2_1

    for i in range(3):
        res_2_0 = layer.conv_preAct(res_out, 128, 3, 1, "SAME", is_train, 'res_2_0' + str(i))
        res_2_1 = layer.conv_preAct(res_2_0, 128, 3, 1, "SAME", is_train, 'res_2_1' + str(i))
        res_out = res_out + res_2_1

    avg_pool = tf.nn.avg_pool(res_out, ksize=[1, 4, 4, 1],
                          strides=[1, 1, 1, 1],
                          padding="VALID")
    rst = layer.fc(avg_pool, 10, 'logits')
    return rst

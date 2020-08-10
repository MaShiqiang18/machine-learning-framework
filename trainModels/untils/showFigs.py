# coding=gbk

import matplotlib.pyplot as plt

def metrice_loss_figs(model, lossFig_path, show_fig=False):
    '''
    ����ģ��ѵ��������lossֵ�͵÷����ţ��������������Ӷ�����ı仯
    :param model: ģ�ͣ���ѵ����ϣ�history���м�¼��
    :param mark: ��ǣ���������ͬһ��Ŀ��ͬһģ�ͣ���ͬ����ʽ��
    :param show_fig: �Ƿ���ʾͼƬ��Ĭ�ϻᱣ�棩
    :return:
    '''

    history_keys = model.history.history.keys()

    metrics_keys = [k for k in history_keys if k not in ['loss', 'val_loss']]
    if len(metrics_keys[0]) > len(metrics_keys[1]):
        metrics_val = metrics_keys[0]
        metrics_train = metrics_keys[1]
    else:
        metrics_val = metrics_keys[1]
        metrics_train = metrics_keys[0]

    y_values_metrics_val = model.history.history[metrics_val]
    y_values_metrics_train = model.history.history[metrics_train]
    x_epochs = range(len(y_values_metrics_val))

    plt.figure()
    plt.plot(x_epochs, y_values_metrics_val, label=metrics_val)
    plt.plot(x_epochs, y_values_metrics_train, label=metrics_train)
    plt.ylabel(metrics_train)
    plt.xlabel('epochs')
    plt.title('Training and validation ' + metrics_train)
    plt.legend(loc="upper right")
    plt.savefig(lossFig_path + '_' + metrics_train + '.jpg')
    if show_fig:
        plt.show()

    y_values_val_loss = model.history.history['val_loss']
    y_values_loss = model.history.history['loss']

    plt.figure()
    plt.plot(x_epochs, y_values_val_loss, label='val_loss')
    plt.plot(x_epochs, y_values_loss, label='loss')
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.title('Training and validation loss')
    plt.legend(loc="upper right")
    plt.savefig(lossFig_path + '_loss.jpg')
    if show_fig:
        plt.show()
# coding=gbk

import matplotlib.pyplot as plt

def metrice_loss_figs(model, lossFig_path, show_fig=False):
    '''
    绘制模型训练过程中loss值和得分随着，迭代次数的增加而引起的变化
    :param model: 模型（已训练完毕，history中有记录）
    :param mark: 标记（用于区别，同一项目，同一模型，不同处理方式）
    :param show_fig: 是否显示图片（默认会保存）
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
# coding=gbk

def time_format(time_cost: int):
    """
    �ķ�ʱ���ʽת��
    :param time_cost:
    :return:
    """
    min = 60
    hour = 60 * 60
    day = 60 * 60 * 24
    if not time_cost or time_cost < 0:
        return 'С��1��'
    elif time_cost < min:
        return '%s��' % time_cost
    elif time_cost < hour:
        return '%s��%s��' % (divmod(time_cost, min))
    elif time_cost < day:
        cost_hour, cost_min = divmod(time_cost, hour)
        return '%sСʱ%s' % (cost_hour, time_format(cost_min))
    else:
        cost_day, cost_hour = divmod(time_cost, day)
        return '%s��%s' % (cost_day, time_format(cost_hour))


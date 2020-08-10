# coding=gbk

def time_format(time_cost: int):
    """
    耗费时间格式转换
    :param time_cost:
    :return:
    """
    min = 60
    hour = 60 * 60
    day = 60 * 60 * 24
    if not time_cost or time_cost < 0:
        return '小于1秒'
    elif time_cost < min:
        return '%s秒' % time_cost
    elif time_cost < hour:
        return '%s分%s秒' % (divmod(time_cost, min))
    elif time_cost < day:
        cost_hour, cost_min = divmod(time_cost, hour)
        return '%s小时%s' % (cost_hour, time_format(cost_min))
    else:
        cost_day, cost_hour = divmod(time_cost, day)
        return '%s天%s' % (cost_day, time_format(cost_hour))


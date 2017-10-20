# -*- coding: utf-8 -*-
import datetime
import logbook
import pandas as pd

import biglearning.module2.common.interface as I
from biglearning.api import M
from biglearning.module2.common.data import Outputs, DataSource
from biglearning.module2.common.utils import smart_object


bigquant_cacheable = True

# 模块接口定义
bigquant_category = '机器学习'
bigquant_friendly_name = '滚动预测'
bigquant_doc_url = 'https://bigquant.com/docs/'
log = logbook.Logger(bigquant_friendly_name)


def generate_predict_inputs(predict_kwargs, model_param_name, data_param_name):
    models = smart_object(predict_kwargs[model_param_name])
    data = predict_kwargs[data_param_name]

    last_train_end = None
    for m in reversed(models):
        m['predict_start_date'] = (pd.to_datetime(m['end_date']) + datetime.timedelta(days=1)).strftime('%Y-%m-%d')
        m['predict_end_date'] = last_train_end
        last_train_end = m['end_date']

    log.info('滚动预测 ..')
    rollings = []
    for m in models:
        if not m or not m['output'] or not m['output'].model:
            continue

        if m['predict_end_date'] is not None:
            expr = '"%s" <= date <= "%s"' % (m['predict_start_date'], m['predict_end_date'])
        else:
            expr = '"%s" <= date' % (m['predict_start_date'])
        predict_data = M.filter.v3(input_data=data, expr=expr)
        if predict_data.row_count < 0:
            continue

        rollings.append({model_param_name: m['output'].model, data_param_name: predict_data.data})
        log.info('预测数据 [%s, %s], 共%s行，对应模型的训练数据 [%s, %s]' % (
            m['predict_start_date'], m['predict_end_date'], predict_data.row_count, m['start_date'], m['end_date']))
    return rollings


def merge_predict_outputs(rollings):
    outputs = Outputs(rollings=rollings)
    predictions = [r['output'].predictions.read_df() for r in rollings if r['output'].predictions is not None]
    if predictions:
        predictions = pd.concat(predictions, copy=False, ignore_index=True)
        outputs.start_date = predictions.date.min().strftime('%Y-%m-%d')
        outputs.end_date = predictions.date.max().strftime('%Y-%m-%d')
        outputs.instruments = sorted(set(predictions.instrument))
        outputs.predictions = DataSource.write_df(predictions)
    else:
        outputs.predictions = None
    return outputs


def bigquant_run(
    predict: I.port('预测，预测模块的延迟执行输出'),
    model_param_name: I.str('模型参数名，predict里用来接收模型的参数名，将从此参数获取模型输入')='model',
    data_param_name: I.str('数据参数名，predict里用来接收数据的参数名，将从此参数获取数据输入')='data') -> [
        I.port('预测结果', 'predictions')
    ]:
    '''
    滚动预测，通用滚动预测
    '''
    predict_kwargs = smart_object(predict)['kwargs']
    if model_param_name not in predict_kwargs:
        raise Exception('在预测模块参数中没有找到 %s' % model_param_name)
    rolling_run_inputs = generate_predict_inputs(
        predict_kwargs,
        model_param_name,
        data_param_name
    )

    rolling_run_outputs = M.rolling_run.v1(
        run=predict,
        input_list=rolling_run_inputs,
        param_name='{0}={0}|{1}={1}'.format(model_param_name, data_param_name)
    )

    return merge_predict_outputs(rolling_run_outputs.data.read_pickle())


def bigquant_postrun(outputs):
    return outputs


if __name__ == '__main__':
    # 测试代码
    pass

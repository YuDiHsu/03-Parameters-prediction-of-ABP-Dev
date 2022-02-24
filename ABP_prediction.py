import datetime
import itertools
import os
import dash
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import xlsxwriter
from dash import dcc
from dash import html
from plotly.subplots import make_subplots
# Common sklearn Model Helpers
from sklearn import model_selection
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
# Libraries for data modelling
from sklearn.linear_model import LogisticRegression
# sklearn modules for performance metrics
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split  # import 'train_test_split'
from sklearn.preprocessing import PolynomialFeatures


def _model_fitting(x, y_G, degree, model_type='Lasso'):
    x_ = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(x)
    model = ''
    if model_type == 'Linear':
        model = LinearRegression(n_jobs=2).fit(x_, y_G)
    if model_type == 'Lasso':
        model = Lasso(alpha=0.1).fit(x_, y_G)
    if model:
        print(f'r_sq: {model.score(x_, y_G)}')
    # # Elastic Net regression
    # model = ElasticNet(alpha=0.1, l1_ratio=0.5).fit(x_, y_G)
    return model


def _predict_value(model, array_predict, degree):
    x_pre = PolynomialFeatures(degree=degree, include_bias=False).fit_transform(
        array_predict)  # quadratic polynominal model
    y_pred = model.predict(x_pre)

    return x_pre, y_pred


def _get_possible_candidate_table(array_predict, y_pred, y_G_true, variation):
    y_candidate_boolean = (abs((y_pred - y_G_true) / y_G_true) <= variation)
    # y_candidate_boolean = (abs((y_pred - y_G_true) / y_pred) <= variation)
    y_candidate = y_pred[y_candidate_boolean]
    x_candidate = array_predict[y_candidate_boolean, :]

    return x_candidate, y_candidate


def _get_value(**kwargs):
    '''
    **kwargs: start, end = the range of number to split by step
              step: the value for each splitting partition
    '''

    kwargs.setdefault('start', 0)
    kwargs.setdefault('step', 0.5)
    _value = kwargs['start']
    value_list = []
    try:
        # get the flatten value of the range the user set
        while _value < kwargs['end']:
            if _value not in value_list:
                value_list.append(_value)
            _value += kwargs['step']

        if kwargs['end'] not in value_list:
            value_list.append(kwargs['end'])

        return value_list
    except ValueError as V:
        print(V)


def _num_after_point(x):
    s = str(x)
    if '.' not in s:
        return 0
    return len(s) - s.index('.') - 1


def _write_to_xlsx(df_list, sh_name, header_name_list, book_name):
    date = datetime.datetime.now().date()
    col_len_width = []
    workbook = xlsxwriter.Workbook(os.path.join('.', f"{book_name}_{date.strftime('%Y%m%d')}.xlsx"))
    for idx, data_ in enumerate(zip(df_list, sh_name)):
        header_format = workbook.add_format(
            {'align': 'center', 'valign': 'vcenter', 'size': 12, 'color': 'black', 'bold': 4})
        header = []
        for h in header_name_list[idx]:
            header.append({'header': h, "format": header_format})

        # print(sheet_name)
        sheet = workbook.add_worksheet(data_[1])
        if len(data_[0]):

            sheet.add_table(0, 0, len(data_[0]), len(data_[0][0]) - 1,
                            {'data': sorted(data_[0], key=lambda x: x[0], reverse=False), 'autofilter': True,
                             'columns': header})

            for m in range(len(data_[0]) + 1):
                sheet.set_row(m, 30, cell_format=header_format)

            col_len_width.append([len(j) for j in header_name_list[idx]])
            for n, l in enumerate(zip(col_len_width[idx], header_name_list[idx])):
                sheet.set_column(n, n, max(l[0], len(l[1])) * 4)
    workbook.close()


def _data_loading():
    df = pd.read_excel(os.path.join('.', 'ABP開發數據_20211119.xlsx'), index_col=None)
    x = df.loc[:, ('交聯劑比例', 'HA混合比例', '粒徑大小')]
    y_g1 = df.loc[:, ('G1', 'G2')]
    # y_g2 = df.loc[:, 'G2']
    return x, y_g1, df


def _cv(x_train, x_test, y_train_G, y_test_G, deg):
    x_poly = PolynomialFeatures(degree=deg, include_bias=False).fit_transform(x_train)
    # X_train, X_test, y_train, y_test = train_test_split(x_poly,
    #                                                     y_G,
    #                                                     test_size=0.25,
    #                                                     random_state=7,
    #                                                     )
    kfold = model_selection.KFold(n_splits=10, random_state=7, shuffle=True)
    modelCV = LogisticRegression(solver='liblinear', random_state=7)
    results = model_selection.cross_val_score(
        modelCV, x_poly, y_train_G, cv=kfold)
    # model_1 = model_fitting(X_train_G1, y_train_G1, deg)
    print("Accuracy_1: %.3f%% (%.3f%%)" % (results.mean() * 100.0, results.std() * 100.0))
    param_grid = {'C': np.arange(1e-03, 2, 0.01)}  # hyper-parameter list to fine-tune
    log_gs = GridSearchCV(LogisticRegression(solver='liblinear',  # setting GridSearchCV
                                             random_state=7),
                          n_jobs=3,
                          return_train_score=True,
                          param_grid=param_grid,
                          scoring='roc_auc',
                          cv=10)
    model = log_gs.fit(x_poly, y_train_G)
    log_opt = model.best_estimator_
    results = log_gs.cv_results_
    # scoring = 'roc_auc'
    print('=' * 20)
    print("best params: " + str(log_gs.best_estimator_))
    print("best params: " + str(log_gs.best_params_))
    print('best score:', log_gs.best_score_)
    print('=' * 20)
    print('Accuracy of Logistic Regression Classifier on test set: {:.2f}'.format(
        log_opt.score(x_test, y_test_G) * 100))
    return log_opt


def _processing(info_dict):
    # input data
    x, y_G, df_ = _data_loading()

    y_G1_true = info_dict['G1_true']  # true solution
    y_G1_variation = info_dict['G1_variation']  # +-10% error is accepted
    y_G2_true = info_dict['G2_true']  # true solution
    y_G2_variation = info_dict['G2_variation']
    deg = info_dict['degree']
    x_pre_list = []
    for n_ in range(info_dict['variable_num']):
        x_pre_range = _get_value(start=info_dict[f'x{n_ + 1}_variable']['start'],
                                 end=info_dict[f'x{n_ + 1}_variable']['end'],
                                 step=info_dict[f'x{n_ + 1}_variable']['step'])
        x_pre_list.append(x_pre_range)
    # # create the factorial array to predict
    array_predict = np.array(list(itertools.product(*x_pre_list)))


    # for y_G1
    X_train, X_test, y_train_G, y_test_G = train_test_split(x, y_G, test_size=0.1, random_state=7)
    y_train_G1 = y_train_G.loc[:, 'G1']
    y_train_G2 = y_train_G.loc[:, 'G2']
    y_test_G1 = y_test_G.loc[:, 'G1']
    y_test_G2 = y_test_G.loc[:, 'G2']

    # log_opt_1 = _cv(X_train, X_test, y_train_G1, y_test_G1, deg)
    # log_opt_2 = _cv(X_train, X_test, y_train_G2, y_test_G2, deg)
    # print('='*100)
    model_1 = _model_fitting(X_train, y_train_G1, deg, 'Lasso')
    # print("Accuracy_1: %.3f%% (%.3f%%)" % (model_1.mean() * 100.0, model_1.std() * 100.0))
    x_val_G1, y_val_G1 = _predict_value(model_1, X_test, deg)
    x_pred_G1, y_pred_G1 = _predict_value(model_1, array_predict, deg)
    x_candidate_G1, y_candidate_G1 = _get_possible_candidate_table(array_predict, y_pred_G1, y_G1_true, y_G1_variation)

    model_2 = _model_fitting(X_train, y_train_G2, deg, 'Lasso')
    # print("Accuracy_1: %.3f%% (%.3f%%)" % (model_2.mean() * 100.0, model_2.std() * 100.0))
    x_pred_G2, y_pred_G2 = _predict_value(model_2, array_predict, deg)
    x_val_G2, y_val_G2 = _predict_value(model_2, X_test, deg)

    df_val = pd.DataFrame(np.hstack([X_test.values, y_test_G1.values.reshape(len(X_test), 1), y_test_G2.values.reshape(len(X_test), 1),
                                     y_val_G1.reshape(len(X_test), 1),  y_val_G2.reshape(len(X_test), 1)]),
                          columns=list(df_.columns)+['y_val_G1', 'y_val_G2'])
    df_val.to_excel('df_val.xlsx', index=False)

    x_candidate_G2, y_candidate_G2 = _get_possible_candidate_table(array_predict, y_pred_G2, y_G2_true, y_G2_variation)

    final_x_pred = x_candidate_G2.tolist() + x_candidate_G1.tolist()

    x_1_list = []
    x_2_list = []
    x_3_list = []
    pred_dict_list = []
    for m in x_candidate_G1.tolist():
        for n in x_candidate_G2.tolist():
            pred_dict = {}
            if m == n:
                pred_dict['交聯劑比例'] = round(m[0], 2)
                pred_dict['HA混合比例'] = round(m[1], 2)
                pred_dict['粒徑大小'] = round(m[-1], 2)
                x_1_list.append(round(m[0], 2))
                x_2_list.append(round(m[1], 2))
                x_3_list.append(round(m[-1], 2))
                pred_G1 = round(y_candidate_G1.tolist()[x_candidate_G1.tolist().index(m)], 2)
                pred_G2 = round(y_candidate_G2.tolist()[x_candidate_G2.tolist().index(n)], 2)
                pred_dict['G1'] = pred_G1
                pred_dict['G2'] = pred_G2
                pred_dict_list.append(pred_dict)
                print(f'交聯劑比例: {round(m[0], 2)}\nHA混合比例: {round(m[1], 2)}\n粒徑大小: {round(m[-1], 2)}')
                print(f'G1: {pred_G1}')
                print(f'G2: {pred_G2}')
                print('-' * 100)
                # print('\n')
    print(f"交聯劑比例: {sorted(list(set(x_1_list)))}")
    print(f"HA混合比例: {sorted(list(set(x_2_list)))}")
    print(f"粒徑大小: {sorted(list(set(x_3_list)))}")
    df = pd.DataFrame(pred_dict_list)

    _write_to_xlsx([df.values.tolist()],
                   ['Ingredient ratio prediction'],
                   [list(df.columns)],
                   'Ingredient_ratio_prediction')

    return df.fillna(0)


def _plot_it(df):
    # df = px.data.election()
    # fig = px.scatter_ternary(df, a="交聯劑比例", b="HA混合比例", c="粒徑大小", hover_name="G1",
    #                          color="G1", size="G1", size_max=15,
    #                          color_discrete_map={"交聯劑比例": "blue", "HA混合比例": "green", "粒徑大小": "red"})
    fig = make_subplots(
        rows=2, cols=1,
        vertical_spacing=0.03,
        specs=[[{"type": "table"}],
               [{"type": "scatter"}],
               # [{"type": "scatter"}]
               # [{"secondary_y": True}]
               ]
    )

    # fig.add_trace(
    #     go.Scatter(
    #         x=df["HA混合比例"],
    #         y=df["粒徑大小"],
    #
    #         mode="markers",
    #         name="交聯劑比例 0.15",
    #         hovertemplate='<b>HA混合比例</b>: %{x}' +
    #                       '<br><b>粒徑大小</b>: %{y}<br>' +
    #                       '%{text}',
    #         text=[f'<b>G1</b>: {g1}<br><b>G2</b>: {g2}' for g1, g2 in zip(df['G1'], df['G2'])],
    #
    #     ),
    #     row=2, col=1
    # )

    fig.add_trace(
        go.Scatter(
            x=df["G1"],
            y=df["G2"],
            mode="markers",
            name="族群趨勢",
            hovertemplate='%{text}<br>' +
                          '<b>G1</b>: %{x}' +
                          '<br><b>G2</b>: %{y}<br>',
            text=[f'<b>HA混合比例</b>: {h}<br><b>粒徑大小</b>: {p}<br><b>交聯劑比例</b>: {r}'
                  for h, p, r in zip(df['HA混合比例'], df['粒徑大小'], df['交聯劑比例'])],
        ),
        row=2, col=1
    )

    fig.add_trace(
        go.Table(
            header=dict(
                values=["交聯劑比例", "HA混合比例", "粒徑大小", "G1", "G2"],
                font=dict(size=10),
                align="left"
            ),
            cells=dict(
                values=[df[k].tolist() for k in list(df.columns)],
                align="left")
        ),
        row=1, col=1
    )
    fig.update_layout(
        height=800,
        showlegend=False,
        title_text="成分比例模型預測",
    )

    app = dash.Dash()
    app.layout = html.Div([
        dcc.Graph(figure=fig)
    ])

    app.run_server(debug=True, use_reloader=False)
    # fig.show()


if __name__ == '__main__':
    info_ = dict(G1_true=261, G1_variation=0.2,
                 G2_true=61, G2_variation=0.2,
                 x1_variable=dict(start=0.2, end=1, step=0.05),  # 交聯劑比例
                 x2_variable=dict(start=1, end=4, step=0.1),  # HA混合比例
                 x3_variable=dict(start=400, end=900, step=10),  # 粒徑大小
                 variable_num=3,
                 degree=3,
                 )

    df = _processing(info_)
    # _plot_it(df)
    # _ = px.data.election()
    # print(_)

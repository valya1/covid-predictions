import random
import time
import operator
import numpy as np
import os
import pandas as pd
import pickle
import gzip
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
from catboost import Pool, CatBoostRegressor

ROOT_PATH = os.path.dirname(os.path.realpath(__file__ + '/../')) + '/'
FEATURES_PATH = ROOT_PATH + 'features/'
INPUT_PATH = ROOT_PATH + 'input/'
SUBM_PATH = ROOT_PATH + 'subm/'
SUBM_PATH_DETAILED = SUBM_PATH + 'detailed/'
MODELS_PATH = ROOT_PATH + 'models/'

USE_LOG = 1

# epochs count
REQUIRED_ITERATIONS = 1

# Days to predict in future
DAYS_TO_PREDICT = 7

# for real validation, None if not needed
STEP_BACK = 7


def save_in_file(arr, file_name):
    pickle.dump(arr, gzip.open(file_name, 'wb+', compresslevel=3))


def days_from_first_case(table):
    from datetime import datetime
    type = 'confirmed'
    first_case = pd.read_csv(FEATURES_PATH + 'first_date_for_rus_{}.csv'.format(type))
    first_case = first_case[['name', 'name2', 'date']].values
    fc = dict()
    for i in range(first_case.shape[0]):
        fc[first_case[i, 1]] = first_case[i, 2]

    delta = []
    for index, row in table.iterrows():
        dt1 = datetime.strptime(row['date'], '%Y.%m.%d')
        dt2 = datetime.strptime(fc[row['name1']], '%Y.%m.%d')
        diff = dt1 - dt2
        delta.append(diff.days)

    table['days_from_first_case_{}'.format(type)] = delta
    return table


def get_importance(gbm, data, features):
    importance = gbm.get_feature_importance(data, thread_count=-1, fstr_type='FeatureImportance')
    imp = dict()
    for i, f in enumerate(features):
        imp[f] = importance[i]
    res = sort_dict_by_values(imp)
    return res


def get_kfold_split_v2(folds_number, train, random_state):
    train_index = list(range(len(train)))
    folds = KFold(n_splits=folds_number, shuffle=True, random_state=random_state)
    ret = []
    for n_fold, (trn_idx, val_idx) in enumerate(folds.split(train_index)):
        ret.append([trn_idx, val_idx])
    return ret


def sort_dict_by_values(a, reverse=True):
    sorted_x = sorted(a.items(), key=operator.itemgetter(1), reverse=reverse)
    return sorted_x


def metrics(true, pred):
    s = (pred + 1) / (true + 1)
    error = np.absolute(np.log10(s)).mean()
    return error


def decrease_table_for_last_date(table):
    unique_dates = sorted(list(table['date'].unique()))
    last_date = unique_dates[-1]
    table = table[table['date'] == last_date]
    return table


# def add_area_and_density(table):
#     s = pd.read_csv(INPUT_PATH + 'additional/data_rus_regions_upd.csv')
#     s['density'] = s['population_2020'].values / s['area'].values
#     table = table.merge(s[['name2', 'area', 'density']], on='name2', how='left')
#     return table


def remove_latest_days(table, days):
    dates = sorted(table['date'].unique())
    dates_valid = dates[:-days]
    table = table[table['date'].isin(dates_valid)]
    return table


def latest_days_only(table, days):
    dates = sorted(table['date'].unique())
    dates_valid = dates[-days:]
    table = table[table['date'].isin(dates_valid)]
    return table


# population, urban population, rural population
# def add_special_additional_features(table):
#     s = pd.read_csv(INPUT_PATH + 'additional/population_rus.csv')
#     s['name1'] = s['name']
#     table = table.merge(s[['name1', 'population', 'population_urban', 'population_rural']], on='name1', how='left')
#     table['population'] = table['population'].fillna(-1)
#     table['population_urban'] = table['population_urban'].fillna(-1)
#     table['population_rural'] = table['population_rural'].fillna(-1)

# return table


def add_weekday(table, day):
    from datetime import datetime, timedelta

    weekday = []
    dates = table['date'].values
    for d in dates:
        datetime_object = datetime.strptime(d, '%Y.%m.%d')
        datetime_object += timedelta(days=day)
        w = datetime_object.weekday()
        weekday.append(w)
    table['weekday'] = weekday
    return table


def get_params():
    return {
        'target': 'target',
        'id': 'id',
        'metric': 'mean_squared_error',
        'metric_function': mean_squared_error
    }


def read_input_data(day, step_back_days=None):
    train = pd.read_csv(FEATURES_PATH + 'features_rus_predict_{}_day_{}.csv'.format(type, day))
    test = pd.read_csv(FEATURES_PATH + 'features_rus_predict_{}_day_{}.csv'.format(type, 0))
    print('Initial train: {} Initial test: {}'.format(len(train), len(test)))

    if step_back_days is not None:
        train = remove_latest_days(train, step_back_days)
        test = latest_days_only(test, step_back_days)

    # Remove zero target (must be faster training)
    l1 = len(train)
    train = train[train['target'] > 0]
    l2 = len(train)
    train.reset_index(drop=True, inplace=True)
    print('Removed zero target. Reduction {} -> {}'.format(l1, l2))

    # the last day for test is being taken only
    # test = decrease_table_for_last_date(test)
    # print('Updated train: {} Updated test: {}'.format(len(train), len(test)))

    # add info about days of week
    train = add_weekday(train, day)
    test = add_weekday(test, day)

    # days from first infected
    train = days_from_first_case(train)
    test = days_from_first_case(test)

    features = list(test.columns.values)
    features.remove('name1')
    features.remove('name2')
    features.remove('date')
    features.remove('target')
    print(len(train), len(test))

    return train, test, features


def create_catboost_model(train, features, params, day):
    import catboost as catb
    print('Catboost version: {}'.format(catb.__version__))
    target_name = params['target']
    start_time = time.time()
    if USE_LOG:
        train[target_name] = np.log10(train[target_name] + 1)

    train[target_name] = train[target_name]

    required_iterations = REQUIRED_ITERATIONS
    seed = 1921
    overall_train_predictions = np.zeros((len(train),), dtype=np.float32)
    overall_importance = dict()

    model_list = []
    for iter1 in range(required_iterations):
        num_folds = random.randint(4, 5)
        learning_rate = random.choice([0.01, 0.03, 0.05])
        depth = random.choice([4, 5, 6])

        ret = get_kfold_split_v2(num_folds, train, seed + iter1)
        full_single_preds = np.zeros((len(train),), dtype=np.float32)
        fold_num = 0

        # обучение происходит с KFold-валидацией, в итоге получаем K классификаторов
        for train_index, valid_index in ret:
            fold_num += 1
            X_train = train.loc[train_index].copy()
            X_valid = train.loc[valid_index].copy()
            y_train = X_train[target_name]
            y_valid = X_valid[target_name]
            early_stop = 100
            model = CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=10000,
                learning_rate=learning_rate,
                depth=depth,
                bootstrap_type='Bayesian',
                task_type='CPU',
                devices='0',
                metric_period=1,
                od_type='Iter',
                od_wait=early_stop,
                random_seed=17,
                l2_leaf_reg=3,
                allow_writing_files=False
            )

            dtrain = Pool(X_train[features].values, label=y_train)
            dvalid = Pool(X_valid[features].values, label=y_valid)

            gbm = model.fit(dtrain, eval_set=dvalid, use_best_model=True, verbose=0)
            model_list.append(gbm)

            imp = get_importance(gbm, dvalid, features)
            print('Importance: {}'.format(imp[:20]))
            for i in imp:
                if i[0] in overall_importance:
                    overall_importance[i[0]] += i[1] / num_folds
                else:
                    overall_importance[i[0]] = i[1] / num_folds

            pred = gbm.predict(X_valid[features].values)
            full_single_preds[valid_index] += pred.copy()

        train_tmp = train.copy()
        train_tmp['pred'] = full_single_preds
        train_tmp = decrease_table_for_last_date(train_tmp)

        score = metrics(train_tmp[target_name].values, train_tmp['pred'].values)
        overall_train_predictions += full_single_preds
        print('Score iter {}: {:.6f} Time: {:.2f} sec'.format(iter1, score, time.time() - start_time))

    overall_train_predictions /= required_iterations
    for el in overall_importance:
        overall_importance[el] /= required_iterations
    imp = sort_dict_by_values(overall_importance)
    names = []
    values = []
    print('Total importance count: {}'.format(len(imp)))
    output_features = 100
    for i in range(min(output_features, len(imp))):
        print('{}: {:.6f}'.format(imp[i][0], imp[i][1]))
        names.append(imp[i][0])
        values.append(imp[i][1])

    if USE_LOG:
        train[target_name] = np.power(10, train[target_name]) - 1
        overall_train_predictions = np.power(10, overall_train_predictions) - 1

    overall_train_predictions[overall_train_predictions < 0] = 0
    train_tmp = train.copy()
    train_tmp['pred'] = overall_train_predictions

    # We now that value must be equal or higher
    count_less = (train_tmp['pred'] < train_tmp['case_day_minus_0']).astype(np.int32).sum()
    if count_less > 0:
        print('Values less than needed: {} ({:.4f} %)'.format(count_less, 100 * count_less / len(train_tmp)))
    train_tmp['pred'] = np.maximum(train_tmp['pred'], train_tmp['case_day_minus_0'])

    train_tmp = decrease_table_for_last_date(train_tmp)
    score = metrics(train[target_name].values, overall_train_predictions)
    print('Total score day {} full: {:.6f}'.format(day, score))
    score = metrics(train_tmp[target_name].values, train_tmp['pred'].values)
    print('Total score day {} last date only: {:.6f}'.format(day, score))

    return overall_train_predictions, score, model_list, imp


def predict_with_catboost_model(test, features, model_list):
    full_preds = []
    for m in model_list:
        preds = m.predict(test[features].values)
        full_preds.append(preds)
    preds = np.array(full_preds).mean(axis=0)

    if USE_LOG:
        preds = np.power(10, preds) - 1

    preds[preds < 0] = 0
    return preds


if __name__ == '__main__':
    start_time = time.time()
    prediction_type = 'rus_regions'
    gbm_type = 'CatB'
    params = get_params()
    target = params['target']
    id = params['id']
    metric = params['metric']
    limit_date = DAYS_TO_PREDICT

    all_scores = dict()
    alldays_preds_train = dict()
    alldays_preds_test = dict()
    type = 'confirmed'
    alldays_preds_train[type] = []
    alldays_preds_test[type] = []
    day = 1
    # for day in range(1, limit_date + 1):
    train, test, features = read_input_data(day, step_back_days=STEP_BACK)
    print('Features: [{}] {}'.format(len(features), features))
    print('Test date: {}'.format(sorted(test['date'].unique())))
    overall_train_predictions, score, model_list, importance = create_catboost_model(train, features, params, day)
    prefix = '{}_day_{}_{}_{:.6f}'.format(gbm_type, day, len(model_list), score)
    save_in_file((score, model_list, importance, overall_train_predictions), MODELS_PATH + prefix + '.pklz')
    all_scores[(type, day)] = score
    train['pred'] = overall_train_predictions
    train['pred'] = np.maximum(train['pred'], train['case_day_minus_0'])
    train[['name1', 'name2', 'date', 'target', 'pred']].to_csv(SUBM_PATH_DETAILED + prefix + '_train.csv',
                                                               index=False, float_format='%.8f')
    train_tmp = decrease_table_for_last_date(train)
    alldays_preds_train[type].append(train_tmp[['name1', 'name2', 'date', 'target', 'pred']].copy())

    overall_test_predictions = predict_with_catboost_model(test, features, model_list)
    test['pred'] = overall_test_predictions

    # Value must be equal or higher
    count_less = (test['pred'] < test['case_day_minus_0']).astype(np.int32).sum()
    if count_less > 0:
        print('Values less than needed for test: {} ({:.4f} %)'.format(count_less, 100 * count_less / len(test)))
    test['pred'] = np.maximum(test['pred'], test['case_day_minus_0'])

    test['shift_day'] = day
    test[['name1', 'name2', 'date', 'pred']].to_csv(SUBM_PATH_DETAILED + prefix + '_test.csv', index=False,
                                                    float_format='%.8f')
    alldays_preds_test[type].append(test[['name1', 'name2', 'date', 'shift_day', 'target', 'pred']].copy())
    print('----------------------------------')

train = pd.concat(alldays_preds_train[type], axis=0)
score = metrics(train['target'].values, train['pred'].values)
all_scores[(type, 'full')] = score
print('Total score {} for all days: {:.6f}'.format(type, score))
prefix = '{}_{}_all_days_{}_{:.6f}'.format(gbm_type, type, len(model_list), score)
train.to_csv(SUBM_PATH + '{}_train.csv'.format(prefix), index=False)
test = pd.concat(alldays_preds_test[type], axis=0)
test.to_csv(SUBM_PATH + '{}_test.csv'.format(prefix), index=False)

prefix = 'rus_regions'
train.to_csv(SUBM_PATH + '{}_train.csv'.format(prefix), index=False)
test.to_csv(SUBM_PATH + '{}_test.csv'.format(prefix), index=False)

# for day in range(1, limit_date + 1):
print('Type: {} Day: {} Score: {:.6f}'.format(type, day, all_scores[(type, day)]))

print('Total score {} for all days: {:.6f}'.format(type, all_scores[(type, 'full')]))

print("Elapsed time overall: {:.2f} seconds".format((time.time() - start_time)))

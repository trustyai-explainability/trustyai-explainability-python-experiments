import argparse

import pandas as pd
import numpy as np
from pt_tabular import TabularFICO
import lime.lime_tabular
from sklearn.model_selection import train_test_split

import trustyai

classpath = [
        "/home/tteofili/dev/kogito-apps/explainability/explainability-core/target/*",
        "../python-trustyai/dep/org/slf4j/slf4j-api/1.7.30/slf4j-api-1.7.30.jar",
        "../python-trustyai/dep/org/apache/commons/commons-lang3/3.12.0/commons-lang3-3.12.0.jar",
        "../python-trustyai/dep/org/apache/commons/commons-math3/3.6.1/commons-math3-3.6.1.jar",
        "../python-trustyai/dep/org/optaplanner/optaplanner-core/8.17.0.Final/optaplanner-core-8.17.0.Final.jar",
    ]

trustyai.init(path=classpath)

from trustyai.model import feature, output
from org.kie.kogito.explainability.model import PredictionInput, PredictionOutput, EncodingParams
from trustyai.model import simple_prediction, Model
from trustyai.explainers import LimeExplainer
from trustyai.metrics import ExplainabilityMetrics
from org.kie.kogito.explainability.model import FeatureImportance


def predict(inputs):
    values = [_feature.value.as_obj() for _feature in inputs[0].features]
    result = predict_proba(np.array([values]))
    false_prob, true_prob = result[0]
    if false_prob > true_prob:
        _prediction = (False, false_prob)
    else:
        _prediction = (True, true_prob)
    _output = output(name="RiskPerformance", dtype="bool", value=_prediction[0], score=_prediction[1])
    return [PredictionOutput([_output])]


def make_feature(name, _value):
    if isinstance(_value, bool):
        return feature(name=name, dtype="bool", value=_value)
    else:
        return feature(name=name, dtype="number", value=_value)


def eval_impact(explainer, model, test_df:pd.DataFrame, decision:str, k:int):
    mean_is = 0
    for idx in np.arange(len(test_df)):
        sample = test_df.iloc[idx].to_dict()
        features = [make_feature(k,v) for k,v in sample.items()]
        sample_input = PredictionInput(features)
        prediction = model.predictAsync([sample_input]).get()
        prediction_obj = simple_prediction(input_features=features, outputs=prediction[0].outputs)
        explanation = explainer.explain(prediction_obj, model)
        saliency = explanation._saliencies[decision]
        top_features_t = saliency.getTopFeatures(k)
        impact = ExplainabilityMetrics.impactScore(cb_model, prediction_obj, top_features_t)
        mean_is += impact
    return mean_is/len(test_df)


def to_fis(exp, k, features):
    top_k = exp.as_list()[:k]
    fis = []
    for e in top_k:
        name = ''
        fn = e[0].split(' ')
        if len(fn) == 3:
            name = fn[0].strip()
        elif len(fn) == 5:
            name = fn[2].strip()
        else:
            fn = e[0].split('=')
            name = fn[0].strip()
        f = [x for x in features if x.getName() == name][0]
        imp = e[1]
        fi = FeatureImportance(f, imp)
        fis.append(fi)
    return fis


def eval_impact_original(explainer, predict_proba, test_df:pd.DataFrame, k:int):
    mean_is = 0
    cb_model = Model(predict)
    for idx in np.arange(len(test_df)):
        sample = test_df.iloc[idx]
        exp = explainer.explain_instance(sample, predict_proba)
        features = [make_feature(k,v) for k,v in sample.items()]
        sample_input = PredictionInput(features)
        outputs = cb_model.predictAsync([sample_input]).get()
        prediction_obj = simple_prediction(input_features=features, outputs=outputs[0].outputs)
        top_features_o = to_fis(exp, k, features)
        impact = ExplainabilityMetrics.impactScore(cb_model, prediction_obj, top_features_o)
        mean_is += impact
    return mean_is/len(test_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run saliency experiments.')
    parser.add_argument('--model_path', metavar='m', type=str, help='the path to the saved model',
                        default='saved_models/fico_tabular_basic')
    parser.add_argument('--dataset_path', metavar='d', type=str, help='the path to the FICO dataset',
                        default='datasets/FICO/heloc_dataset_v1.csv')
    parser.add_argument('--top_k', metavar='k', type=int, default=1,
                        help='no. of salient features to drop for impact-score eval')
    parser.add_argument('--samples', metavar='s', type=int, default=-1,
                        help='no. of samples to eval')

    args = parser.parse_args()

    tab_model = TabularFICO()
    tab_model.load(args.model_path)

    predict_proba = lambda x : tab_model.predict(x)[['Bad_probability', 'Good_probability']].values

    data_df = pd.read_csv(args.dataset_path)
    train_df, test_df = train_test_split(data_df, test_size=0.2)
    train_df, valid_df = train_test_split(train_df, test_size=0.1)
    unl_test_df = test_df.drop(['RiskPerformance'],axis=1)

    cat_indices = []
    for col in tab_model.categorical_cols:
        cat_indices = cat_indices + [train_df.columns.get_loc(col)]

    # original LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(train_df.drop(['RiskPerformance'],axis=1).values,
                                                       feature_names=tab_model.continuous_cols + tab_model.categorical_cols,
                                                       categorical_features=cat_indices,
                                                       class_names=['Bad', 'Good'])

    cb_model = Model(predict)

    tlime_explainer = LimeExplainer(samples=5000, perturbations=10, seed=0, normalise_weights=False)

    t_is = []
    o_is = []
    for k in np.arange(1, args.top_k):
        o_is.append(eval_impact_original(explainer, predict_proba, unl_test_df[:args.samples], k))
        print(f'impact-score@{k} for original-lime-explainer: {o_is}')

        t_is.append(eval_impact(tlime_explainer, cb_model, unl_test_df[:args.samples], 'RiskPerformance', k))
        print(f'impact-score@{k} for trustyai-lime-explainer: {t_is}')

    print(f'Original-LIME:{o_is}')
    print(f'trustyAI-LIME:{t_is}')

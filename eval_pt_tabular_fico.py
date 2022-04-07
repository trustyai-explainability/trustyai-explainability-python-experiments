import argparse

import pandas as pd
import numpy as np
from pt_tabular import TabularFICO
import lime.lime_tabular
import shap
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
from trustyai.explainers import LimeExplainer, SHAPExplainer, _ShapConfig, _ShapKernelExplainer
from trustyai.metrics import ExplainabilityMetrics
from org.kie.kogito.explainability.model import FeatureImportance


def predict(inputs):
    outputs = []
    values = np.zeros((len(inputs),len(inputs[0].features)))
    for idx in range(len(inputs)):
        values[idx] = np.array([_feature.value.as_obj() for _feature in inputs[idx].features])
    results = predict_proba(values)
    for result in results:
        false_prob, true_prob = result
        if false_prob > true_prob:
            _prediction = (False, false_prob)
        else:
            _prediction = (True, true_prob)
        _output = output(name="RiskPerformance", dtype="bool", value=_prediction[0], score=_prediction[1])
        po = PredictionOutput([_output])
        outputs.append(po)
    return outputs


def make_feature(name, _value):
    if isinstance(_value, bool):
        return feature(name=name, dtype="bool", value=_value)
    else:
        return feature(name=name, dtype="number", value=_value)


def eval_lime_impact(explainer, model, test_df:pd.DataFrame, decision:str, k:int):
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


def eval_shap_impact(shap_explainer, model, predictions, k:int):
    mean_is = 0
    for prediction in predictions:
        explanation = shap_explainer.explain(prediction, model)
        saliency = explanation.getSaliencies()[0]
        top_features_t = saliency.getTopFeatures(k)
        impact = ExplainabilityMetrics.impactScore(cb_model, prediction, top_features_t)
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


def to_fis_shap(shap_values, k, features):
    idx = 0
    fis = []
    for f in features:
        fi = FeatureImportance(f, shap_values[idx])
        idx += 1
        fis.append(fi)
    return fis


def eval_lime_impact_original(explainer, predict_proba, test_df:pd.DataFrame, k:int):
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


def eval_shap_impact_original(shap_explainer, predict_proba, test_df:pd.DataFrame, k:int):
    mean_is = 0
    cb_model = Model(predict)
    for idx in np.arange(len(test_df)):
        sample = test_df.iloc[idx]
        shap_values = shap_explainer.shap_values(sample)
        features = [make_feature(k,v) for k,v in sample.items()]
        sample_input = PredictionInput(features)
        outputs = cb_model.predictAsync([sample_input]).get()
        prediction_obj = simple_prediction(input_features=features, outputs=outputs[0].outputs)
        top_features_o = to_fis_shap(shap_values, k, features)
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
    parser.add_argument('--samples', metavar='n', type=int, default=-1,
                        help='no. of samples to eval')
    parser.add_argument('--shap_bg_size', metavar='b', type=int, default=100,
                        help='no. of samples in SHAP background')
    parser.add_argument('--lime', metavar='l', type=bool, default=False,
                        help='whether to run LIME experiments')
    parser.add_argument('--shap', metavar='s', type=bool, default=False,
                        help='whether to run SHAP experiments')

    args = parser.parse_args()

    tab_model = TabularFICO()
    tab_model.load(args.model_path)

    predict_proba = lambda x : tab_model.predict(x)[['Bad_probability', 'Good_probability']].values
    cb_model = Model(predict)

    data_df = pd.read_csv(args.dataset_path)
    train_df, test_df = train_test_split(data_df, test_size=0.2)
    train_df, valid_df = train_test_split(train_df, test_size=0.1)
    unl_test_df = test_df.drop(['RiskPerformance'],axis=1)
    unl_train_df = train_df.drop(['RiskPerformance'],axis=1)

    if args.lime:
        print('running LIME experiments')
        cat_indices = []
        for col in tab_model.categorical_cols:
            cat_indices = cat_indices + [train_df.columns.get_loc(col)]

        # original LIME explainer
        lime_explainer = lime.lime_tabular.LimeTabularExplainer(train_df.drop(['RiskPerformance'],axis=1).values,
                                                           feature_names=tab_model.continuous_cols + tab_model.categorical_cols,
                                                           categorical_features=cat_indices,
                                                           class_names=['Bad', 'Good'])


        # trustyAI LIME explainer
        tlime_explainer = LimeExplainer(samples=5000, perturbations=10, seed=0, normalise_weights=False)

        t_is = []
        o_is = []
        for k in np.arange(1, args.top_k):
            o_is.append(eval_lime_impact_original(lime_explainer, predict_proba, unl_test_df[:args.samples], k))
            print(f'impact-score@{k} for original-lime-explainer: {o_is}')

            t_is.append(eval_lime_impact(tlime_explainer, cb_model, unl_test_df[:args.samples], 'RiskPerformance', k))
            print(f'impact-score@{k} for trustyai-lime-explainer: {t_is}')

        print(f'Original-LIME:{o_is}')
        print(f'trustyAI-LIME:{t_is}')

    if args.shap:
        print('running SHAP experiments')
        shap_bg_size = args.shap_bg_size

        # original SHAP explainer
        oshap_explainer = shap.KernelExplainer(predict_proba, unl_train_df[:shap_bg_size])

        background = []
        for idx in np.arange(len(unl_train_df)):
            sample = unl_train_df.iloc[idx]
            features = [make_feature(k, v) for k, v in sample.items()]
            sample_input = PredictionInput(features)
            background.append(sample_input)

        test_data = []
        for idx in np.arange(len(unl_test_df)):
            sample = unl_test_df.iloc[idx]
            features = [make_feature(k, v) for k, v in sample.items()]
            sample_input = PredictionInput(features)
            test_data.append(sample_input)

        prediction_outputs = cb_model.predictAsync(test_data).get()

        predictions = []
        for i in np.arange(len(test_data)):
            features = test_data[i].features
            outputs = prediction_outputs[i].outputs
            pred = simple_prediction(input_features=features, outputs=outputs)
            predictions.append(pred)

        # trustyAI SHAP explainer
        tshap_explainer = SHAPExplainer(background=background[:shap_bg_size])
        tshap_explainer._config = _ShapConfig.builder().withBackground(background).withBatchSize(20).withLink(
            _ShapConfig.LinkType.IDENTITY).build()
        tshap_explainer._explainer = _ShapKernelExplainer(tshap_explainer._config)

        ts_is = []
        os_is = []
        for k in np.arange(1, args.top_k):
            os_is.append(eval_shap_impact_original(oshap_explainer, predict_proba, unl_test_df[:args.samples], k))
            print(f'impact-score@{k} for original-shap-explainer: {os_is}')

            ts_is.append(eval_shap_impact(tshap_explainer, cb_model, predictions, k))
            print(f'impact-score@{k} for trustyai-shap-explainer: {ts_is}')

        print(f'Original-SHAP:{os_is}')
        print(f'trustyAI-SHAP:{ts_is}')

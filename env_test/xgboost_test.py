import numpy as np
import skl2onnx
import onnx
import pickle
import sklearn
import numpy
import onnxruntime as rt
import onnxmltools
from onnxmltools import convert_sklearn
import xgboost
import lightgbm
import catboost
from xgboost import XGBClassifier, XGBRegressor, XGBRFClassifier, XGBRFRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from skl2onnx.common.data_types import FloatTensorType
from sklearn.model_selection import cross_validate, GridSearchCV, RandomizedSearchCV, train_test_split
from sklearn.ensemble import (AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier,
                              GradientBoostingClassifier, RandomForestClassifier, HistGradientBoostingClassifier,
                              AdaBoostRegressor, BaggingRegressor, ExtraTreesRegressor, GradientBoostingRegressor,
                              RandomForestClassifier, HistGradientBoostingRegressor)
from sklearn.svm import SVC, NuSVC, LinearSVC, SVR, NuSVR, LinearSVR
from sklearn.linear_model import (LogisticRegression, RidgeClassifier, SGDClassifier, Perceptron,
                                  PassiveAggressiveClassifier, Lasso, ElasticNet, Lars, LassoLars,
                                  OrthogonalMatchingPursuit, BayesianRidge, ARDRegression, SGDRegressor,
                                  PassiveAggressiveRegressor, RANSACRegressor, TheilSenRegressor, HuberRegressor,
                                  QuantileRegressor)
from sklearn.kernel_ridge import KernelRidge
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.datasets import load_digits, load_diabetes
from sklearn.preprocessing import (TargetEncoder, OneHotEncoder, OrdinalEncoder, LabelEncoder,
                                   StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler)
from sklearn.impute import SimpleImputer
from sklearn.cluster import KMeans, MiniBatchKMeans
from cn_clip.global_def import dataset_map, algorithm_map, cv_search_map

encoder_map = {"target_mean": (TargetEncoder, None),
               "one_hot": (OneHotEncoder, {'handle_unknown': 'ignore'}),
               "ordinal": (OrdinalEncoder, {'handle_unknown': 'use_encoded_value', 'unknown_value': np.nan}),
               "target_ordinal": (LabelEncoder, None)}
scalar_map = {"standard": StandardScaler, "min_max": MinMaxScaler, "max_abs": MaxAbsScaler, "robust": RobustScaler}

hyperparams = {"n_clusters": 10, "n_init": 10}
search_params = {"max_depth": [1, 2], "n_estimators": [20, 40], "learning_rate": [0.1, 0.2]}
cluster_params = {"n_clusters": [5, 10], "n_init": [5, 10, 20]}

dataset = dataset_map["digits"]()
X, y = dataset.data, dataset.target
print(X.shape, X.dtype)
print(y.shape, y.dtype)

algorithm = algorithm_map["classification"]["xgboost"]()
scores = cross_validate(algorithm, X, y)
print("Average score:", np.average(scores['test_score']))

cv_search = cv_search_map["grid"](algorithm, search_params)
cv_search.fit(X, y)
# best_index = cv_search.cv_results_['rank_test_score'].argmin()
# best_param = cv_search.cv_results_['params'][best_index]
best_score = cv_search.best_score_
best_param = cv_search.best_params_
print("Best score:", best_score)
print("Best parameters:", best_param)
algorithm.set_params(**best_param)
algorithm.fit(X, y)

iris = dataset_map["digits"]()
X, y = iris.data, iris.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1)
print(X_train.shape, X_train.dtype)
print(y_train.shape, y_train.dtype)

clr = SVC()
clr.fit(X_train, y_train)
pickle.dumps(clr)
print("SVC训练集分数:", clr.score(X_train, y_train))
print("SVC测试集分数:", clr.score(X_test, y_test))
onx = convert_sklearn(clr, "svc", [("features", FloatTensorType([None, X_train.shape[1]]))])
pickle.dumps(onx)

providers = ['CUDAExecutionProvider'] if 'CUDAExecutionProvider' in rt.get_available_providers() else []
providers.append('CPUExecutionProvider')
print(providers)
sess = rt.InferenceSession(onx.SerializeToString(), providers=providers)
input_name = sess.get_inputs()[0].name
print(input_name)
label_name = sess.get_outputs()[0].name
print(label_name)
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

cluster = algorithm_map["clustering"]["mb_kmeans"](**hyperparams)
cv_search = cv_search_map["grid"](cluster, cluster_params)
cv_search.fit(X_train, y_train)
best_score = cv_search.best_score_
print("Cluster score:", best_score)
cluster.fit(X_train, y_train)
onx = convert_sklearn(cluster, "mb_kmeans", [("features", FloatTensorType([None, X_train.shape[1]]))])
pickle.dumps(onx)
sess = rt.InferenceSession(onx.SerializeToString(), providers=providers)
input_name = sess.get_inputs()[0].name
print(input_name)
label_name = sess.get_outputs()[0].name
print(label_name)
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

clr = NuSVC()
clr.fit(X_train, y_train)
pickle.dumps(clr)
print("NuSVC训练集分数:", clr.score(X_train, y_train))
print("NuSVC测试集分数:", clr.score(X_test, y_test))
onx = convert_sklearn(clr, "nusvc", [("features", FloatTensorType([None, X_train.shape[1]]))])
pickle.dumps(onx)

sess = rt.InferenceSession(onx.SerializeToString(), providers=providers)
input_name = sess.get_inputs()[0].name
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {input_name: X_test.astype(numpy.float32)})[0]
print(pred_onx)

# clr = xgb.XGBClassifier()
# clr.fit(X_train, y_train)
# pickle.dumps(clr)
# print("XGB训练集分数:", clr.score(X_train, y_train))
# print("XGB测试集分数:", clr.score(X_test, y_test))

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx:", onnx.__version__)
print("onnxruntime:", rt.__version__)
print("onnxmltools", onnxmltools.__version__)
print("skl2onnx:", skl2onnx.__version__)
print("xgboost:", xgboost.__version__)
print("lightgbm:", lightgbm.__version__)
print("catboost:", catboost.__version__)

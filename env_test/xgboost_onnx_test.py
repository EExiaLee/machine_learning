import numpy
import onnx
import onnxruntime as rt
import sklearn
from sklearn.datasets import load_digits, load_diabetes, load_iris, load_wine, load_linnerud, load_breast_cancer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
import xgboost
from xgboost import XGBClassifier, XGBRegressor
import skl2onnx
from skl2onnx.common.data_types import FloatTensorType
# from skl2onnx import convert_sklearn, update_registered_converter
# from skl2onnx.common.shape_calculator import (
#     calculate_linear_classifier_output_shapes,
#     calculate_linear_regressor_output_shapes
# )  # noqa
from onnxmltools import convert_xgboost
# from onnxmltools.convert.xgboost.operator_converters.XGBoost import convert_xgboost
import onnxmltools.convert.common.data_types

preprocessor_map = {"standard": StandardScaler, "min_max": MinMaxScaler, "max_abs": MaxAbsScaler,
                    "robust": RobustScaler, "impute": SimpleImputer}

data = load_digits()
# data = load_diabetes()
X = data.data
y = data.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

preprocessor = preprocessor_map["standard"]
pp_params = {}
scaler = preprocessor(**pp_params)
# pipe = Pipeline([("preprocessor", scaler), ("classifier", XGBClassifier(n_estimators=3))])
# pipe = Pipeline([("preprocessor", scaler), ("regressor", XGBRegressor(n_estimators=3))])
# pipe = XGBRegressor(n_estimators=3)
pipe = XGBClassifier(n_estimators=3)
pipe.fit(X, y)

# The conversion fails but it is expected.

# try:
#     convert_sklearn(
#         pipe,
#         "xgboost",
#         [("input", FloatTensorType([None, X.shape[1]]))],
#         target_opset={"": 12, "ai.onnx.ml": 2},
#     )
# except Exception as e:
#     print(e)

# The error message tells no converter was found
# for XGBoost models. By default, *sklearn-onnx*
# only handles models from *scikit-learn* but it can
# be extended to every model following *scikit-learn*
# API as long as the module knows there exists a converter
# for every model used in a pipeline. That's why
# we need to register a converter.
# update_registered_converter(
#     XGBClassifier,
#     "XGBoostXGBClassifier",
#     calculate_linear_classifier_output_shapes,
#     convert_xgboost,
#     options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
# )
# update_registered_converter(
#     XGBRegressor,
#     "XGBoostXGBRegressor",
#     calculate_linear_regressor_output_shapes,
#     convert_xgboost,
#     options={"nocl": [True, False]},
# )

# model_onnx = convert_sklearn(
#     pipe,
#     "xgboost",
#     [("input", FloatTensorType([None, X.shape[1]]))],
#     target_opset={"": 12, "ai.onnx.ml": 2},
# )
model_onnx = convert_xgboost(pipe, "xgboost", [("features", FloatTensorType([None, X.shape[1]]))])

# And save.
# with open("xgboost.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())

# for step in pipe:
#     print(step)
print("predict", pipe.predict(X[:5]))
# print("predict_proba", pipe.predict_proba(X[:1]))

sess = rt.InferenceSession(model_onnx.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {"features": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0].flatten())
# print("predict_proba", pred_onx[1][:1])

print("numpy:", numpy.__version__)
print("scikit-learn:", sklearn.__version__)
print("onnx: ", onnx.__version__)
print("onnxruntime: ", rt.__version__)
print("skl2onnx: ", skl2onnx.__version__)
print("onnxmltools: ", onnxmltools.__version__)
print("xgboost: ", xgboost.__version__)

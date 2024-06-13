import onnxruntime as rt
# from skl2onnx import convert_sklearn, update_registered_converter
# from skl2onnx.common.shape_calculator import (
#     calculate_linear_classifier_output_shapes,
# )  # noqa
# from onnxmltools.convert.lightgbm.operator_converters.LightGbm import (
#     convert_lightgbm,
# )  # noqa
from onnxmltools import convert_lightgbm
from skl2onnx.common.data_types import FloatTensorType
import numpy
from sklearn.datasets import load_digits, load_diabetes
# from sklearn.pipeline import Pipeline
# from sklearn.preprocessing import StandardScaler
from lightgbm import LGBMClassifier, LGBMRegressor

# data = load_digits()
dataset = load_diabetes()
X = dataset.data
y = dataset.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

# pipe = Pipeline([("scaler", StandardScaler()), ("lgbm", LGBMClassifier(n_estimators=3))])
# pipe = LGBMClassifier(n_estimators=3)
pipe = LGBMRegressor(n_estimators=3)
pipe.fit(X, y)

# update_registered_converter(
#     LGBMClassifier,
#     "LightGbmLGBMClassifier",
#     calculate_linear_classifier_output_shapes,
#     convert_lightgbm,
#     options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
# )
#
# model_onnx = convert_sklearn(
#     pipe,
#     "lightgbm",
#     [("input", FloatTensorType([None, X.shape[1]]))],
#     target_opset={"": 12, "ai.onnx.ml": 2},
# )
model_onnx = convert_lightgbm(pipe, "lightgbm", [("features", FloatTensorType([None, X.shape[1]]))])

# And save.
# with open("lightgbm.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())

print("predict", pipe.predict(X[:5]))
# print("predict_proba", pipe.predict_proba(X[:1]))

sess = rt.InferenceSession(model_onnx.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {"features": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0].flatten())
# print("predict_proba", pred_onx[1][:1])

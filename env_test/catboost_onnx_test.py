import numpy
from onnx.helper import get_attribute_value
from sklearn.datasets import load_digits, load_diabetes
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, RobustScaler
from sklearn.impute import SimpleImputer
import onnxruntime as rt
from onnxmltools.convert import convert_catboost, convert_sklearn
# from skl2onnx import convert_sklearn, update_registered_converter
# from skl2onnx.common.shape_calculator import (
#     calculate_linear_classifier_output_shapes,
#     calculate_linear_regressor_output_shapes
# )  # noqa
from skl2onnx.common.data_types import (
    FloatTensorType,
    Int64TensorType,
    guess_tensor_type,
)
from skl2onnx._parse import _apply_zipmap, _get_sklearn_operator_name
from catboost import CatBoostClassifier, CatBoostRegressor, Pool
from catboost.utils import convert_to_onnx_object

preprocessor_map = {"standard": StandardScaler, "min_max": MinMaxScaler, "max_abs": MaxAbsScaler,
                    "robust": RobustScaler, "impute": SimpleImputer}

# dataset = load_digits()
dataset = load_diabetes()
X = dataset.data
y = dataset.target

ind = numpy.arange(X.shape[0])
numpy.random.shuffle(ind)
X = X[ind, :].copy()
y = y[ind].copy()

preprocessor = preprocessor_map["min_max"]
pp_params = {"feature_range": (0, 1)}
scaler = preprocessor(**pp_params)
# pipe = Pipeline([("scaler", scaler), ("classifier", CatBoostClassifier(n_estimators=3))])
# pipe = Pipeline([("scaler", scaler), ("regressor", CatBoostRegressor(n_estimators=3))])
pipe = CatBoostRegressor(n_estimators=3)
# pipe = CatBoostClassifier(n_estimators=3)
pipe.fit(X, y, cat_features=[])


# def skl2onnx_parser_castboost(scope, model, inputs, custom_parsers):
#     options = scope.get_options(model, dict(zipmap=True))
#     no_zipmap = isinstance(options["zipmap"], bool) and not options["zipmap"]
#
#     alias = _get_sklearn_operator_name(type(model))
#     this_operator = scope.declare_local_operator(alias, model)
#     this_operator.inputs = inputs
#
#     label_variable = scope.declare_local_variable("label", Int64TensorType())
#     prob_dtype = guess_tensor_type(inputs[0].type)
#     probability_tensor_variable = scope.declare_local_variable(
#         "probabilities", prob_dtype
#     )
#     this_operator.outputs.append(label_variable)
#     this_operator.outputs.append(probability_tensor_variable)
#     probability_tensor = this_operator.outputs
#
#     if no_zipmap:
#         return probability_tensor
#
#     return _apply_zipmap(
#         options["zipmap"], scope, model, inputs[0].type, probability_tensor
#     )
#
#
# def skl2onnx_convert_catboost(scope, operator, container):
#     """
#     CatBoost returns an ONNX graph with a single node.
#     This function adds it to the main graph.
#     """
#     onx = convert_to_onnx_object(operator.raw_operator)
#     opsets = {d.domain: d.version for d in onx.opset_import}
#     if "" in opsets and opsets[""] >= container.target_opset:
#         raise RuntimeError("CatBoost uses an opset more recent than the target one.")
#     if len(onx.graph.initializer) > 0 or len(onx.graph.sparse_initializer) > 0:
#         raise NotImplementedError(
#             "CatBoost returns a model initializers. This option is not implemented yet."
#         )
#     if (
#             len(onx.graph.node) not in (1, 2)
#             or not onx.graph.node[0].op_type.startswith("TreeEnsemble")
#             or (len(onx.graph.node) == 2 and onx.graph.node[1].op_type != "ZipMap")
#     ):
#         types = ", ".join(map(lambda n: n.op_type, onx.graph.node))
#         raise NotImplementedError(
#             f"CatBoost returns {len(onx.graph.node)} != 1 (types={types}). "
#             f"This option is not implemented yet."
#         )
#     node = onx.graph.node[0]
#     atts = {}
#     for att in node.attribute:
#         atts[att.name] = get_attribute_value(att)
#     container.add_node(
#         node.op_type,
#         [operator.inputs[0].full_name],
#         [operator.outputs[0].full_name, operator.outputs[1].full_name],
#         op_domain=node.domain,
#         op_version=opsets.get(node.domain, None),
#         **atts,
#     )


# update_registered_converter(
#     CatBoostClassifier,
#     "CatBoostCatBoostClassifier",
#     calculate_linear_classifier_output_shapes,
#     skl2onnx_convert_catboost,
#     parser=skl2onnx_parser_castboost,
#     options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
# )
# update_registered_converter(
#     CatBoostRegressor,
#     "CatBoostCatBoostRegressor",
#     calculate_linear_regressor_output_shapes,
#     skl2onnx_convert_catboost,
#     # parser=skl2onnx_parser_castboost,
#     options={"nocl": [True, False], "zipmap": [True, False, "columns"]},
# )

# model_onnx = convert_sklearn(
#     pipe,
#     "catboost",
#     [("input", FloatTensorType([None, X.shape[1]]))],
#     target_opset={"": 12, "ai.onnx.ml": 2},
# )
model_onnx = convert_catboost(pipe, "catboost")

# And save.
# with open("catboost.onnx", "wb") as f:
#     f.write(model_onnx.SerializeToString())

print("predict", pipe.predict(X[:5]).flatten())
# print("predict_proba", pipe.predict_proba(X[:1]))

sess = rt.InferenceSession(model_onnx.SerializeToString(), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
label_name = sess.get_outputs()[0].name
pred_onx = sess.run([label_name], {"features": X[:5].astype(numpy.float32)})
print("predict", pred_onx[0].flatten())
# print("predict_proba", pred_onx[1][:1])

from skl2onnx.common.data_types import FloatTensorType
from onnxmltools import convert_sklearn, convert_catboost, convert_lightgbm, convert_xgboost
from pickle import loads
from thrift.protocol import TBinaryProtocol
from thrift.server import TServer
from thrift.transport import TSocket
from thrift.transport import TTransport
from cn_clip.global_def import convert_onnx_host, convert_onnx_port
from cn_clip.to_onnx import ConvertONNXService


class ConvertONNXHandler(ConvertONNXService.Iface):

    def convert_sklearn(self, model, algorithm, features):
        print(algorithm, features)
        onnx_model = convert_sklearn(loads(model), algorithm, [("features", FloatTensorType([None, features]))])
        return onnx_model.SerializeToString()

    def convert_xgboost(self, model, algorithm, features):
        print(algorithm, features)
        onnx_model = convert_xgboost(loads(model), algorithm, [("features", FloatTensorType([None, features]))])
        return onnx_model.SerializeToString()

    def convert_lightgbm(self, model, algorithm, features):
        print(algorithm, features)
        onnx_model = convert_lightgbm(loads(model), algorithm, [("features", FloatTensorType([None, features]))])
        return onnx_model.SerializeToString()

    def convert_catboost(self, model, algorithm):
        print(algorithm)
        onnx_model = convert_catboost(loads(model), algorithm)
        return onnx_model.SerializeToString()


if __name__ == "__main__":
    handler = ConvertONNXHandler()
    processor = ConvertONNXService.Processor(handler)

    transport = TSocket.TServerSocket(host=convert_onnx_host, port=convert_onnx_port)
    tfactory = TTransport.TBufferedTransportFactory()
    pfactory = TBinaryProtocol.TBinaryProtocolFactory()

    server = TServer.TThreadPoolServer(processor, transport, tfactory, pfactory)
    print(f"Started an ONNX conversion service at {convert_onnx_host}:{convert_onnx_port}")
    server.serve()

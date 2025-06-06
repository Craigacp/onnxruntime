// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/providers/qnn/builder/opbuilder/base_op_builder.h"
#include "core/providers/qnn/builder/qnn_model_wrapper.h"
#include "core/providers/qnn/builder/op_builder_factory.h"
#include "core/providers/qnn/builder/qnn_utils.h"

namespace onnxruntime {
namespace qnn {

class GemmOpBuilder : public BaseOpBuilder {
 public:
  GemmOpBuilder() : BaseOpBuilder("GemmOpBuilder") {}
  ORT_DISALLOW_COPY_ASSIGNMENT_AND_MOVE(GemmOpBuilder);

 protected:
  Status ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                       const NodeUnit& node_unit,
                       const logging::Logger& logger,
                       std::vector<std::string>& input_names,
                       bool do_op_validation) const override ORT_MUST_USE_RESULT;

  Status ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                     const NodeUnit& node_unit,
                                     std::vector<std::string>&& input_names,
                                     const logging::Logger& logger,
                                     bool do_op_validation) const override ORT_MUST_USE_RESULT;

 private:
  Status ExplictOpCheck(const NodeUnit& node_unit) const;
};

Status GemmOpBuilder::ExplictOpCheck(const NodeUnit& node_unit) const {
  NodeAttrHelper node_helper(node_unit);
  auto alpha = node_helper.Get("alpha", (float)1.0);
  if (alpha != 1.0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN FullyConnected Op only support alpha=1.0.");
  }
  auto beta = node_helper.Get("beta", (float)1.0);
  if (beta != 1.0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN FullyConnected Op only support beta=1.0.");
  }

  // input C shape need to be [M] or [1, M]
  if (node_unit.Inputs().size() == 3) {
    auto& inputB = node_unit.Inputs()[1];
    std::vector<uint32_t> inputB_shape;
    QnnModelWrapper::GetOnnxShape(inputB.node_arg, inputB_shape);

    auto& inputC = node_unit.Inputs()[2];
    std::vector<uint32_t> inputC_shape;
    QnnModelWrapper::GetOnnxShape(inputC.node_arg, inputC_shape);

    auto transB = node_helper.Get("transB", static_cast<int64_t>(0));
    auto M = (transB == 0) ? inputB_shape.at(1) : inputB_shape.at(0);
    if (inputC_shape.size() == 0 || (inputC_shape.size() == 1 && inputC_shape.at(0) != M) ||
        (inputC_shape.size() == 2 && (inputC_shape.at(0) != 1 || inputC_shape.at(1) != M))) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "QNN FullyConnected Op only support C with shape [M].");
    }
  }

  return Status::OK();
}

Status GemmOpBuilder::ProcessInputs(QnnModelWrapper& qnn_model_wrapper,
                                    const NodeUnit& node_unit,
                                    const logging::Logger& logger,
                                    std::vector<std::string>& input_names,
                                    bool do_op_validation) const {
  if (do_op_validation) {
    ORT_RETURN_IF_ERROR(ExplictOpCheck(node_unit));
  }
  Qnn_DataType_t qnn_data_type = QNN_DATATYPE_FLOAT_32;

  // for Input A, B, C: 1 -- need transpose, 0 -- not needed
  std::vector<int64_t> input_trans_flag(3, 0);
  NodeAttrHelper node_helper(node_unit);
  input_trans_flag.at(0) = node_helper.Get("transA", (int64_t)0);
  auto transB = node_helper.Get("transB", (int64_t)0);
  // QNN input_1 [m, n] vs Onnx [n, m]
  input_trans_flag.at(1) = transB == 0 ? 1 : 0;

  const auto& inputs = node_unit.Inputs();
  for (size_t input_i = 0; input_i < inputs.size(); ++input_i) {
    QnnQuantParamsWrapper quantize_param;
    ORT_RETURN_IF_ERROR(quantize_param.Init(qnn_model_wrapper, inputs[input_i]));

    bool is_quantized_tensor = inputs[input_i].quant_param.has_value();
    const auto& input_name = inputs[input_i].node_arg.Name();

    // Only skip if the input tensor has already been added (by producer op) *and* we don't need
    // to transpose it.
    if (qnn_model_wrapper.IsQnnTensorWrapperExist(input_name) && input_trans_flag[input_i] == 0) {
      LOGS(logger, VERBOSE) << "Tensor already added, skip it: " << input_name;
      input_names.push_back(input_name);
      continue;
    }

    const auto* type_proto = inputs[input_i].node_arg.TypeAsProto();
    ORT_RETURN_IF_ERROR(utils::GetQnnDataType(is_quantized_tensor, type_proto, qnn_data_type));

    std::vector<uint32_t> input_shape;
    ORT_RETURN_IF_NOT(qnn_model_wrapper.GetOnnxShape(inputs[input_i].node_arg, input_shape), "Cannot get shape");

    std::vector<uint8_t> unpacked_tensor;
    bool is_constant_input = qnn_model_wrapper.IsConstantInput(input_name);
    if (is_constant_input) {
      const auto& input_tensor = qnn_model_wrapper.GetConstantTensor(input_name);
      if (1 == input_trans_flag.at(input_i)) {
        ORT_RETURN_IF_ERROR(quantize_param.HandleTranspose<size_t>(std::vector<size_t>({1, 0})));
        ORT_RETURN_IF_ERROR(
            utils::TwoDimensionTranspose(qnn_model_wrapper, input_shape, *input_tensor, unpacked_tensor));
      } else {
        ORT_RETURN_IF_ERROR(qnn_model_wrapper.UnpackInitializerData(*input_tensor, unpacked_tensor));
      }
    }

    std::string input_tensor_name = input_name;
    if (1 == input_trans_flag.at(input_i) && !is_constant_input) {
      ORT_RETURN_IF(quantize_param.IsPerChannel(), "Non-constant Gemm inputs only support per-tensor quantization");

      // Add Transpose node
      std::vector<uint32_t> old_input_shape(input_shape);
      input_shape[0] = old_input_shape[1];
      input_shape[1] = old_input_shape[0];
      const std::string& node_input_name(input_name);
      input_tensor_name = input_tensor_name + "_ort_qnn_ep_transpose";
      std::vector<uint32_t> perm{1, 0};
      ORT_RETURN_IF_ERROR(qnn_model_wrapper.AddTransposeNode(node_unit.Index(), node_input_name, input_tensor_name,
                                                             old_input_shape, perm, input_shape,
                                                             qnn_data_type, quantize_param, do_op_validation,
                                                             qnn_model_wrapper.IsGraphInput(node_input_name)));
    }

    if (2 == input_i && 2 == input_shape.size()) {
      input_shape[0] = input_shape[1];
      input_shape.resize(1);
    }

    input_names.push_back(input_tensor_name);
    Qnn_TensorType_t tensor_type = qnn_model_wrapper.GetTensorType(input_tensor_name);
    QnnTensorWrapper input_tensorwrapper(input_tensor_name, tensor_type, qnn_data_type, std::move(quantize_param),
                                         std::move(input_shape), std::move(unpacked_tensor));
    ORT_RETURN_IF_NOT(qnn_model_wrapper.AddTensorWrapper(std::move(input_tensorwrapper)), "Failed to add tensor.");

    if (1 == input_i) {
      // Workaround that inserts a QNN Convert op before input[1] (converts from quantized uint16 to signed symmetric int16)
      // to avoid a QNN validation failure.
      //
      // QNN graph WITHOUT workaround (fails validation):
      //     input_0_uint16 ---> FC ---> output_uint16
      //                         ^
      //                         |
      //     input_1_uint16 -----+
      //
      // QNN graph WITH workaround (passes validation):
      //     input_0_uint16 ----------------------> FC ---> output_uint16
      //                                            ^
      //                                            |
      //     input_1_uint16 --> Convert(to int16) --+

      std::string weight_input_name = input_tensor_name;
      const auto& weight_tensor_wrapper = qnn_model_wrapper.GetQnnTensorWrapper(weight_input_name);

      if (weight_tensor_wrapper.GetTensorDataType() == QNN_DATATYPE_UFIXED_POINT_16) {
        const auto& quant_param_wrapper = weight_tensor_wrapper.GetQnnQuantParams();
        const Qnn_QuantizeParams_t& quant_param = quant_param_wrapper.Get();
        const auto& transformed_input1_shape = weight_tensor_wrapper.GetTensorDims();

        ORT_RETURN_IF_NOT(quant_param_wrapper.IsPerTensor(),
                          "FC's INT16 weight inputs only support INT16 per-tensor quantization");

        // Pop FC weight. Insert Convert op after Weight
        input_names.pop_back();
        const std::string& fc_output_name = node_unit.Outputs()[0].node_arg.Name();
        std::string convert_output_name = weight_input_name + "_convert_" + fc_output_name;

        ORT_RETURN_IF_ERROR(utils::InsertConvertOp(qnn_model_wrapper,
                                                   weight_input_name,
                                                   convert_output_name,
                                                   QNN_DATATYPE_UFIXED_POINT_16,
                                                   QNN_DATATYPE_SFIXED_POINT_16,
                                                   quant_param.scaleOffsetEncoding.offset,
                                                   quant_param.scaleOffsetEncoding.scale,
                                                   transformed_input1_shape,
                                                   true,  // Symmetric
                                                   do_op_validation));
        input_names.push_back(convert_output_name);
      }
    }
  }

  return Status::OK();
}

Status GemmOpBuilder::ProcessAttributesAndOutputs(QnnModelWrapper& qnn_model_wrapper,
                                                  const NodeUnit& node_unit,
                                                  std::vector<std::string>&& input_names,
                                                  const logging::Logger& logger,
                                                  bool do_op_validation) const {
  ORT_RETURN_IF_ERROR(ProcessOutputs(qnn_model_wrapper, node_unit, std::move(input_names), {},
                                     logger, do_op_validation, GetQnnOpType(node_unit.OpType())));
  return Status::OK();
}

void CreateGemmOpBuilder(const std::string& op_type, OpBuilderRegistrations& op_registrations) {
  op_registrations.AddOpBuilder(op_type, std::make_unique<GemmOpBuilder>());
}

}  // namespace qnn
}  // namespace onnxruntime

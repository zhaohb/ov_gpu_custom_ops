// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

//! [op:common_include]
#include <openvino/op/op.hpp>
#include <vector>
//! [op:common_include]

//! [op:header]
namespace TemplateExtension {

class SimpleGPU : public ov::op::Op {
public:
    OPENVINO_OP("SimpleGPU");

    SimpleGPU() = default;
    SimpleGPU(const ov::Output<ov::Node>& A, const ov::Output<ov::Node>& B, const ov::Output<ov::Node>& C, int32_t queryLen, int32_t kvSeqLen, int32_t batchSize, int32_t hasMask);
    // SimpleGPU(const ov::Output<ov::Node>& A, const ov::Output<ov::Node>& B, const ov::Output<ov::Node>& C);
    void validate_and_infer_types() override;
    std::shared_ptr<ov::Node> clone_with_new_inputs(const ov::OutputVector& new_args) const override;
    bool visit_attributes(ov::AttributeVisitor& visitor) override;

    bool evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const override;
    bool has_evaluate() const override;


private:
    int32_t queryLen;
    int32_t kvSeqLen;
    int32_t batchSize;
    int32_t hasMask;
};
//! [op:header]

}  // namespace TemplateExtension

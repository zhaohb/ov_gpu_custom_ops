// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "simplegpu.hpp"

using namespace TemplateExtension;

//! [op:ctor]
SimpleGPU::SimpleGPU(const ov::Output<ov::Node>& A, const ov::Output<ov::Node>& B, const ov::Output<ov::Node>& C , const ov::Output<ov::Node>& D, int32_t queryLen, int32_t kvSeqLen, int32_t batchSize, int32_t hasMask): Op({A, B, C, D}), queryLen(queryLen), kvSeqLen(kvSeqLen), batchSize(batchSize), hasMask(hasMask){
    constructor_validate_and_infer_types();
}
//! [op:ctor]

//! [op:validate]
void SimpleGPU::validate_and_infer_types() {
    set_output_type(0, get_input_element_type(0), get_input_partial_shape(0));
}
//! [op:validate]

//! [op:copy]
std::shared_ptr<ov::Node> SimpleGPU::clone_with_new_inputs(const ov::OutputVector& new_args) const {
    // return std::make_shared<SimpleGPU>(new_args[0], new_args[1], new_args[2]);
    return std::make_shared<SimpleGPU>(new_args[0], new_args[1], new_args[2], new_args[3], queryLen, kvSeqLen, batchSize, hasMask);
}
//! [op:copy]

//! [op:visit_attributes]
bool SimpleGPU::visit_attributes(ov::AttributeVisitor& visitor) {
    visitor.on_attribute("queryLen", queryLen);
    visitor.on_attribute("kvSeqLen", kvSeqLen);
    visitor.on_attribute("batchSize", batchSize);
    visitor.on_attribute("hasMask", hasMask);
    return true;
}
//! [op:visit_attributes]

//! [op:evaluate]
bool SimpleGPU::evaluate(ov::TensorVector& outputs, const ov::TensorVector& inputs) const {
    return true;
}

bool SimpleGPU::has_evaluate() const {
    return true;
}
//! [op:evaluate]

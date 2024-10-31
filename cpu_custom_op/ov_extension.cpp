// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <openvino/core/extension.hpp>
#include <openvino/core/op_extension.hpp>
#include <openvino/frontend/extension.hpp>
#include <openvino/frontend/extension/conversion.hpp>

#include "simplegpu/simplegpu.hpp"

// clang-format off
//! [ov_extension:entry_point]
OPENVINO_CREATE_EXTENSIONS(
    std::vector<ov::Extension::Ptr>({
        // Register operation itself, required to be read from IR
        std::make_shared<ov::OpExtension<TemplateExtension::SimpleGPU>>(),

        // Register operaton mapping, required when converted from framework model format
        std::make_shared<ov::frontend::OpExtension<TemplateExtension::SimpleGPU>>(),
#if 0
        std::make_shared<ov::frontend::ConversionExtension(
            "FloatBucketizePlugin",
            [](const ov::frontend::NodeContext& node){
                auto data = node.get_ng_inputs().at(0);
                std::vector<std::float> boundaries = node.get_attribute_value<std::vector<std::float>>("boundaries", {});
		
                auto boundaries_len = node.get_attribute<std::int32_t>("boundaries_len");

		ov::opset13::Constant::create(ov::element::f32, {}, {node.get_attribute<std::vector<float>>("boundaries")});
                return {std::make_shared<TemplateExtension::FloatBucketizePlugin>(node.get_input(0), ov::opset13::Constant::create(ov::element::f32, {}, {node.get_attribute<std::vector<float>>("boundaries")}), boundaries_len)};
        })
#endif
    }));
//! [ov_extension:entry_point]
// clang-format on

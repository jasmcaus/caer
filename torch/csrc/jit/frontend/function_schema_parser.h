#pragma once

#include <ATen/core/Macros.h>
#include <ATen/core/function_schema.h>
#include <c10/util/either.h>
#include <string>

namespace torch {
namespace jit {

CAFFE2_API c10::either<c10::OperatorName, c10::FunctionSchema> parseSchemaOrName(
    const std::string& schemaOrName);
CAFFE2_API c10::FunctionSchema parseSchema(const std::string& schema);
CAFFE2_API c10::OperatorName parseName(const std::string& name);

} // namespace jit
} // namespace torch

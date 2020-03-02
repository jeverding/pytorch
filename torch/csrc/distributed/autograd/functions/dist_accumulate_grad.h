#pragma once

#include <memory>

#include <torch/csrc/WindowsTorchApiMacro.h>
#include <torch/csrc/autograd/function.h>
#include <torch/csrc/autograd/variable.h>
#include <torch/csrc/distributed/autograd/context/context.h>

namespace torch {

namespace autograd {
class AccumulateGrad;
}

namespace distributed::autograd {

using torch::autograd::AccumulateGrad;
using torch::autograd::Node;
using torch::autograd::Variable;
using torch::autograd::variable_list;

class TORCH_API DistAccumulateGrad : public Node {
 public:
  DistAccumulateGrad(
      std::shared_ptr<AccumulateGrad> accumulateGrad,
      std::shared_ptr<DistAutogradContext> autogradContext);

  ~DistAccumulateGrad() = default;

  std::shared_ptr<AccumulateGrad> restoreAccumulateGrad();

  variable_list apply(variable_list&& grads) override;

 private:
  // 'accumulateGrad_' got replaced by this DistAccumulateGrad in the autograd
  // graph.
  std::shared_ptr<AccumulateGrad> accumulateGrad_;
  std::shared_ptr<DistAutogradContext> autogradContext_;
};

}} // namespace torch::distributed::autograd

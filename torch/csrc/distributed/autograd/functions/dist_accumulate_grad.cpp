#include <torch/csrc/distributed/autograd/functions/dist_accumulate_grad.h>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/variable.h>

namespace torch::distributed::autograd {

DistAccumulateGrad::DistAccumulateGrad(
      std::shared_ptr<AccumulateGrad> accumulateGrad,
      std::shared_ptr<DistAutogradContext> autogradContext)
    : Node(accumulateGrad->sequence_nr()),
      accumulateGrad_(accumulateGrad),
      autogradContext_(std::move(autogradContext)) {
  move_from(*accumulateGrad_);
}

std::shared_ptr<AccumulateGrad> DistAccumulateGrad::restoreAccumulateGrad() {
  // Move back metadata (e.g. hooks) to AccumulateGrad.
  accumulateGrad_->move_from(*this);
  return std::move(accumulateGrad_);
}

variable_list DistAccumulateGrad::apply(variable_list&& grads) {
  // XXX: this method is not thread-safe!
  torch::autograd::check_input_variables("DistAccumulateGrad", grads, 1, 0);
  TORCH_INTERNAL_ASSERT(accumulateGrad_);

  autogradContext_->accumulateGrad(
      accumulateGrad_->variable, grads[0], 1 /* num_expected_refs */);
  return variable_list();
}

} // namespace torch::distributed::autograd

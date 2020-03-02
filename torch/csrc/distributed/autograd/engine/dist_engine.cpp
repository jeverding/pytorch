#include <queue>

#include <torch/csrc/autograd/functions/accumulate_grad.h>
#include <torch/csrc/autograd/input_buffer.h>
#include <torch/csrc/distributed/autograd/context/container.h>
#include <torch/csrc/distributed/autograd/engine/dist_engine.h>
#include <torch/csrc/distributed/autograd/functions/dist_accumulate_grad.h>

namespace torch {
namespace distributed {
namespace autograd {

using torch::autograd::AccumulateGrad;
using torch::autograd::edge_list;
using torch::autograd::Engine;
using torch::autograd::FutureVariableList;
using torch::autograd::GraphRoot;
using torch::autograd::GraphTask;
using torch::autograd::Node;
using torch::autograd::validate_outputs;
using torch::autograd::variable_list;

static constexpr char* kNumBackwardPasses = "num_current_backward_passes";
static constexpr char* kEngineCPUQueueSize =
    "local_autograd_engine_cpu_queue_size";
static constexpr char* kNumAutogradContexts = "num_autograd_contexts";


DistEngine::DistEngine()
    : initializedContextIds_(), engine_(Engine::get_default_engine()) {}

DistEngine& DistEngine::getInstance() {
  // Leaky singleton to avoid module destructor race.
  static DistEngine* engine = new DistEngine();
  return *engine;
}

void DistEngine::restoreGraph(DistEngineContext&& context) const {
  while (!context.distAccumulateGradCtx.empty()) {
    const auto& oneEdge = context.distAccumulateGradCtx.top();
    TORCH_INTERNAL_ASSERT(
        oneEdge.to ==
        dynamic_cast<DistAccumulateGrad*>(
            oneEdge.from->next_edge(oneEdge.edgeIndex).function.get()));
    auto accumulateGradFn = oneEdge.to->restoreAccumulateGrad();
    oneEdge.from->next_edge(oneEdge.edgeIndex).function =
        std::move(accumulateGradFn);
    context.distAccumulateGradCtx.pop();
  }
  LOG(ERROR) << "Restoring the graph is done.";
}

void DistEngine::validateRootsAndRetrieveEdges(
    const variable_list& roots,
    edge_list& rootEdges,
    variable_list& grads) {
  TORCH_CHECK(!roots.empty(), "No tensors provided for gradient computation.");
  TORCH_INTERNAL_ASSERT(rootEdges.empty());
  TORCH_INTERNAL_ASSERT(grads.empty());

  // Verify roots are all scalar and require gradients.
  for (const auto& root : roots) {
    TORCH_CHECK(
        root.requires_grad(), "requires_grad not set on: ", root.name());
    TORCH_CHECK(
        root.numel() == 1,
        root.name(),
        " is not a scalar, all roots need to be scalar");
    TORCH_CHECK(
        root.grad_fn(),
        root.name(),
        " does not have a valid gradient function.");

    // Compute the root edges and generate the appropriate gradients.
    rootEdges.push_back(torch::autograd::impl::gradient_edge(root));
    grads.push_back(at::ones_like(root, LEGACY_CONTIGUOUS_MEMORY_FORMAT));
    LOG(ERROR) << "root edge from a scalar output: " << rootEdges.back().function->name() << " seq: " << rootEdges.back().function->sequence_nr();
  }

  // Validate rootEdges and grads.
  validate_outputs(
      rootEdges, grads, [](const std::string& msg) { return msg; });
}

// NB: this function modifies the autograd graph. More specifically it replaces
// all AccumulateGrad nodes by DistAccumulateGrad.
auto DistEngine::computeDependencies(
    const ContextPtr& autogradContext,
    const edge_list& rootEdges,
    const variable_list& grads,
    const std::shared_ptr<Node>& graphRoot,
    edge_list& outputEdges,
    bool retainGraph) -> DistEngineContext {
  TORCH_INTERNAL_ASSERT(graphRoot, "graphRoot is null!");
  LOG(ERROR) << "computeDependencies for graph root: @" << graphRoot.get() << " for " << rootEdges.size() << " root edges.";

  DistEngineContext distEngineContext;
  // Build the graph task and graph root.
  auto graphTask = std::make_shared<GraphTask>(
      /* keep_graph */ retainGraph,
      /* create_graph */ false,
      /* depth */ 0,
      /* exit_on_error */ true);

  // Run BFS to traverse the graph locally. The roots of the graph are
  // GraphRoot and all send functions for this autograd context.
  std::unordered_set<Node*> seen;
  std::queue<Node*> queue;
  queue.push(static_cast<Node*>(graphRoot.get()));

  auto sendFunctions = autogradContext->sendFunctions();

  // Add all the send functions to the queue as roots.
  for (const auto& mapEntry : sendFunctions) {
    // Increment 'outstanding_tasks_' for GraphTask for each send_function
    // since we want the local autograd engine to wait for all of them.
    graphTask->outstanding_tasks_++;
    queue.push(mapEntry.second.get());
  }

  // Traverse the graph.
  auto& dependencies = graphTask->dependencies_;
  // If a node is in this map, it should be replaced by its corresponding value
  // in the graph, i.e. all edges pointing to the key should point to the value.
  std::unordered_map<Node*, std::shared_ptr<DistAccumulateGrad>>
      accumulateGradReplacements;
  while (!queue.empty()) {
    auto fn = queue.front();
    LOG(ERROR) << "processing function @" << fn->sequence_nr() << " in computeDependencies()";
    queue.pop();

    std::string line = fn->name();
    for (size_t index = 0; index < fn->num_outputs(); ++index) {
      const auto& edge = fn->next_edge(index);
      if (auto nextFn = edge.function.get()) {
        line += "\n    ----> " + nextFn->name() + " @" +
            std::to_string(nextFn->sequence_nr());
        dependencies[nextFn] += 1;
        const bool wasInserted = seen.insert(nextFn).second;
        if (wasInserted) {
          // Seeing this function for the first time.
          queue.push(nextFn);
          if (auto ptr = dynamic_cast<AccumulateGrad*>(nextFn)) {
            LOG(ERROR) << "AccumulateGrad has " << nextFn->num_inputs()
                       << " inputs and " << edge.function.use_count()
                       << " shared_ptr use_count for variable: "
                       << ptr->variable.toString();
          }

          if (auto accumulateGradFn = dynamic_cast<AccumulateGrad*>(nextFn)) {
            LOG(ERROR) << "AccumulateGrad @"
                       << accumulateGradFn->sequence_nr() << " on variable @"
                       << accumulateGradFn->variable.unsafeGetTensorImpl();

            // Replace an AccumulateGrad node by DistAccumulateGrad,
            // because the former accumulates grads to the variable's '.grad'
            // without considering the context id. That may cause data race
            // on the '.grad', since multiple context ids may have grads for
            // the same '.grad'.
            auto distAccumulateGradFn = std::make_shared<DistAccumulateGrad>(
                std::dynamic_pointer_cast<AccumulateGrad>(edge.function),
                autogradContext);
            LOG(ERROR) << "created a DistAccumulateGrad @"
                       << distAccumulateGradFn.get() << " for AccumulateGrad @"
                       << nextFn;
            distEngineContext.distAccumulateGradCtx.push(
                DistAccumulateGradContext{
                    .from = fn,
                    .to = distAccumulateGradFn.get(),
                    .edgeIndex = index,
                });
            accumulateGradReplacements[nextFn] = distAccumulateGradFn;
            fn->next_edge(index).function = std::move(distAccumulateGradFn);
          }
        } else {
          auto itr = accumulateGradReplacements.find(nextFn);
          if (itr != accumulateGradReplacements.end()) {
            LOG(ERROR) << "Replaced a AccumulateGrad @" << nextFn
                       << " by DistAccumulateGrad @" << itr->second.get();
            fn->next_edge(index).function = itr->second;
            distEngineContext.distAccumulateGradCtx.push(
                DistAccumulateGradContext{
                    .from = fn,
                    .to = itr->second.get(),
                    .edgeIndex = index,
                });
          }
        }
      }
    }
    LOG(ERROR) << "Edges from node " << fn->name() << " @"
               << std::to_string(fn->sequence_nr()) << ": " << line;
  }
  for (const auto& pr : accumulateGradReplacements) {
    const auto& node = pr.first;
    const auto& distAccumulateGrad = pr.second;
    TORCH_INTERNAL_ASSERT(
        dependencies.count(distAccumulateGrad.get()) == 0,
        "DistAccumulateGrad nodes shouldn't be in dependencies");
    auto itr = dependencies.find(node);
    TORCH_INTERNAL_ASSERT(
        itr != dependencies.end(),
        "Replaced AccumulateGrad nodes should be in dependencies.");
    dependencies[distAccumulateGrad.get()] = itr->second;
    dependencies.erase(itr);
  }

  LOG(ERROR) << "Replaced " << accumulateGradReplacements.size()
             << " AccumulateGrad nodes.";
  TORCH_INTERNAL_ASSERT(
      graphTask->exec_info_.empty(), "Should run the whole graph!");
  // Let autograd context take ownership of the GraphTask.
  autogradContext->setGraphTask(std::move(graphTask));
  return distEngineContext;
}

std::shared_ptr<rpc::FutureMessage> DistEngine::runEngineAndAccumulateGradients(
    const ContextPtr& autogradContext,
    const std::shared_ptr<Node>& graphRoot,
    const edge_list& outputEdges,
    std::function<void()> onFinish) {
  // Cleanup previous state for outstanding RPCs. Outstanding RPCs could be
  // lingering if we're running backward multiple times and some of the
  // passes ran into errors.
  autogradContext->clearOutstandingRpcs();

  auto futureGrads = engine_.execute_with_graph_task(
      autogradContext->retrieveGraphTask(), graphRoot);

  // Build a future that waits for the callbacks to execute (since callbacks
  // execute after the original future is completed). This ensures we return a
  // future that waits for all gradient accumulation to finish.
  auto accumulateGradFuture = std::make_shared<rpc::FutureMessage>();

  futureGrads->addCallback(
      [autogradContext,
       outputEdges,
       accumulateGradFuture,
       onFinish = std::move(onFinish)](
          const variable_list& grads,
          const c10::optional<torch::utils::FutureError>& error) {
        onFinish();

        if (error) {
          // Don't accumulate gradients if we receive an error.
          // We must add the node information here since DistEngine::execute
          // waits on accumulateGradFuture and will throw an exception once we
          // set the error below.
          std::string errorMsg = c10::str(
              "Error on Node ",
              DistAutogradContainer::getInstance().getWorkerId(),
              ": ",
              error->what());
          accumulateGradFuture->setError(errorMsg);
          return;
        }

        TORCH_INTERNAL_ASSERT(grads.size() == outputEdges.size());
        accumulateGradFuture->markCompleted(rpc::Message());
      });

  return accumulateGradFuture;
}

std::shared_ptr<rpc::FutureMessage> DistEngine::executeSendFunctionAsync(
    const ContextPtr& autogradContext,
    const std::shared_ptr<Node>& sendFunction,
    bool retainGraph) {
  std::unique_lock<std::mutex> lock(initializedContextIdsLock_);
  if (initializedContextIds_.find(autogradContext->contextId()) ==
      initializedContextIds_.end()) {
    edge_list outputEdges;
    // Pass in a dummy graphRoot since all send functions are the roots.
    auto dummyRoot = std::make_shared<GraphRoot>(edge_list(), variable_list());
    auto distEngineContext = computeDependencies(
        autogradContext, {}, {}, dummyRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized and unlock.
    initializedContextIds_.insert(autogradContext->contextId());
    lock.unlock();

    // Enqueue the current send function.
    auto graphTask = autogradContext->retrieveGraphTask();
    engine_.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask, sendFunction, torch::autograd::InputBuffer(0)));

    // Run the autograd engine.
    auto accumulateGradFuture = runEngineAndAccumulateGradients(
        autogradContext,
        dummyRoot,
        outputEdges,
        [this, distEngineContext = std::move(distEngineContext)]() mutable {
          this->restoreGraph(std::move(distEngineContext));
        });

    // Build the 'uber' future that waits for everything.
    auto callbackFuture = std::make_shared<rpc::FutureMessage>();

    accumulateGradFuture->addCallback(
        [autogradContext, callbackFuture](
            const rpc::Message& message /* unused */,
            const c10::optional<torch::utils::FutureError>& error) mutable {
          if (error) {
            // Perform cleanup at the end of the backward pass (before we mark
            // the future as completed).
            DistEngine::getInstance().cleanupBackwardPass(autogradContext);

            // Skip any further processing on errors.
            callbackFuture->setError(error->what());
            return;
          }

          // Wait for all RPCs after the autograd engine is done.
          auto rpcFuture =
              autogradContext->clearAndWaitForOutstandingRpcsAsync();
          rpcFuture->addCallback(
              [callbackFuture, autogradContext](
                  const rpc::Message& /* unused */,
                  const c10::optional<torch::utils::FutureError>& error) {
                // Perform cleanup at the end of the backward pass (before we
                // mark the future as completed).
                DistEngine::getInstance().cleanupBackwardPass(autogradContext);

                // Finally mark the 'uber' future as completed.
                if (!error) {
                  callbackFuture->markCompleted(rpc::Message());
                } else {
                  callbackFuture->setError(error->what());
                }
              });
        });

    // Return the future which waits for all async processing to be done.
    return callbackFuture;
  } else {
    lock.unlock();
    auto graphTask = autogradContext->retrieveGraphTask();
    engine_.enqueue_blocked_task_on_cpu(torch::autograd::NodeTask(
        graphTask, sendFunction, torch::autograd::InputBuffer(0)));
    return std::make_shared<rpc::FutureMessage>(rpc::Message());
  }
}

void DistEngine::execute(
    int64_t contextId,
    const variable_list& roots,
    bool retainGraph) {
  // Retrieve the context for the given context_id. This will throw if the
  // context_id is invalid.
  auto autogradContext =
      DistAutogradContainer::getInstance().retrieveContext(contextId);

  // Perform initial pre-processing.
  edge_list rootEdges;
  variable_list grads;
  validateRootsAndRetrieveEdges(roots, rootEdges, grads);

  std::shared_ptr<Node> graphRoot =
      std::make_shared<GraphRoot>(rootEdges, grads);
  edge_list outputEdges;
  DistEngineContext distEngineContext;
  // Compute dependencies locally, starting from all roots and all 'send'
  // functions.
  {
    std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
    // Context should not have been initialized already.
    TORCH_INTERNAL_ASSERT(
        initializedContextIds_.find(autogradContext->contextId()) ==
        initializedContextIds_.end());

    LOG(ERROR) << "computing dependencies and set up the graphTask in the context.";
    distEngineContext = computeDependencies(
        autogradContext, rootEdges, grads, graphRoot, outputEdges, retainGraph);

    // Mark the autograd context id as initialized.
    initializedContextIds_.insert(autogradContext->contextId());
  }

  BackwardPassCleanupGuard guard(autogradContext);

  // This needs to be blocking and as a result we wait for the future to
  // complete.
  runEngineAndAccumulateGradients(
      autogradContext,
      graphRoot,
      outputEdges,
      [this, distEngineContext = std::move(distEngineContext)]() mutable {
        this->restoreGraph(std::move(distEngineContext));
      })
      ->wait();

  // Wait for all of the outstanding rpcs to complete.
  autogradContext->clearAndWaitForOutstandingRpcsAsync()->wait();
}

void DistEngine::cleanupBackwardPass(const ContextPtr& autogradContext) {
  // Validate only the GraphTask is holding a reference to the Future
  // which holds gradients for the backward pass. This ensures that
  // after 'resetGraphTask' is called below, there are no remaining
  // references left to the gradients for the backward pass.
  //
  // This ensures our 'use_count' checks in
  // AccumulateGrad::accumulateGradAndCallHooks are correct and we're
  // not leaking any references to the gradients anywhere else.
  const auto& futureGrads =
      autogradContext->retrieveGraphTask()->future_result_;
  TORCH_INTERNAL_ASSERT(futureGrads.use_count() == 1);

  // Reset the graph task once we're done with all processing.
  autogradContext->resetGraphTask();

  // Clear any outstanding rpcs.
  autogradContext->clearOutstandingRpcs();

  // Clear the context id once we're done with the autograd engine
  // processing.
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  initializedContextIds_.erase(autogradContext->contextId());
}

size_t DistEngine::numBackwardPasses() const {
  std::lock_guard<std::mutex> guard(initializedContextIdsLock_);
  return initializedContextIds_.size();
}

std::unordered_map<std::string, std::string> DistEngine::getDebugInfo() const {
  std::unordered_map<std::string, std::string> debugInfo;
  debugInfo[kNumBackwardPasses] = std::to_string(numBackwardPasses());
  debugInfo[kEngineCPUQueueSize] =
      std::to_string(engine_.ready_queue_size(at::kCPU));
  debugInfo[kNumAutogradContexts] = std::to_string(
      DistAutogradContainer::getInstance().numAutogradContexts());
  return debugInfo;
}

} // namespace autograd
} // namespace distributed
} // namespace torch

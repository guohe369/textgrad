import argparse
import concurrent
from dotenv import load_dotenv
load_dotenv(override=True)

from tqdm import tqdm
import textgrad as tg
from textgrad.tasks import load_task

import numpy as np
import random

def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

def config():
    parser = argparse.ArgumentParser(description="Optimize a prompt for a task.")
    parser.add_argument("--task", type=str, default="BBH_object_counting", help="The task to evaluate the model on.")
    parser.add_argument("--evaluation_engine", type=str, default="gpt-4o", help="The API to use for evaluation.")
    parser.add_argument("--test_engine", type=str, default="gpt-3.5-turbo-0125", help="The API to use for evaluation.")
    parser.add_argument("--batch_size", type=int, default=3, help="The batch size to use for training.")
    parser.add_argument("--max_epochs", type=int, default=3, help="The maximum number of epochs to train for.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--run_validation", action="store_true", help="Whether to run validation or not.")
    parser.add_argument("--do_not_run_larger_model", action="store_true", help="Whether to run the larger model or not.")
    parser.add_argument("--num_threads", type=int, default=32, help="The number of threads to use for evaluation.")
    return parser.parse_args()

args = config()

def eval_sample(item, eval_fn, model):
    x, y = item
    x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
    y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
    
    try:
        response = model(x)
        
        # Check if this is a stopping response
        if hasattr(response, '_is_stopping_response') and response._is_stopping_response:
            # For stopping responses, treat as incorrect (return 0)
            # This prevents the evaluation from crashing while maintaining the stopping signal
            return 0
        
        # Check for special stopping value
        if response.value == "OPTIMIZATION_STOP":
            return 0
            
        eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
        return int(eval_output_variable.value)
    except Exception as e:
        # Enhanced error handling with logging
        print(f"Error in eval_sample: {str(e)}")
        try:
            eval_output_variable = eval_fn([x, y, response])
            eval_output_parsed = eval_fn.parse_output(eval_output_variable)
            return int(eval_output_parsed)
        except Exception as e2:
            # If all else fails, return 0 to prevent crash
            print(f"Secondary error in eval_sample: {str(e2)}. Returning 0.")
            return 0
    

def eval_dataset(test_set, eval_fn, model, max_samples: int=None):
    if max_samples is None:
        max_samples = len(test_set)
    accuracy_list = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_threads) as executor:
        futures = []
        for _, sample in enumerate(test_set):
            
            future = executor.submit(eval_sample, sample, eval_fn, model)
            futures.append(future)
            if len(futures) >= max_samples:
                break
        tqdm_loader = tqdm(concurrent.futures.as_completed(futures), total=len(futures), position=0)
        for future in tqdm_loader:
            acc_item = future.result()
            accuracy_list.append(acc_item)
            tqdm_loader.set_description(f"Accuracy: {np.mean(accuracy_list)}")
    return accuracy_list 

def run_validation_revert(system_prompt: tg.Variable, results, model, eval_fn, val_set):
    val_performance = np.mean(eval_dataset(val_set, eval_fn, model))
    previous_performance = np.mean(results["validation_acc"][-1])
    print("val_performance: ", val_performance)
    print("previous_performance: ", previous_performance)
    previous_prompt = results["prompt"][-1]
    
    if val_performance < previous_performance:
        print(f"rejected prompt: {system_prompt.value}")
        system_prompt.set_value(previous_prompt)
        val_performance = previous_performance

    results["validation_acc"].append(val_performance)


set_seed(args.seed)
llm_api_eval = tg.get_engine(engine_name=args.evaluation_engine)
llm_api_test = tg.get_engine(engine_name=args.test_engine)
tg.set_backward_engine(llm_api_eval, override=True)

# Load the data and the evaluation function
train_set, val_set, test_set, eval_fn = load_task(args.task, evaluation_api=llm_api_eval)
print("Train/Val/Test Set Lengths: ", len(train_set), len(val_set), len(test_set))
STARTING_SYSTEM_PROMPT = train_set.get_task_description()

train_loader = tg.tasks.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
print(STARTING_SYSTEM_PROMPT)


# Testing the 0-shot performance of the evaluation engine
system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True, 
                            role_description="system prompt to the language model")
model_evaluation = tg.BlackboxLLM(llm_api_eval, system_prompt)

if not args.do_not_run_larger_model:
    reference = np.mean(eval_dataset(test_set, eval_fn, model_evaluation))


system_prompt = tg.Variable(STARTING_SYSTEM_PROMPT, 
                            requires_grad=True,
                            role_description="structured system prompt to a somewhat capable language model that specifies the behavior and strategies for the QA task")
model = tg.BlackboxLLM(llm_api_test, system_prompt)

optimizer = tg.TextualGradientDescent(engine=llm_api_eval, parameters=[system_prompt])

results = {"test_acc": [], "prompt": [], "validation_acc": []}
results["test_acc"].append(eval_dataset(test_set, eval_fn, model))
results["validation_acc"].append(eval_dataset(val_set, eval_fn, model))
results["prompt"].append(system_prompt.get_value())


for epoch in range(args.max_epochs):
    for steps, (batch_x, batch_y) in enumerate((pbar := tqdm(train_loader, position=0))):
        pbar.set_description(f"Training step {steps}. Epoch {epoch}")
        optimizer.zero_grad()
        losses = []
        for (x, y) in zip(batch_x, batch_y):
            x = tg.Variable(x, requires_grad=False, role_description="query to the language model")
            y = tg.Variable(y, requires_grad=False, role_description="correct answer for the query")
            response = model(x)
            try:
                eval_output_variable = eval_fn(inputs=dict(prediction=response, ground_truth_answer=y))
            except:
                eval_output_variable = eval_fn([x, y, response])
            losses.append(eval_output_variable)
        total_loss = tg.sum(losses)
        total_loss.backward()
        optimizer.step()
        if args.run_validation:
            run_validation_revert(system_prompt, results, model, eval_fn, val_set)
        print("sys prompt: ", system_prompt)
        test_acc = eval_dataset(test_set, eval_fn, model)
        results["test_acc"].append(test_acc)
        results["prompt"].append(system_prompt.get_value())
        if steps == 3:
            break

# Also dump the final results
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from textgrad.utils.file_utils import get_figures_path, safe_save_json, log_file_operation

# Save results with automatic directory creation and standardized naming
output_file = get_figures_path(args.task, args.test_engine)
try:
    safe_save_json(results, output_file, indent=2)
    log_file_operation("save_results", output_file, success=True)
    print(f"Results saved to: {output_file}")
except Exception as e:
    log_file_operation("save_results", output_file, success=False)
    print(f"Error saving results to {output_file}: {e}")
    # Fallback: try to save in current directory
    fallback_file = f"results_{args.task}_{args.test_engine}.json"
    try:
        safe_save_json(results, fallback_file, indent=2)
        print(f"Results saved to fallback location: {fallback_file}")
    except Exception as e2:
        print(f"Failed to save results: {e2}")
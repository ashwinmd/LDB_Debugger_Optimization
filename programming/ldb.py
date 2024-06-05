from executors import PyExecutor
from generators import PyGenerator, model_factory
from typing import List
from multiprocessing import Pool
from filelock import FileLock
import random
from transformers import GPT2Tokenizer
from utils import *
import sys
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")

def debug(i, item, log_path, small_model_name, big_model_name, num_items, pass_at_k, max_iters, iters_to_run_small, port="", level = "block"):
    exe = PyExecutor()
    gen = PyGenerator()
    small_model = model_factory(small_model_name, port)
    big_model = model_factory(big_model_name, port)

    use_small_model = True #Will be set to false when you exceed iters_to_run_small
    current_model = small_model

    cur_pass = 0
    is_solved = False
    implementations = []
    test_feedback = []
    cur_func_impl = ""
    dataset_type = item["task_id"].split("/")[0]
    token_nums = 0
    token_nums_small = 0
    token_nums_big = 0
    api_calls = 0
    api_calls_small = 0
    api_calls_big = 0
    while cur_pass < pass_at_k and not is_solved:
        cur_iter = 0
        tests_i = item['given_tests']
        # clean test_i
        tests_i = [test for test in tests_i if item['entry_point'] in test and 'assert False' not in test]
        # first attempt
        cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], item["seed"], item["entry_point"])
        implementations.append(cur_func_impl)
        # call the executor to return failed_test
        is_passing, failed_tests, _ = exe.execute(cur_func_impl, tests_i)
        test_feedback.append(failed_tests)
        # if solved, exit early
        if is_passing:
            is_solved = exe.evaluate(item["entry_point"], cur_func_impl, item["test"], timeout=10)
            break
        # use debug to iteratively improve
        last_func_impl = ""
        if current_model.is_chat:
            messages = []
        else:
            messages = ""
        while cur_iter < max_iters:
            # get self-reflection by debugging a random failed tests
            # The output is 
            # 1. the wrong blocks [wrong block]
            # 2. the explanation [explanation]
            if dataset_type in ["HumanEval", "MBPP"]:
                # Add comments
                if not find_comment(cur_func_impl, item["entry_point"]):
                    debug_cur_func_impl = insert_comment(cur_func_impl, extrace_comment(item["prompt"]), item["entry_point"])
                else:
                    debug_cur_func_impl = cur_func_impl
            elif dataset_type in ["TransCoder"]:
                # Add C++ translation as comments
                debug_cur_func_impl = convert_comment(item["prompt"]) + cur_func_impl
            

            if cur_iter >= iters_to_run_small:
                use_small_model = False
                current_model = big_model
                print("Using big model")
            else:
                print("Using small model")

            selected_test = failed_tests[random.randint(0,len(failed_tests)-1)] if len(failed_tests) >= 1 else None
            generate_function = None

            api_calls += 1 # for below
            if use_small_model: api_calls_small+=1
            else: api_calls_big+=1

            messages = gen.ldb_debug(item["prompt"], debug_cur_func_impl, selected_test, item["entry_point"], current_model, messages, dataset_type, level)

            api_calls += 1 # for below
            if use_small_model: api_calls_small+=1
            else: api_calls_big+=1

            cur_func_impl, cur_messages = gen.ldb_generate(
                func_sig=item["prompt"],
                model=current_model,
                prev_func_impl=cur_func_impl,
                messages=messages,
                failed_tests=selected_test,
                dataset_type=dataset_type)
            
            messages = cur_messages

            tokensToAdd = 0
            if isinstance(messages, str):
                tokensToAdd = len(tokenizer.tokenize(messages))
                
            else:
                tokensToAdd = sum([len(tokenizer.tokenize(msg.content)) for msg in messages])
            
            token_nums += tokensToAdd
            if use_small_model: token_nums_small += tokensToAdd
            else: token_nums_big += tokensToAdd

            cur_func_impl = prepare_function_from_seed(dataset_type, item["prompt"], cur_func_impl, item["entry_point"])
            last_func_impl = cur_func_impl
            implementations.append(cur_func_impl)
            # check if all internal unit tests pass
            is_passing, failed_tests, _ = exe.execute(
                cur_func_impl, tests_i)
            test_feedback.append(failed_tests)
            # if passed, check if it passes the real tests, exit early
            if is_passing or cur_iter == max_iters - 1:
                if is_passing:
                    print(f'{item["task_id"]} pass generated tests, check real tests')
                else:
                    print(f'{item["task_id"]} fail generated tests, check real tests')
                is_solved = exe.evaluate(
                    item["entry_point"], cur_func_impl, item["test"], timeout=10)
                if is_solved:
                    item["solution"] = cur_func_impl
                cur_iter += 1
                sys.stdout.flush()
                break
            cur_iter += 1
            sys.stdout.flush()
        cur_pass += 1
    item["is_passing"] = is_passing
    item["is_solved"] = is_solved
    item["implementations"] = implementations
    item["test_feedback"] = test_feedback
    item["solution"] = cur_func_impl
    item["generated_test"] = tests_i
    item["debug_iter"] = cur_iter
    item["token_nums"] = token_nums
    item["token_nums_small_model"] = token_nums_small
    item["token_nums_big_model"] = token_nums_big
    item["api_calls"] = api_calls
    item["api_calls_small_model"] = api_calls_small
    item["api_calls_big_model"] = api_calls_big
    with FileLock(log_path + ".lock"):
        write_jsonl(log_path, [item], append=True)
    print(f'completed {i+1}/{num_items}')

def run_ldb(
    dataset: List[dict],
    small_model_name: str,
    big_model_name: str,
    max_iters: int,
    iters_to_run_small: int,
    n_proc: int,
    pass_at_k: int,
    log_path: str,
    verbose: bool,
    seedfile: str = None,
    testfile: str = None,
    port: str = "",
    level: str = "block"
) -> None:
    print("Number of proc:", n_proc)
    num_items = len(dataset)
    args = iter([(i, item, log_path, small_model_name, big_model_name, num_items, pass_at_k, max_iters, iters_to_run_small, port, level) for i, item in enumerate_resume(dataset, log_path, seedfile, testfile)])
    if n_proc == 1:
        for item in args:
            debug(*item)
    else:
        with Pool(n_proc) as pool:
            pool.starmap(debug, args)
    print("Accuracy:", count_solved(log_path))
    
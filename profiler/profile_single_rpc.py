import argparse
import datetime
import time

from multi_host_main import main

from profiler.utils import find_factors

if __name__ == "__main__":
    date = datetime.datetime.now().strftime("%Y%m%d")
    expr_names = []
    for size in [7, 13, 34, 70]:
        if size == 7:
            n_nodes = 1
        elif size == 13:
            n_nodes = 2
        elif size == 34:
            n_nodes = 4
        elif size == 70:
            n_nodes = 8

        num_gpus = n_nodes * 8
        for num_mp in [1, 2, 4, 8]:
            remain = num_gpus // num_mp
            for num_dp in find_factors(remain):
                num_pp = remain // num_dp
                expr_names.append(f"profile-s{size}p{num_pp}m{num_mp}d{num_dp}")

    for expr_name in expr_names:
        st = time.monotonic()
        print(f"running expr_name: {expr_name} at {date}")
        args = argparse.Namespace()
        setattr(args, "expr_name", expr_name)
        setattr(args, "trial_name", date)
        error = main(args, if_raise=False)
        print(f"expr_name: {expr_name} at {date} done, error: {error}, "
              f"timecost {time.monotonic() - st:.2f}")

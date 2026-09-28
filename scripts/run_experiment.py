"""Run several DQN variants x seeds in parallel, grouped as one experiment for scripts/plot.py.

Presets:
    add-one      DQN, and DQN + each single improvement       ("what does X add on its own?")
    cumulative   DQN, +double, +double+dueling, ... Rainbow     (matches the implementation order)
    ablation     Rainbow, and Rainbow without each improvement   ("what does Rainbow lose without X?",
                                                                  like Fig. 3 of the Rainbow paper)
    custom       --variants "dqn" "double,dueling" "mine=--double --lr 1e-3"
                 (component list, or name=flags)

Examples:
    # only the variants you have implemented so far; finished runs are skipped, so re-run later with more
    python scripts/run_experiment.py --experiment lunar-add-one --preset add-one --only dqn dqn+double \\
        --env-id LunarLander-v3 --total-steps 300000 --seeds 5
    python scripts/plot.py runs/lunar-add-one

Arguments after `--` go to every run, e.g.  ... -- --lr 1e-4 --buffer-size 100000
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

COMPONENTS = {
    "double": ["--double"],
    "dueling": ["--dueling"],
    "nstep": ["--n-step", "3"],
    "per": ["--per"],
    "noisy": ["--noisy"],
    "c51": ["--c51"],
}


def flags_for(components):
    return [f for c in components for f in COMPONENTS[c]]


def preset_variants(preset, custom=None):
    """Returns an ordered dict: variant name -> list of extra CLI flags."""
    names = list(COMPONENTS)
    if preset == "add-one":
        return {"dqn": [], **{f"dqn+{c}": COMPONENTS[c] for c in names}}
    if preset == "cumulative":
        return {"dqn": [], **{"dqn+" + "+".join(names[: i + 1]): flags_for(names[: i + 1])
                              for i in range(len(names))}}
    if preset == "ablation":
        return {"rainbow": flags_for(names),
                **{f"rainbow-no-{c}": flags_for([n for n in names if n != c]) for c in names}}
    if preset == "custom":
        variants = {}
        for spec in custom or []:
            if "=" in spec:
                name, flags = spec.split("=", 1)
                variants[name] = flags.split()
            else:
                comps = [c for c in spec.split(",") if c and c != "dqn"]
                unknown = set(comps) - set(COMPONENTS)
                if unknown:
                    sys.exit(f"Unknown components {unknown}; choose from {list(COMPONENTS)}")
                variants["+".join(["dqn", *comps])] = flags_for(comps)
        return variants
    sys.exit(f"Unknown preset {preset}")


def run_one(cmd, log_path):
    env = {**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}  # 1 thread per run: no oversubscription
    start = time.time()
    with open(log_path, "w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    return proc.returncode, time.time() - start


def main():
    argv = sys.argv[1:]
    passthrough = []
    if "--" in argv:
        i = argv.index("--")
        argv, passthrough = argv[:i], argv[i + 1:]

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--experiment", required=True, help="results go to runs/<experiment>/")
    p.add_argument("--preset", default="add-one", choices=["add-one", "cumulative", "ablation", "custom"])
    p.add_argument("--variants", nargs="*", help="for --preset custom")
    p.add_argument("--only", nargs="*", help="run only these variant names from the preset")
    p.add_argument("--algo", default="dqn", help="module in rl/algos/")
    p.add_argument("--env-id", default="CartPole-v1")
    p.add_argument("--total-steps", type=int, default=100_000)
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..N-1)")
    p.add_argument("--workers", type=int, default=os.cpu_count())
    p.add_argument("--rerun", action="store_true", help="re-run runs that already finished")
    p.add_argument("--dry-run", action="store_true", help="only print the commands")
    args = p.parse_args(argv)

    variants = preset_variants(args.preset, args.variants)
    if args.only:
        missing = set(args.only) - set(variants)
        if missing:
            sys.exit(f"--only: {missing} not in preset. Available: {list(variants)}")
        variants = {k: v for k, v in variants.items() if k in args.only}

    exp_dir = os.path.join(ROOT, "runs", args.experiment)
    log_dir = os.path.join(exp_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Remember the variant order (plots use it for ordering and stable colors)
    order_path = os.path.join(exp_dir, "variants.json")
    order = json.load(open(order_path)) if os.path.exists(order_path) else []
    order += [v for v in variants if v not in order]
    json.dump(order, open(order_path, "w"), indent=2)

    jobs = []
    for name, flags in variants.items():
        for seed in range(args.seeds):
            if not args.rerun and os.path.exists(os.path.join(exp_dir, name, f"s{seed}", "model.pt")):
                print(f"skip (done): {name} s{seed}")
                continue
            cmd = [sys.executable, "-m", f"rl.algos.{args.algo}", "--env-id", args.env_id,
                   "--seed", str(seed), "--total-steps", str(args.total_steps),
                   "--experiment", args.experiment, "--name", name, *flags, *passthrough]
            jobs.append((name, seed, cmd, os.path.join(log_dir, f"{name}_s{seed}.log")))

    print(f"{len(jobs)} runs, {args.workers} in parallel -> {exp_dir}")
    if args.dry_run:
        for _, _, cmd, _ in jobs:
            print(" ".join(cmd[1:]))
        return

    failed = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(run_one, cmd, log): (name, seed, log) for name, seed, cmd, log in jobs}
        for fut in as_completed(futures):
            name, seed, log = futures[fut]
            code, secs = fut.result()
            status = "ok" if code == 0 else f"FAILED (exit {code})"
            print(f"{status:>16}  {name} s{seed}  ({secs / 60:.1f} min)  log: {os.path.relpath(log, ROOT)}", flush=True)
            if code != 0:
                failed.append(log)

    if failed:
        print(f"\n{len(failed)} run(s) failed. Last lines of {os.path.relpath(failed[0], ROOT)}:")
        print("".join(open(failed[0]).readlines()[-15:]))
        sys.exit(1)
    print(f"\nDone. Plot with:  python scripts/plot.py runs/{args.experiment}")


if __name__ == "__main__":
    main()

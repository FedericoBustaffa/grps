import argparse
import json

import pandas as pd

from grps import RPSModel
from grps import evolution_policies as evo

policy_mapping = {
    "inheritance": evo.Inheritance,
    "stochastic": evo.Stochastic,
    "genetic": evo.Genetic,
}


def get_policy(params: dict, idx: int) -> evo.EvolutionPolicy:
    p = params["policies"][idx]
    if p == "inheritance":
        return evo.Inheritance()
    elif p == "stochastic":
        return evo.Stochastic(sigma=params["sigma"])
    else:
        return evo.Genetic(sigma=params["sigma"], radius=params["radius"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="config file for the simulation")
    parser.add_argument("seeds", type=int, help="number of simulation seeds")
    args = parser.parse_args()

    fp = open(args.config, "r")
    params = json.load(fp)
    fp.close()

    all_runs = []
    print(json.dumps(params, indent=2))
    policies = {}
    policies["rock"] = get_policy(params, 0)
    policies["paper"] = get_policy(params, 1)
    policies["scissors"] = get_policy(params, 2)

    filename = args.config.split("/")[1].split(".")[0]
    for seed in range(args.seeds):
        model = RPSModel(
            dim=params["grid_dim"],
            policies=policies,
            initial_invasions=params["invasions"],
            rng=seed,
        )
        model.run_for(params["epochs"], filename)

        model_df = model.datacollector.get_model_vars_dataframe()

        model_df["seed"] = seed
        model_df["grid_dim"] = params["grid_dim"]

        model_df["R_init_invasion"] = params["invasions"][0]
        model_df["P_init_invasion"] = params["invasions"][1]
        model_df["S_init_invasion"] = params["invasions"][2]

        model_df["R_policy"] = params["policies"][0]
        model_df["P_policy"] = params["policies"][1]
        model_df["S_policy"] = params["policies"][2]

        if "sigma" in params.keys():
            model_df["sigma"] = params["sigma"]
        if "radius" in params.keys():
            model_df["radius"] = params["radius"]
        all_runs.append(model_df)

    results = pd.concat(all_runs, ignore_index=True)
    filename = args.config.split("/")[1].split(".")[0]
    # results.to_csv(f"results/{filename}.csv", index=False, header=True)

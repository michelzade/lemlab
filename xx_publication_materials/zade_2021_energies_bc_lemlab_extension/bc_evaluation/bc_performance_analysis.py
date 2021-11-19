import os
import time
import json
import test_utils
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from lemlab.lem import clearing_ex_ante
from bc_test_settlement import test_meter_info, test_user_info, test_meter_readings, test_prices_settlement, \
    test_transaction_logs, test_balancing_energy, test_clearing_results_ex_ante


def compute_performance_analysis(path_results=None):
    print(f"Performance analysis started...")
    config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = test_utils.setup_test_general(
        generate_random_test_data=False, test_config_path="sim_test_config.yaml")

    if path_results is not None:
        with open(f"{path_results}/evaluation_config.json", "w+") as write_file:
            json.dump(config, write_file)

    # Clear all tables
    test_utils.reset_all_market_tables(db_obj=db_obj, bc_obj_market=bc_obj_clearing_ex_ante,
                                       bc_obj_settlement=bc_obj_settlement)

    # Insert users to db and bc with grid meters for the settlement
    ids_users, ids_meters, ids_market_agents = test_utils.setup_random_prosumers(db_obj=db_obj,
                                                                                 bc_obj_market=bc_obj_clearing_ex_ante,
                                                                                 n_prosumers=
                                                                                 config["bc_performance_analysis"][
                                                                                     "n_prosumers"])

    # Retrieve range of positions that shall be simulated
    n_positions_range = np.arange(config["bc_performance_analysis"]["n_positions"]["min"],
                                  config["bc_performance_analysis"]["n_positions"]["max"],
                                  config["bc_performance_analysis"]["n_positions"]["increment"])

    # Result dictionary
    result_dict = dict()
    result_dict["timing"] = dict()
    result_dict["exception"] = dict()
    result_dict["exception"]["db"] = dict()
    result_dict["exception"]["bc"] = dict()
    result_dict["timing"]["db"] = dict()
    result_dict["timing"]["bc"] = dict()
    result_dict["timing"]["db"]["post_positions"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["db"]["clear_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["db"]["log_meter_readings"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["db"]["settle_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["db"]["full_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["bc"]["post_positions"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["bc"]["clear_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["bc"]["log_meter_readings"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["bc"]["settle_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["timing"]["bc"]["full_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["gas_consumption"] = dict()
    result_dict["gas_consumption"]["post_positions"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["gas_consumption"]["clear_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["gas_consumption"]["log_meter_readings"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["gas_consumption"]["settle_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["gas_consumption"]["full_market"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"] = dict()
    result_dict["equality_check"]["user_info"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"]["meter_info"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"]["market_results"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"]["prices_settlement"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"]["transaction_logs"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"]["balancing_energy"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))
    result_dict["equality_check"]["meter_readings"] = pd.DataFrame(
        index=range(config["bc_performance_analysis"]["n_samples"]))

    print(f"Setup complete.")
    for sample in range(config["bc_performance_analysis"]["n_samples"]):
        print(f"### Sample No. {sample} ###")
        # Loop through all position samples and execute market clearing
        for n_positions in n_positions_range:
            print(f"Test with {n_positions} positions.")
            # Reset market tables before each iteration
            test_utils.reset_dynamic_market_data_tables(db_obj=db_obj, bc_obj_market=bc_obj_clearing_ex_ante,
                                                        bc_obj_settlement=bc_obj_settlement)
            # Compute random market positions
            positions = test_utils.create_random_positions(db_obj=db_obj,
                                                           config=config,
                                                           ids_user=ids_meters,
                                                           n_positions=n_positions,
                                                           verbose=False)

            # Initialize clearing parameters
            config_retailer = None
            t_override = round(time.time())
            shuffle = False
            plotting = False
            verbose = False
            sim_path = ""

            #############################
            # Central database LEM ######
            #############################
            try:
                # Post positions ###
                t_post_positions_start_db = time.time()
                db_obj.post_positions(positions)
                t_post_positions_end_db = time.time()
                t_post_positions_db = t_post_positions_end_db - t_post_positions_start_db
                result_dict["timing"]["db"]["post_positions"].loc[sample, n_positions] = t_post_positions_db
                # Clear market ex ante ###
                t_clear_market_start_db = time.time()
                clearing_ex_ante.market_clearing(db_obj=db_obj, config_lem=config["lem"],
                                                 config_retailer=config_retailer,
                                                 t_override=t_override, plotting=plotting, verbose=verbose,
                                                 rounding_method=False)
                t_clear_market_end_db = time.time()
                t_clear_market_db = t_clear_market_end_db - t_clear_market_start_db
                result_dict["timing"]["db"]["clear_market"].loc[sample, n_positions] = t_clear_market_db
                # Simulate meter readings from market results with random errors for settlement
                simulated_meter_readings_delta, ts_delivery_list = test_utils.simulate_meter_readings_from_market_results(
                    db_obj=db_obj, rand_percent_var=15)
                # Log meter readings to LEM ###
                t_log_meter_readings_start_db = time.time()
                db_obj.log_readings_meter_delta(simulated_meter_readings_delta)
                t_log_meter_readings_end_db = time.time()
                t_log_meter_readings_db = t_log_meter_readings_end_db - t_log_meter_readings_start_db
                result_dict["timing"]["db"]["log_meter_readings"].loc[sample, n_positions] = t_log_meter_readings_db
                # Settle market ###
                t_settle_market_start_db = time.time()
                test_utils.settle_market_db(config=config, db_obj=db_obj, ts_delivery_list=ts_delivery_list,
                                            path_sim=sim_path)
                t_settle_market_end_db = time.time()
                t_settle_market_db = t_settle_market_end_db - t_settle_market_start_db
                result_dict["timing"]["db"]["settle_market"].loc[sample, n_positions] = t_settle_market_db
                # Full computation time ###
                t_full_market_db = t_post_positions_db + t_clear_market_db + t_log_meter_readings_db + t_settle_market_db
                result_dict["timing"]["db"]["full_market"].loc[sample, n_positions] = t_full_market_db
                print(f"Central LEM successfully cleared {n_positions} positions in {t_full_market_db} s.")
            except Exception as e:
                print(e)
                result_dict["exception"]["db"][sample] = e
                break

            #############################
            # Blockchain LEM ############
            #############################
            try:
                # convert energy qualities from string to int
                positions = test_utils._convert_qualities_to_int(db_obj, positions, config['lem']['types_quality'])
                t_post_positions_start_bc = time.time()
                _, gas_consumption_pp = bc_obj_clearing_ex_ante.push_all_positions(
                    positions, temporary=True, permanent=False)
                result_dict["gas_consumption"]["post_positions"].loc[sample, n_positions] = gas_consumption_pp
                t_post_positions_end_bc = time.time()
                t_post_positions_bc = t_post_positions_end_bc - t_post_positions_start_bc
                result_dict["timing"]["bc"]["post_positions"].loc[sample, n_positions] = t_post_positions_bc
                # Clear market ex ante ###
                t_clear_market_start_bc = time.time()
                gas_consumption_cm = bc_obj_clearing_ex_ante.market_clearing_ex_ante(
                    config["lem"], config_retailer=config_retailer,
                    t_override=t_override, shuffle=shuffle, verbose=verbose)
                result_dict["gas_consumption"]["clear_market"].loc[sample, n_positions] = gas_consumption_cm
                t_clear_market_end_bc = time.time()
                t_clear_market_bc = t_clear_market_end_bc - t_clear_market_start_bc
                result_dict["timing"]["bc"]["clear_market"].loc[sample, n_positions] = t_clear_market_bc
                # Log meter readings to LEM ###
                t_log_meter_readings_start_bc = time.time()
                _, gas_consumption_lmr = bc_obj_settlement.log_meter_readings_delta(simulated_meter_readings_delta)
                result_dict["gas_consumption"]["log_meter_readings"].loc[sample, n_positions] = gas_consumption_lmr
                t_log_meter_readings_end_bc = time.time()
                t_log_meter_readings_bc = t_log_meter_readings_end_bc - t_log_meter_readings_start_bc
                result_dict["timing"]["bc"]["log_meter_readings"].loc[sample, n_positions] = t_log_meter_readings_bc
                # Settle market ###
                t_settle_market_start_bc = time.time()
                gas_consumption_sm = test_utils.settle_market_bc(config=config,
                                                                 bc_obj_settlement=bc_obj_settlement,
                                                                 ts_delivery_list=ts_delivery_list)
                result_dict["gas_consumption"]["settle_market"].loc[sample, n_positions] = gas_consumption_sm
                t_settle_market_end_bc = time.time()
                t_settle_market_bc = t_settle_market_end_bc - t_settle_market_start_bc
                result_dict["timing"]["bc"]["settle_market"].loc[sample, n_positions] = t_settle_market_bc
                # Full market clearing ###
                t_full_market_bc = t_post_positions_bc + t_clear_market_bc + t_log_meter_readings_bc + t_settle_market_bc
                gas_consumption_all = gas_consumption_pp + gas_consumption_cm + gas_consumption_lmr + gas_consumption_sm
                result_dict["gas_consumption"]["full_market"].loc[sample, n_positions] = gas_consumption_all
                result_dict["timing"]["bc"]["full_market"].loc[sample, n_positions] = t_full_market_bc
                print(f"Blockchain LEM successfully cleared {n_positions} positions in {t_full_market_bc} s.")
            except Exception as e:
                print(e)
                result_dict["exception"]["bc"][sample] = e
                break

            # Check equality of db and bc entries
            try:
                test_user_info(db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante)
                result_dict["equality_check"]["user_info"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["user_info"].loc[sample, n_positions] = diff
                pass

            try:
                test_meter_info(db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante)
                result_dict["equality_check"]["meter_info"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["meter_info"].loc[sample, n_positions] = diff
                pass

            try:
                test_clearing_results_ex_ante(db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante)
                result_dict["equality_check"]["market_results"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["market_results"].loc[sample, n_positions] = diff
                pass

            try:
                test_meter_readings(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
                result_dict["equality_check"]["meter_readings"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["meter_readings"].loc[sample, n_positions] = diff
                pass

            try:
                test_prices_settlement(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
                result_dict["equality_check"]["prices_settlement"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["prices_settlement"].loc[sample, n_positions] = diff
                pass

            try:
                test_balancing_energy(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
                result_dict["equality_check"]["balancing_energy"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["balancing_energy"].loc[sample, n_positions] = diff
                pass

            try:
                test_transaction_logs(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
                result_dict["equality_check"]["transaction_logs"].loc[sample, n_positions] = 0
            except AssertionError as e:
                diff_str = str(e).split("\n")[2].split("(")[-1].split(")")[0]
                diff = float(diff_str[:-2])
                result_dict["equality_check"]["transaction_logs"].loc[sample, n_positions] = diff
                pass

            # Plot results after each iteration
            plot_time_complexity_analysis(db_timings=result_dict["timing"]["db"],
                                          bc_timings=result_dict["timing"]["bc"],
                                          path_results=path_results)
            plot_time_complexity_distributions(db_timings=result_dict["timing"]["db"],
                                               bc_timings=result_dict["timing"]["bc"],
                                               path_results=path_results)
            plot_gas_consumption(gas_consumption_dict=result_dict["gas_consumption"], path_results=path_results)
            plot_equality_check(result_dict["equality_check"], path_results=path_results)

            save_results(result_dict=result_dict, path_results=path_results)

    return result_dict


def plot_time_complexity_analysis(db_timings, bc_timings, path_results=None, only_full_market=False, reference=None):
    reference_value = 1
    fig = plt.figure()
    ax = plt.subplot(111)
    if only_full_market:
        key = "full_market"
        df = db_timings["full_market"]
        if reference:
            reference_value = df.iloc[:, 0].mean()
        y_error = [df.max() - df.mean(), df.mean() - df.min()]
        ax.errorbar(x=df.columns, y=df.mean() / reference_value, yerr=[e / reference_value for e in y_error],
                    marker="x",
                    label="db: " + key.replace("_", " "))
        df = bc_timings["full_market"]
        y_error = [df.max() - df.mean(), df.mean() - df.min()]
        ax.errorbar(x=df.columns, y=df.mean() / reference_value, yerr=[e / reference_value for e in y_error],
                    marker="x",
                    label="bc: " + key.replace("_", " "))
    else:
        if reference:
            reference_value = db_timings["post_positions"].iloc[:, 0].mean()
        for key, df in db_timings.items():
            y_error = [df.max() - df.mean(), df.mean() - df.min()]
            ax.errorbar(x=df.columns, y=df.mean() / reference_value, yerr=[e / reference_value for e in y_error],
                        marker="x", label="db: " + key.replace("_", " "))
        for key, df in bc_timings.items():
            y_error = [df.max() - df.mean(), df.mean() - df.min()]
            ax.errorbar(x=df.columns, y=df.mean() / reference_value, yerr=[e / reference_value for e in y_error],
                        marker="x", label="bc: " + key.replace("_", " "))
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("Computational effort in relation to a full market clearing\nwith 50 bids on a central database")
    ax.set_xlabel("Number of inserted buy and ask bids")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(0.15, 1.05), frameon=False, ncol=2)
    if path_results is not None:
        fig.savefig(f"{path_results}/timing_results.svg")
    plt.show()


def plot_time_complexity_distributions(db_timings, bc_timings, path_results=None):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    norm = db_timings["full_market"].mean() / 100
    l1 = ax1.stackplot(db_timings["full_market"].columns,
                       db_timings["post_positions"].mean() / norm,
                       db_timings["clear_market"].mean() / norm,
                       db_timings["log_meter_readings"].mean() / norm,
                       db_timings["settle_market"].mean() / norm,
                       labels=["post_positions", "clear_market", "log_meter_readings", "settle_market"]
                       )
    ax1.set_ylabel("Share of computation effort in %")
    ax1.set_xlabel("Number of inserted buy and ask bids")
    ax1.set_title("Central database LEM")

    norm = bc_timings["full_market"].mean() / 100
    l2 = ax2.stackplot(bc_timings["full_market"].columns,
                       bc_timings["post_positions"].mean() / norm,
                       bc_timings["clear_market"].mean() / norm,
                       bc_timings["log_meter_readings"].mean() / norm,
                       bc_timings["settle_market"].mean() / norm
                       )
    ax2.set_xlabel("Number of inserted buy and ask bids")
    ax2.set_title("Blockchain LEM")
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0, box.width, box.height * 0.9])
    # Put a legend to the right of the current axis
    # ax.legend(loc='center left', bbox_to_anchor=(0.15, 1.05), frameon=False, ncol=2)

    fig.legend(l1, labels=["post_positions", "clear_market", "log_meter_readings", "settle_market"],
               loc="center left", frameon=False, bbox_to_anchor=(0.2, 0.05), ncol=4)
    plt.subplots_adjust(bottom=0.18)
    if path_results is not None:
        fig.savefig(f"{path_results}/computation_distribution.svg")
    plt.show()


def plot_gas_consumption(gas_consumption_dict, path_results=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    h = [ax.errorbar(x=df.columns, y=df.mean(), yerr=[df.max() - df.mean(), df.mean() - df.min()], marker="x",
                     label=key.replace("_", " ")) for key, df in gas_consumption_dict.items()]
    ax.grid()
    ax.set_ylabel("Gas consumption")
    ax.set_xlabel("Number of inserted buy and ask bids")
    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height*0.95])
    n_col = 3
    n_entries = 5
    leg1 = ax.legend(handles=h[:n_entries // n_col * n_col], ncol=n_col, loc="center left", frameon=False,
                     bbox_to_anchor=(0, 1.1))
    plt.gca().add_artist(leg1)
    leg2 = ax.legend(handles=h[n_entries // n_col * n_col:], ncol=n_entries - n_entries // n_col * n_col)
    leg2.remove()
    leg1._legend_box._children.append(leg2._legend_handle_box)
    leg1._legend_box.stale = True
    if path_results is not None:
        fig.savefig(f"{path_results}/gas_consumption.svg")
    plt.show()


def plot_equality_check(equality_check_dict, path_results=None):
    for key, df in equality_check_dict.items():
        svm = sns.heatmap(df, annot=True, vmin=0, vmax=100, cmap="RdYlGn_r", cbar_kws={"label": "Percentage deviation"})
        plt.xlabel("Number of inserted bids")
        plt.ylabel("Sample number")
        plt.title(key)
        if path_results is not None:
            fig = svm.get_figure()
            fig.savefig(f"{path_results}/equality_check_{key}.svg")
        plt.show()


def create_results_folder(path_results):
    current_time_str = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(f"{path_results}/{current_time_str}")
    os.mkdir(f"{path_results}/{current_time_str}/timing")
    os.mkdir(f"{path_results}/{current_time_str}/timing/db")
    os.mkdir(f"{path_results}/{current_time_str}/timing/bc")
    os.mkdir(f"{path_results}/{current_time_str}/gas_consumption")
    os.mkdir(f"{path_results}/{current_time_str}/equality_check")
    os.mkdir(f"{path_results}/{current_time_str}/exception")

    return f"{path_results}/{current_time_str}"


def save_results(result_dict, path_results):
    for key, df in result_dict["timing"]["db"].items():
        df.to_csv(f"{path_results}/timing/db/{key}.csv")
    for key, df in result_dict["timing"]["bc"].items():
        df.to_csv(f"{path_results}/timing/bc/{key}.csv")
    for key, df in result_dict["gas_consumption"].items():
        df.to_csv(f"{path_results}/gas_consumption/{key}.csv")
    for key, df in result_dict["equality_check"].items():
        df.to_csv(f"{path_results}/equality_check/{key}.csv")


def load_results(path_result_folder):
    result_dict = dict()
    result_dict["equality_check"] = dict()
    for file_name in os.listdir(f"{path_result_folder}/equality_check"):
        result_dict["equality_check"][file_name[:-4]] = pd.read_csv(f"{path_result_folder}/equality_check/{file_name}",
                                                                    index_col=0)
    result_dict["timing"] = dict()
    result_dict["timing"]["db"] = dict()
    result_dict["timing"]["bc"] = dict()
    for file_name in os.listdir(f"{path_result_folder}/timing/db"):
        result_dict["timing"]["db"][file_name[:-4]] = pd.read_csv(f"{path_result_folder}/timing/db/{file_name}",
                                                                  index_col=0)
    for file_name in os.listdir(f"{path_result_folder}/timing/bc"):
        result_dict["timing"]["bc"][file_name[:-4]] = pd.read_csv(f"{path_result_folder}/timing/bc/{file_name}",
                                                                  index_col=0)
    result_dict["gas_consumption"] = dict()
    for file_name in os.listdir(f"{path_result_folder}/gas_consumption"):
        result_dict["gas_consumption"][file_name[:-4]] = pd.read_csv(
            f"{path_result_folder}/gas_consumption/{file_name}",
            index_col=0)

    return result_dict


if __name__ == '__main__':
    # Create result folder
    path_result_folder = create_results_folder(path_results="evaluation_results")
    # Compute performance analysis
    results = compute_performance_analysis(path_results=path_result_folder)
    # Save results to files
    save_results(results, path_result_folder)
    # path_result_folder = "evaluation_results/2021_11_17_15_30_52"
    # Load data from previous analysis
    results = load_results(path_result_folder=path_result_folder)
    # Plot results
    plot_time_complexity_analysis(results["timing"]["db"], results["timing"]["bc"], path_results=path_result_folder,
                                  only_full_market=True, reference=True)
    plot_time_complexity_distributions(db_timings=results["timing"]["db"], bc_timings=results["timing"]["bc"],
                                       path_results=path_result_folder)
    plot_gas_consumption(results["gas_consumption"], path_results=path_result_folder)
    plot_equality_check(results["equality_check"], path_results=path_result_folder)

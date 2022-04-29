import os
import time
import yaml
import json
import test_utils
import tikzplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from telegram_connector_class import SenderTelegram
from lemlab.lem import clearing_ex_ante
from bc_test_settlement import test_meter_info, test_user_info, test_meter_readings, test_prices_settlement, \
    test_transaction_logs, test_balancing_energy, test_clearing_results_ex_ante
from xx_publication_materials.zade_2022_iewt_bc_extension.bc_connection.bc_node \
    import kill_process_by_name, start_openethereum_node, restart_openethereum_node


def compute_performance_analysis(path_results=None, exception_handler=None, path_openethereum_toml=None):
    try:
        # Start openethereum node
        start_openethereum_node(path_toml_file=path_openethereum_toml)
        print(f"Performance analysis started...")
        config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = test_utils.setup_test_general(
            generate_random_test_data=False, test_config_path="sim_test_config.yaml")

        if path_results is not None:
            with open(f"{path_results}/evaluation_config.json", "w+") as write_file:
                json.dump(config, write_file)

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
            # Loop through all position samples and execute market clearing
            for n_positions in n_positions_range:
                success = False
                n_restarts = 0
                max_n_restarts = config["bc_performance_analysis"]["max_n_restart_node"]
                while not success:
                    try:
                        print(f"### Sample #{sample} with {n_positions} positions.")
                        # Clear all tables
                        test_utils.reset_all_market_tables(db_obj=db_obj, bc_obj_market=bc_obj_clearing_ex_ante,
                                                           bc_obj_settlement=bc_obj_settlement)

                        # Insert users to db and bc with grid meters for settlement
                        ids_users, ids_meters, ids_market_agents = \
                            test_utils.setup_random_prosumers(db_obj=db_obj,
                                                              bc_obj_market=bc_obj_clearing_ex_ante,
                                                              n_prosumers=
                                                              config["bc_performance_analysis"]["n_prosumers"])
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
                        simulated_meter_readings_delta, ts_delivery_list = \
                            test_utils.simulate_meter_readings_from_market_results(
                                db_obj=db_obj, rand_percent_var=15)
                        # Log meter readings to LEM ###
                        t_log_meter_readings_start_db = time.time()
                        db_obj.log_readings_meter_delta(simulated_meter_readings_delta)
                        t_log_meter_readings_end_db = time.time()
                        t_log_meter_readings_db = t_log_meter_readings_end_db - t_log_meter_readings_start_db
                        result_dict["timing"]["db"]["log_meter_readings"].loc[
                            sample, n_positions] = t_log_meter_readings_db
                        # Settle market ###
                        t_settle_market_start_db = time.time()
                        test_utils.settle_market_db(config=config, db_obj=db_obj, ts_delivery_list=ts_delivery_list,
                                                    path_sim=sim_path)
                        t_settle_market_end_db = time.time()
                        t_settle_market_db = t_settle_market_end_db - t_settle_market_start_db
                        result_dict["timing"]["db"]["settle_market"].loc[sample, n_positions] = t_settle_market_db
                        # Full computation time ###
                        t_full_market_db = t_post_positions_db + t_clear_market_db + t_log_meter_readings_db + \
                                           t_settle_market_db
                        result_dict["timing"]["db"]["full_market"].loc[sample, n_positions] = t_full_market_db
                        print(f"Central LEM successfully cleared {n_positions} positions in {t_full_market_db} s.")

                        #############################
                        # Blockchain LEM ############
                        #############################
                        # convert energy qualities from string to int
                        positions = test_utils._convert_qualities_to_int(db_obj, positions,
                                                                         config['lem']['types_quality'])
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
                        _, gas_consumption_lmr = bc_obj_settlement.log_meter_readings_delta(
                            simulated_meter_readings_delta)
                        result_dict["gas_consumption"]["log_meter_readings"].loc[
                            sample, n_positions] = gas_consumption_lmr
                        t_log_meter_readings_end_bc = time.time()
                        t_log_meter_readings_bc = t_log_meter_readings_end_bc - t_log_meter_readings_start_bc
                        result_dict["timing"]["bc"]["log_meter_readings"].loc[
                            sample, n_positions] = t_log_meter_readings_bc
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
                        t_full_market_bc = t_post_positions_bc + t_clear_market_bc + t_log_meter_readings_bc + \
                                           t_settle_market_bc
                        gas_consumption_all = gas_consumption_pp + gas_consumption_cm + gas_consumption_lmr + \
                                              gas_consumption_sm
                        result_dict["gas_consumption"]["full_market"].loc[sample, n_positions] = gas_consumption_all
                        result_dict["timing"]["bc"]["full_market"].loc[sample, n_positions] = t_full_market_bc
                        print(f"Blockchain LEM successfully cleared {n_positions} positions in {t_full_market_bc} s.")
                        success = True

                    except Exception as e:
                        # Restart openethereum node
                        restart_openethereum_node(path_toml_file=path_openethereum_toml)
                        n_restarts += 1
                        if n_restarts >= max_n_restarts:
                            raise e

                # Check equality of db and bc entries
                result_dict["equality_check"]["user_info"].loc[sample, n_positions] = test_user_info(
                    db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante, n_sample=sample,
                    n_positions=n_positions, path_results=path_results)

                result_dict["equality_check"]["meter_info"].loc[sample, n_positions] = test_meter_info(
                    db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante, n_sample=sample,
                    n_positions=n_positions, path_results=path_results)

                result_dict["equality_check"]["market_results"].loc[
                    sample, n_positions] = \
                    test_clearing_results_ex_ante(db_obj=db_obj,
                                                  bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante,
                                                  n_sample=sample, n_positions=n_positions,
                                                  path_results=path_results)

                result_dict["equality_check"]["meter_readings"].loc[sample, n_positions] = test_meter_readings(
                    db_obj=db_obj, bc_obj_settlement=bc_obj_settlement,
                    n_sample=sample, n_positions=n_positions, path_results=path_results)

                result_dict["equality_check"]["prices_settlement"].loc[sample, n_positions] = test_prices_settlement(
                    db_obj=db_obj, bc_obj_settlement=bc_obj_settlement,
                    n_sample=sample, n_positions=n_positions, path_results=path_results)

                result_dict["equality_check"]["balancing_energy"].loc[sample, n_positions] = test_balancing_energy(
                    db_obj=db_obj, bc_obj_settlement=bc_obj_settlement,
                    bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante, n_sample=sample, n_positions=n_positions,
                    path_results=path_results)

                result_dict["equality_check"]["transaction_logs"].loc[sample, n_positions] = test_transaction_logs(
                    db_obj=db_obj, bc_obj_settlement=bc_obj_settlement,
                    n_sample=sample, n_positions=n_positions, path_results=path_results)

                # Plot results after each iteration
                plot_time_complexity_analysis(db_timings=result_dict["timing"]["db"],
                                              bc_timings=result_dict["timing"]["bc"],
                                              path_results=path_results, show=False)
                plot_time_complexity_distributions(db_timings=result_dict["timing"]["db"],
                                                   bc_timings=result_dict["timing"]["bc"],
                                                   path_results=path_results, show=False)
                plot_gas_consumption(gas_consumption_dict=result_dict["gas_consumption"], path_results=path_results,
                                     show=False)
                plot_equality_check(result_dict["equality_check"], path_results=path_results, show=False)

                save_results(result_dict=result_dict, path_results=path_results)

        kill_process_by_name("openethereum")

    except Exception as e:
        exception_handler.send_msg(message=f"Exception: BC-Performance-Analysis "
                                           f"\nSample #{sample} with {n_positions} positions. "
                                           f"\n{str(e)}")
        kill_process_by_name("openethereum")
        raise e

    return result_dict


def plot_time_complexity_analysis(db_timings, bc_timings, path_results=None, only_full_market=False, reference=None,
                                  show=True):
    reference_value = 1
    fig = plt.figure()
    ax = plt.subplot(111)
    if only_full_market:
        key = "full_market"
        df = db_timings["full_market"]
        x_values = [int(x) for x in df.columns]
        if reference:
            reference_value = df.iloc[:, 0].mean()
        y_error = [df.max() - df.mean(), df.mean() - df.min()]
        ax.errorbar(x=x_values, y=df.mean() / reference_value, yerr=[e / reference_value for e in y_error],
                    marker="x",
                    label="db: " + key.replace("_", " "))
        # Calculate fitting lines
        a_db, b_db, c_db = np.polyfit(x=x_values, y=df.mean() / reference_value, deg=2)
        ax.plot(x_values, [a_db * x ^ 2 + b_db * x + c_db for x in x_values], linestyle="dashed",
                label=f"DB linear fit: m = {round(a_db, 2)} and b = {round(b_db, 2)}")
        df = bc_timings["full_market"]
        y_error = [df.max() - df.mean(), df.mean() - df.min()]
        ax.errorbar(x=x_values, y=df.mean() / reference_value, yerr=[e / reference_value for e in y_error],
                    marker="x",
                    label="bc: " + key.replace("_", " "))
        # Calculate fitting lines
        a_bc, b_bc, c_bc = np.polyfit(x=x_values, y=df.mean() / reference_value, deg=2)
        ax.plot(x_values, [a_bc * x ^ 2 + b_bc * x + c_bc for x in x_values], linestyle="dashed",
                label=f"BC linear fit: m = {round(a_bc, 2)} and b = {round(b_bc, 2)}")
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
        fig.savefig(f"{path_results}/figures/timing_results.svg")
    if show:
        plt.show()
    else:
        plt.close()


def plot_time_complexity_analysis_box(results, path_results=None, reference=None, show=True):
    reference_value = 1
    fig = plt.figure(figsize=[7, 5])
    ax = plt.subplot(111)
    df = results["timing"]["db"]["full_market"]
    x_values = [int(x) for x in df.columns]
    if reference:
        reference_value = df.iloc[:, 0].mean()
    c = "#228B22"
    bp0 = ax.boxplot(df / reference_value, positions=x_values, widths=20, manage_ticks=False, patch_artist=True,
                     boxprops=dict(facecolor="#F5F5F5", color=c),
                     capprops=dict(color=c),
                     whiskerprops=dict(color=c),
                     medianprops=dict(color="#006400"),
                     flierprops=dict(markeredgecolor="#006400", markerfacecolor="#B4EEB4"),
                     showfliers=True)  # ,
    # whis=(0, 100))
    # Calculate fitting lines
    coeff_db, residuals_db, _, _, _ = np.polyfit(x=x_values, y=df.mean() / reference_value, deg=2, full=True)
    lp0 = ax.plot(x_values, [np.polyval(coeff_db, x) / reference_value for x in x_values], linestyle="dotted",
                  color="#2BCE48")
    df = results["timing"]["bc"]["full_market"]
    c = "#68228B"
    bp2 = ax.boxplot(df / reference_value, positions=x_values, widths=20, manage_ticks=False, patch_artist=True,
                     boxprops=dict(facecolor="#F5F5F5", color=c),
                     capprops=dict(color=c),
                     whiskerprops=dict(color=c),
                     medianprops=dict(color="#8A2BE2"),
                     flierprops=dict(markeredgecolor="#8A2BE2", markerfacecolor="#EEAEEE"),
                     showfliers=True)  # ,
    # whis=(0, 100))  # , positions=x_values, widths=20, manage_ticks=True)
    # Calculate fitting lines
    coeff_bc, residuals_bc, _, _, _ = np.polyfit(x=x_values, y=df.mean() / reference_value, deg=2, full=True)
    lp1 = ax.plot(x_values, [np.polyval(coeff_bc, x) / reference_value for x in x_values],
                  linestyle="dotted", color="#BF3EFF")
    ax.set_xticks(x_values)
    ax.set_xticklabels(df.columns)
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("Computation time in s")
    ax.set_xlabel("Number of inserted buy and ask bids")
    ax2 = ax.twinx()
    mp0 = ax2.plot(x_values, results["timing"]["ratios"]["full_market"], color="#FF6103", linestyle="None", marker="d")
    coeff_ratio, residuals_ratio, _, _, _ = np.polyfit(x=x_values,
                                                       y=results["timing"]["ratios"]["full_market"] / reference_value,
                                                       deg=1, full=True)
    lp2 = ax2.plot(x_values, [np.polyval(coeff_ratio, x) / reference_value for x in x_values],
                   linestyle="dashed", color="#FF6103")
    ax2.set_ylabel('Ratio of mean centralized to\nblockchain computation times', color="#FF6103")
    ax2.set_ylim([0, 600])
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.95, box.height])
    # Put a legend to the right of the current axis
    f = mticker.ScalarFormatter(useOffset=False, useMathText=True)
    g = lambda x, pos: "${}$".format(f._formatSciNotation('%1.1e' % x))
    fmt = mticker.FuncFormatter(g)
    leg = ax.legend([bp0["caps"][0], bp2["caps"][0], lp2[0]],
                    ['Central LEM',
                     # "Linear fit: m = {}".format(fmt(m_db)) + ", b = {}".format(fmt(b_db)),
                     # "$x^2$-fit: a={}".format(fmt(a_db)) + ", b={}".format(fmt(b_db)) + ", c={}".format(fmt(c_db)),
                     'Blockchain LEM',
                     # "Linear fit: m = {}".format(fmt(m_bc)) + ", b = {}".format(fmt(b_bc)),
                     # "$x^2$-fit: a={}".format(fmt(a_bc)) + ", b={}".format(fmt(b_bc)) + ", c={}".format(fmt(c_bc))
                     "Ratio"
                     ],
                    loc='center left', bbox_to_anchor=(0.11, 1.08), frameon=False, ncol=3)
    [line.set_linewidth(3) for line in leg.get_lines()]
    if path_results is not None:
        fig.savefig(f"{path_results}/figures/timing_results_boxplot.svg")
        tikzplotlib.save(f"{path_results}/figures/timing_results_boxplot.tex")
    if show:
        plt.show()
    else:
        plt.close()


def plot_time_complexity_distributions(db_timings, bc_timings, path_results=None, show=True):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    norm_db = db_timings["full_market"].mean() / 100
    l1 = ax1.stackplot(db_timings["full_market"].columns,
                       db_timings["post_positions"].mean() / norm_db,
                       db_timings["clear_market"].mean() / norm_db,
                       db_timings["log_meter_readings"].mean() / norm_db,
                       db_timings["settle_market"].mean() / norm_db,
                       labels=["Posting bids", "Market clearing", "Logging meter readings", "Market settlement"]
                       )
    ax1.set_ylabel("Share of computation time in %")
    ax1.set_xlabel("Number of inserted buy and ask bids")
    ax1.set_title("Central LEM")

    norm_bc = bc_timings["full_market"].mean() / 100
    l2 = ax2.stackplot(bc_timings["full_market"].columns,
                       bc_timings["post_positions"].mean() / norm_bc,
                       bc_timings["clear_market"].mean() / norm_bc,
                       bc_timings["log_meter_readings"].mean() / norm_bc,
                       bc_timings["settle_market"].mean() / norm_bc
                       )
    ax2.set_xlabel("Number of inserted buy and ask bids")
    ax2.set_title("Blockchain LEM")
    fig.legend(loc="center", frameon=False, bbox_to_anchor=(0.5, 0.03), ncol=4)
    plt.subplots_adjust(left=0.05, top=0.95, wspace=0.1, right=0.98, bottom=0.15)
    if path_results is not None:
        fig.savefig(f"{path_results}/figures/computation_distribution.svg")
        tikzplotlib.save(f"{path_results}/figures/computation_distribution.tex")
    if show:
        plt.show()
    else:
        plt.close()


def plot_gas_consumption(gas_consumption_dict, path_results=None, show=True):
    fig = plt.figure()
    ax = plt.subplot(111)
    h = [ax.errorbar(x=df.columns, y=df.mean(), yerr=[df.max() - df.mean(), df.mean() - df.min()], marker="x",
                     label=key.replace("_", " ")) for key, df in gas_consumption_dict.items()]
    ax.grid()
    ax.set_ylabel("Gas consumption")
    ax.set_xlabel("Number of inserted buy and ask bids")
    # Shrink current axis
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height * 0.95])
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
        fig.savefig(f"{path_results}/figures/gas_consumption.svg")
        # tikzplotlib.save(f"{path_results}/figures/gas_consumption.tex")
    if show:
        plt.show()
    else:
        plt.close()


def plot_gas_consumption_box(gas_consumption_dict, path_results=None, show=True, color_dict=None):
    fig = plt.figure(figsize=[7, 5])
    ax = plt.subplot(111)
    if color_dict is None:
        color_dict = {"post_positions": "blue", "clear_market": "orange", "log_meter_readings": "green",
                      "settle_market": "red", "full_market": "black"}
    plot_dict = {}
    for key, df in gas_consumption_dict.items():
        plot_dict[key] = {}
        df = gas_consumption_dict[key]
        x_values = [int(x) for x in df.columns]
        c = color_dict[key]
        bp = ax.boxplot(df, positions=x_values, widths=20, manage_ticks=False, patch_artist=True,
                        boxprops=dict(facecolor=c, color=c, alpha=0.3),
                        capprops=dict(color=c),
                        whiskerprops=dict(color=c),
                        medianprops=dict(color=c),
                        flierprops=dict(markeredgecolor=c, markerfacecolor=c),
                        showfliers=True)  # ,
        # Calculate fitting lines
        coeff_db, residuals_db, _, _, _ = np.polyfit(x=x_values, y=df.mean(), deg=2, full=True)
        lp = ax.plot(x_values, [np.polyval(coeff_db, x) for x in x_values], linestyle="dotted", color=c)
        plot_dict[key]["bp"] = bp
        plot_dict[key]["lp"] = lp

    ax.set_xticks(x_values)
    ax.set_xticklabels(df.columns)
    ax.grid()
    ax.set_yscale("log")
    ax.set_ylabel("Computational effort in Ethereum's \"Gas\"")
    ax.set_xlabel("Number of inserted buy and ask bids")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width, box.height])
    leg = ax.legend([plot_dict[key]["bp"]["caps"][0] for key, plot in plot_dict.items()], plot_dict.keys(),
                    loc='center left', bbox_to_anchor=(0, 1.08), frameon=False, ncol=3)
    [line.set_linewidth(3) for line in leg.get_lines()]
    if path_results is not None:
        fig.savefig(f"{path_results}/figures/gas_consumption_box.svg")
        tikzplotlib.save(f"{path_results}/figures/gas_consumption_box.tex")
    if show:
        plt.show()
    else:
        plt.close()


def plot_equality_check(equality_check_dict, path_results=None, show=True):
    for key, df in equality_check_dict.items():
        svm = sns.heatmap(df.dropna(), annot=True, vmin=0, vmax=100, cmap="RdYlGn_r",
                          cbar_kws={"label": "Percentage deviation"})
        plt.xlabel("Number of inserted bids")
        plt.ylabel("Sample number")
        plt.title(key)
        if path_results is not None:
            fig = svm.get_figure()
            fig.savefig(f"{path_results}/figures/equality_check_{key}.svg")
        if show:
            plt.show()
        else:
            plt.close()


def calc_ratios(results):
    results["timing"]["ratios"] = pd.DataFrame(index=list(results['timing']['db']['full_market'].columns),
                                               columns=list(results['timing']['db'].keys()))
    for key, df in results['timing']['db'].items():
        results["timing"]["ratios"][key] = results["timing"]["bc"][key].mean() / results["timing"]["db"][key].mean()

    return results


def create_results_folder(path_results):
    current_time_str = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M_%S")
    os.mkdir(f"{path_results}/{current_time_str}")
    os.mkdir(f"{path_results}/{current_time_str}/figures")
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
    # # load telegram configuration
    # with open(f"telegram_config.yaml") as config_file:
    #     config = yaml.load(config_file, Loader=yaml.FullLoader)
    # # Create sender object
    # telegram_sender = SenderTelegram(
    #     {'bot_token': config['telegram']['bot_token'], 'chat_ids': config['telegram']['chat_ids']})
    # # Create result folder
    # path_result_folder = create_results_folder(path_results="evaluation_results")
    # path_openethereum_toml = "C:\\Users\\ga47num\\oe_EBL\\oeEBL_user.toml"
    # # Compute performance analysis
    # results = compute_performance_analysis(path_results=path_result_folder,
    #                                        exception_handler=telegram_sender,
    #                                        path_openethereum_toml=path_openethereum_toml)
    # # Save results to files
    # save_results(results, path_result_folder)
    # # # path_result_folder = "evaluation_results/2021_11_21_13_22_24"
    path_result_folder = "H:/Dissertation/Local Energy Markets/Solidity toolbox for LEMs/2021_11_25_19_01_16"
    # Load data from previous analysis
    results = load_results(path_result_folder=path_result_folder)
    # Calculate ratios
    results = calc_ratios(results)
    # Plot results

    color_dict = {"post_positions": "blue", "clear_market": "orange", "log_meter_readings": "green",
                  "settle_market": "red", "full_market": "black"}
    # plot_time_complexity_analysis(results["timing"]["db"], results["timing"]["bc"], path_results=path_result_folder,
    #                               only_full_market=True, reference=True)
    # plot_time_complexity_analysis_box(results, path_results=path_result_folder, reference=False, show=True)
    # plot_time_complexity_distributions(db_timings=results["timing"]["db"], bc_timings=results["timing"]["bc"],
    #                                    path_results=path_result_folder)
    plot_gas_consumption(results["gas_consumption"], path_results=path_result_folder)
    plot_gas_consumption_box(results["gas_consumption"], path_results=path_result_folder, color_dict=color_dict)
    # plot_equality_check(results["equality_check"], path_results=path_result_folder)

import os
import time
import json
import test_utils
import numpy as np
import pandas as pd
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

    df_timing_results = pd.DataFrame(index=n_positions_range)
    df_equality_check = pd.DataFrame(index=n_positions_range)
    df_gas_consumption = pd.DataFrame(index=n_positions_range)

    print(f"Setup complete.")
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
            df_timing_results.loc[n_positions, "post_positions_db"] = t_post_positions_db
            # Clear market ex ante ###
            t_clear_market_start_db = time.time()
            clearing_ex_ante.market_clearing(db_obj=db_obj, config_lem=config["lem"], config_retailer=config_retailer,
                                             t_override=t_override, plotting=plotting, verbose=verbose,
                                             rounding_method=False)
            t_clear_market_end_db = time.time()
            t_clear_market_db = t_clear_market_end_db - t_clear_market_start_db
            df_timing_results.loc[n_positions, "clear_market_db"] = t_clear_market_db
            # Simulate meter readings from market results with random errors for settlement
            simulated_meter_readings_delta, ts_delivery_list = test_utils.simulate_meter_readings_from_market_results(
                db_obj=db_obj, rand_percent_var=15)
            # Log meter readings to LEM ###
            t_log_meter_readings_start_db = time.time()
            db_obj.log_readings_meter_delta(simulated_meter_readings_delta)
            t_log_meter_readings_end_db = time.time()
            t_log_meter_readings_db = t_log_meter_readings_end_db - t_log_meter_readings_start_db
            df_timing_results.loc[n_positions, "log_meter_readings_db"] = t_log_meter_readings_db
            # Settle market ###
            t_settle_market_start_db = time.time()
            test_utils.settle_market_db(config=config, db_obj=db_obj, ts_delivery_list=ts_delivery_list,
                                        path_sim=sim_path)
            t_settle_market_end_db = time.time()
            t_settle_market_db = t_settle_market_end_db - t_settle_market_start_db
            df_timing_results.loc[n_positions, "settle_market_db"] = t_settle_market_db
            # Full computation time ###
            t_full_market_db = t_post_positions_db + t_clear_market_db + t_log_meter_readings_db + t_settle_market_db
            df_timing_results.loc[n_positions, "full_market_db"] = t_full_market_db
            print(f"Central LEM successfully cleared {n_positions} positions.")
        except Exception as e:
            print(e)
            df_timing_results.loc[n_positions, "db_exception"] = e
            break

        #############################
        # Blockchain LEM ############
        #############################
        try:
            # convert energy qualities from string to int
            positions = test_utils._convert_qualities_to_int(db_obj, positions, config['lem']['types_quality'])
            t_post_positions_start_bc = time.time()
            _, df_gas_consumption.loc[n_positions, "push_positions"] = bc_obj_clearing_ex_ante.push_all_positions(
                positions, temporary=True, permanent=False)
            t_post_positions_end_bc = time.time()
            t_post_positions_bc = t_post_positions_end_bc - t_post_positions_start_bc
            df_timing_results.loc[n_positions, "post_positions_bc"] = t_post_positions_bc
            # Clear market ex ante ###
            t_clear_market_start_bc = time.time()
            df_gas_consumption.loc[n_positions, "clear_market"] = bc_obj_clearing_ex_ante.market_clearing_ex_ante(
                config["lem"], config_retailer=config_retailer,
                t_override=t_override, shuffle=shuffle, verbose=verbose)
            t_clear_market_end_bc = time.time()
            t_clear_market_bc = t_clear_market_end_bc - t_clear_market_start_bc
            df_timing_results.loc[n_positions, "clear_market_bc"] = t_clear_market_bc
            # Log meter readings to LEM ###
            t_log_meter_readings_start_bc = time.time()
            _, df_gas_consumption.loc[n_positions, "log_meter_readings"] = bc_obj_settlement.log_meter_readings_delta(
                simulated_meter_readings_delta)
            t_log_meter_readings_end_bc = time.time()
            t_log_meter_readings_bc = t_log_meter_readings_end_bc - t_log_meter_readings_start_bc
            df_timing_results.loc[n_positions, "log_meter_readings_bc"] = t_log_meter_readings_bc
            # Settle market ###
            t_settle_market_start_bc = time.time()
            df_gas_consumption.loc[n_positions, "settle_market"] = test_utils.settle_market_bc(config=config,
                                                                                               bc_obj_settlement=bc_obj_settlement,
                                                                                               ts_delivery_list=ts_delivery_list)
            t_settle_market_end_bc = time.time()
            t_settle_market_bc = t_settle_market_end_bc - t_settle_market_start_bc
            df_timing_results.loc[n_positions, "settle_market_bc"] = t_settle_market_bc
            # Full computation time ###
            t_full_market_bc = t_post_positions_bc + t_clear_market_bc + t_log_meter_readings_bc + t_settle_market_bc
            df_timing_results.loc[n_positions, "full_market_bc"] = t_full_market_bc
            print(f"Blockchain LEM successfully cleared {n_positions} positions.")
        except Exception as e:
            print(e)
            df_timing_results.loc[n_positions, "bc_exception"] = e
            break

        # Check equality of db and bc entries
        try:
            test_user_info(db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante)
            df_equality_check.loc[n_positions, "user_info_equal"] = True
        except AssertionError as a:
            df_equality_check.loc[n_positions, "user_info_equal"] = False
            pass

        try:
            test_meter_info(db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante)
            df_equality_check.loc[n_positions, "meter_info_equal"] = True
        except AssertionError:
            df_equality_check.loc[n_positions, "meter_info_equal"] = False
            pass

        try:
            test_clearing_results_ex_ante(db_obj=db_obj, bc_obj_clearing_ex_ante=bc_obj_clearing_ex_ante)
            df_equality_check.loc[n_positions, "market_results_equal"] = True
        except AssertionError:
            df_equality_check.loc[n_positions, "market_results_equal"] = False
            pass

        try:
            test_meter_readings(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
            df_equality_check.loc[n_positions, "meter_readings_equal"] = True
        except AssertionError:
            df_equality_check.loc[n_positions, "meter_readings_equal"] = False
            pass

        try:
            test_prices_settlement(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
            df_equality_check.loc[n_positions, "prices_settlement_equal"] = True
        except AssertionError:
            df_equality_check.loc[n_positions, "prices_settlement_equal"] = False
            pass

        try:
            test_balancing_energy(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
            df_equality_check.loc[n_positions, "balancing_energy_equal"] = True
        except AssertionError:
            df_equality_check.loc[n_positions, "balancing_energy_equal"] = False
            pass

        try:
            test_transaction_logs(db_obj=db_obj, bc_obj_settlement=bc_obj_settlement)
            df_equality_check.loc[n_positions, "transaction_logs_equal"] = True
        except AssertionError:
            df_equality_check.loc[n_positions, "transaction_logs_equal"] = False
            pass

        if (df_equality_check.loc[n_positions, :] == True).all():
            df_equality_check.loc[n_positions, "data_equal"] = True
        else:
            df_equality_check.loc[n_positions, "data_equal"] = False

        # Plot results after each iteration
        plot_performance_analysis_results(df_timing_results, path_results=path_results, data_type="timing_results")

    if df_timing_results.empty:
        print(f"No results were computed.")
    else:
        plot_performance_analysis_results(df_timing_results, path_results=path_results, data_type="timing_results")
        df_timing_results.to_csv(f"{path_results}/timing_results.csv")

    return df_timing_results, df_equality_check, df_gas_consumption


def plot_performance_analysis_results(results, path_results=None, data_type=None):
    fig = plt.figure()
    ax = plt.subplot(111)
    for column in results.columns:
        ax.plot(results.loc[:, column], marker="x", label=column.replace("_", " "))
    ax.grid()
    if data_type == "timing_results":
        ax.set_ylabel("Computation time in s")
    elif data_type == "gas_consumption":
        ax.set_ylabel("Gas consumption")
    else:
        ax.set_ylabel("Unknown data type")
    ax.set_xlabel("Number of inserted buy and ask bids")
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.7, box.height])
    # Put a legend to the right of the current axis
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), frameon=False)
    if path_results is not None:
        fig.savefig(f"{path_results}/{data_type}.svg")
    plt.show()


def create_results_folder(path_results):
    current_time_str = pd.Timestamp.now().strftime("%Y_%m_%d_%H_%M")
    os.mkdir(f"{path_results}/{current_time_str}")

    return f"{path_results}/{current_time_str}"


if __name__ == '__main__':
    # Create result folder
    result_folder = create_results_folder(path_results="evaluation_results")
    # Compute performance analysis
    timing_results, equality_check, gas_consumption = compute_performance_analysis(path_results=result_folder)
    # Plot results
    plot_performance_analysis_results(timing_results, path_results=result_folder, data_type="timing_results")
    plot_performance_analysis_results(gas_consumption, path_results=result_folder, data_type="gas_consumption")
    # Load results and plot them
    timing_results_read = pd.read_csv(f"{result_folder}/timing_results.csv", index_col=0)
    plot_performance_analysis_results(timing_results_read, data_type="timing_results")

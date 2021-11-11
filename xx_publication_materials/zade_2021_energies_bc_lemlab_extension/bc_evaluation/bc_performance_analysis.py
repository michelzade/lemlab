import time

import pandas as pd

import test_utils
import numpy as np
from lemlab.lem import clearing_ex_ante


def compute_performance_analysis():
    config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = test_utils.setup_test_general(
        generate_random_test_data=False)

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

    timing_results = pd.DataFrame(index=n_positions_range)

    # Loop through all position samples and execute market clearing
    for n_positions in n_positions_range:
        # Reset market tables before each iteration
        test_utils.reset_dynamic_market_data_tables(db_obj=db_obj, bc_obj_market=bc_obj_clearing_ex_ante,
                                                    bc_obj_settlement=bc_obj_settlement)
        # Compute random market positions
        positions = test_utils.create_random_positions(db_obj=db_obj,
                                                       config=config,
                                                       ids_user=ids_meters,
                                                       n_positions=n_positions,
                                                       verbose=False)

        # Post positions ###
        t_post_positions_start_db = time.time()
        db_obj.post_positions(positions)
        t_post_positions_end_db = time.time()
        # convert energy qualities from string to int
        positions = test_utils._convert_qualities_to_int(db_obj, positions, config['lem']['types_quality'])
        t_post_positions_start_bc = time.time()
        bc_obj_clearing_ex_ante.push_all_positions(positions, temporary=True, permanent=False)
        t_post_positions_end_bc = time.time()

        # Initialize clearing parameters
        config_retailer = None
        t_override = round(time.time())
        shuffle = False
        plotting = False
        verbose = False

        # Clear market ex ante ###
        t_clear_market_start_bc = time.time()
        bc_obj_clearing_ex_ante.market_clearing_ex_ante(config["lem"], config_retailer=config_retailer,
                                                        t_override=t_override, shuffle=shuffle, verbose=verbose)
        t_clear_market_end_bc = time.time()
        t_clear_market_start_db = time.time()
        clearing_ex_ante.market_clearing(db_obj=db_obj, config_lem=config["lem"], config_retailer=config_retailer,
                                         t_override=t_override, plotting=plotting, verbose=verbose,
                                         rounding_method=False)
        t_clear_market_end_db = time.time()

        # Simulate meter readings from market results with random errors for settlement
        simulated_meter_readings_delta, ts_delivery_list = test_utils.simulate_meter_readings_from_market_results(
            db_obj=db_obj, rand_percent_var=15)

        # Log meter readings to LEM ###
        t_log_meter_readings_start_bc = time.time()
        bc_obj_settlement.log_meter_readings_delta(simulated_meter_readings_delta)
        t_log_meter_readings_end_bc = time.time()
        t_log_meter_readings_start_db = time.time()
        db_obj.log_readings_meter_delta(simulated_meter_readings_delta)
        t_log_meter_readings_end_db = time.time()

        sim_path = "../../../simulation_results/test_sim"

        # Settle market
        t_settle_market_start_db = time.time()
        test_utils.settle_market_db(config=config, db_obj=db_obj, ts_delivery_list=ts_delivery_list, path_sim=sim_path)
        t_settle_market_end_db = time.time()
        t_settle_market_start_bc = time.time()
        test_utils.settle_market_bc(config=config, bc_obj_settlement=bc_obj_settlement,
                                    ts_delivery_list=ts_delivery_list)
        t_settle_market_end_bc = time.time()

        t_post_positions_db = t_post_positions_end_db - t_post_positions_start_db
        t_post_positions_bc = t_post_positions_end_bc - t_post_positions_start_bc
        t_clear_market_db = t_clear_market_end_db - t_clear_market_start_db
        t_clear_market_bc = t_clear_market_end_bc - t_clear_market_start_bc
        t_log_meter_readings_db = t_log_meter_readings_end_db - t_log_meter_readings_start_db
        t_log_meter_readings_bc = t_log_meter_readings_end_bc - t_log_meter_readings_start_bc
        t_settle_market_db = t_settle_market_end_db - t_settle_market_start_db
        t_settle_market_bc = t_settle_market_end_bc - t_settle_market_start_bc
        t_full_market_db = t_post_positions_db + t_clear_market_db + t_log_meter_readings_db + t_settle_market_db
        t_full_market_bc = t_post_positions_bc + t_clear_market_bc + t_log_meter_readings_bc + t_settle_market_bc

        timing_results.loc[n_positions, "t_full_market_db"] = t_full_market_db
        timing_results.loc[n_positions, "t_full_market_bc"] = t_full_market_bc
        timing_results.loc[n_positions, "t_post_positions_db"] = t_post_positions_db
        timing_results.loc[n_positions, "t_post_positions_bc"] = t_post_positions_bc
        timing_results.loc[n_positions, "t_clear_market_db"] = t_clear_market_db
        timing_results.loc[n_positions, "t_clear_market_bc"] = t_clear_market_bc
        timing_results.loc[n_positions, "t_log_meter_readings_db"] = t_log_meter_readings_db
        timing_results.loc[n_positions, "t_log_meter_readings_bc"] = t_log_meter_readings_bc
        timing_results.loc[n_positions, "t_settle_market_db"] = t_settle_market_db
        timing_results.loc[n_positions, "t_settle_market_bc"] = t_settle_market_bc

    return timing_results


if __name__ == '__main__':
    timing_results = compute_performance_analysis()

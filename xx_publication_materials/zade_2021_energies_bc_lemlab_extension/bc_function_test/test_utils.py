import time
import yaml
import random
import string

import numpy as np
import pandas as pd

from lemlab.db_connection.db_connection import DatabaseConnection
from xx_publication_materials.zade_2021_energies_bc_lemlab_extension.bc_connection.bc_connection import \
    BlockchainConnection
from lemlab.lem import clearing_ex_ante, settlement
from current_scenario_file import scenario_file_path


def setup_test_general(generate_random_test_data=False):
    yaml_file = scenario_file_path

    # load configuration file
    with open(f"" + yaml_file) as config_file:
        config = yaml.load(config_file, Loader=yaml.FullLoader)

    # Create a db connection object
    db_obj = DatabaseConnection(db_dict=config['db_connections']['database_connection_admin'],
                                lem_config=config['lem'])
    # Create bc connection objects to ClearingExAnte and Settlement contract
    market_contract_dict = config['db_connections']['bc_dict']
    market_contract_dict["contract_name"] = "ClearingExAnte"
    bc_obj_clearing_ex_ante = BlockchainConnection(bc_dict=market_contract_dict)
    settlement_contract_dict = config['db_connections']['bc_dict']
    settlement_contract_dict["contract_name"] = "Settlement"
    bc_obj_settlement = BlockchainConnection(bc_dict=settlement_contract_dict)

    if generate_random_test_data:
        init_random_data(db_obj=db_obj, bc_obj_market=bc_obj_clearing_ex_ante,
                         config=config, bc_obj_settlement=bc_obj_settlement)

    return config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement


def init_random_data(db_obj, bc_obj_market, config, bc_obj_settlement):
    # Clear data from db and bc
    db_obj.init_db(clear_tables=True, reformat_tables=True)
    bc_obj_market.clear_temp_data()
    bc_obj_market.clear_permanent_data()
    bc_obj_settlement.clear_data()

    # Create list of random user ids and meter ids
    ids_users_random = create_user_ids(num=config['prosumer']['general_number_of'])
    ids_meter_random = create_user_ids(num=config['prosumer']['general_number_of'])
    ids_market_agents = create_user_ids(num=config['prosumer']['general_number_of'])

    tx_hash = None
    # Register meters and users on database
    for z in range(len(ids_users_random)):
        cols, types = db_obj.get_table_columns(db_obj.db_param.NAME_TABLE_INFO_USER, dtype=True)
        col_data = [ids_users_random[z], 1000, 0, 10000, 100, 'green', 10, 'zi', 0, ids_market_agents[z], 0, 2147483648]
        if any([type(data) != typ for data, typ in zip(col_data, types)]):
            raise TypeError("The types of the data and the columns do not match for the info_user")
        df_insert = pd.DataFrame(
            data=[col_data],
            columns=cols)

        # Register users on bc and db
        db_obj.register_user(df_in=df_insert)
        bc_obj_market.register_user(df_user=df_insert)

        cols, types = db_obj.get_table_columns(db_obj.db_param.NAME_TABLE_INFO_METER, dtype=True)
        col_data = [ids_meter_random[z], ids_users_random[z], "0", "virtual grid meter", '0' * 10, 'green', 0,
                    2147483648,
                    'test']
        if any([type(data) != typ for data, typ in zip(col_data, types)]):
            raise TypeError("The types of data and columns do not match for the id_meter")
        df_insert = pd.DataFrame(
            data=[col_data],
            columns=cols)

        # Register meters on db and bc
        db_obj.register_meter(df_in=df_insert)
        tx_hash = bc_obj_market.register_meter(df_insert)

    bc_obj_market.wait_for_transact(tx_hash)

    # Compute random market positions
    positions = create_random_positions(db_obj=db_obj,
                                        config=config,
                                        ids_user=ids_users_random,
                                        n_positions=100,
                                        verbose=False)
    # Post positions on db
    db_obj.post_positions(positions)
    # on the bc, energy quality needs to be converted to int. In the db it is stored as a string
    positions = _convert_qualities_to_int(db_obj, positions, config['lem']['types_quality'])
    bc_obj_market.push_all_positions(positions, temporary=True, permanent=False)


def setup_clearing_ex_ante_test(generate_random_test_data):
    config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = setup_test_general(generate_random_test_data)

    # Initialize clearing parameters
    config_retailer = None
    t_override = round(time.time())
    shuffle = False
    plotting = False
    verbose = False

    # Clear market ex ante on db and bc
    bc_obj_clearing_ex_ante.market_clearing_ex_ante(config["lem"], config_retailer=config_retailer,
                                                    t_override=t_override, shuffle=shuffle, verbose=verbose)
    clearing_ex_ante.market_clearing(db_obj=db_obj, config_lem=config["lem"], config_retailer=config_retailer,
                                     t_override=t_override, plotting=plotting, verbose=verbose, rounding_method=False)

    return config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement


def setup_settlement_test(generate_random_test_data):
    config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = setup_clearing_ex_ante_test(generate_random_test_data)

    # Simulate meter readings from market results with random errors
    simulated_meter_readings_delta, ts_delivery_list = simulate_meter_readings_from_market_results(
        db_obj=db_obj, rand_percent_var=15)

    # Log meter readings delta
    bc_obj_settlement.log_meter_readings_delta(simulated_meter_readings_delta)
    db_obj.log_readings_meter_delta(simulated_meter_readings_delta)

    # Calculate/determine balancing energies
    bc_obj_settlement.determine_balancing_energy(ts_delivery_list)
    settlement.determine_balancing_energy(db_obj, ts_delivery_list)

    sim_path = "../../../simulation_results/test_sim"

    # Set settlement prices in db and bc
    settlement.set_prices_settlement(db_obj=db_obj, path_simulation=sim_path, list_ts_delivery=ts_delivery_list)
    bc_obj_settlement.set_prices_settlement(ts_delivery_list)

    # Update balances according to balancing energies db and bc
    ts_now = round(time.time())
    id_retailer = "retailer01"
    settlement.update_balance_balancing_costs(db_obj=db_obj, t_now=ts_now,
                                              list_ts_delivery=ts_delivery_list,
                                              id_retailer=id_retailer, lem_config=config["lem"])
    bc_obj_settlement.update_balance_balancing_costs(list_ts_delivery=ts_delivery_list,
                                                     ts_now=ts_now, supplier_id=id_retailer)

    # Update balances with levies on db and bc
    settlement.update_balance_levies(db_obj=db_obj, t_now=ts_now, list_ts_delivery=ts_delivery_list,
                                     id_retailer=id_retailer, lem_config=config["lem"])
    bc_obj_settlement.update_balance_levies(list_ts_delivery=ts_delivery_list, ts_now=ts_now, id_retailer=id_retailer)

    return config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement


def simulate_meter_readings_from_market_results(db_obj=None, rand_percent_var=15):
    """
    Read market results from data base
    Aggregate users with meters for each timestep
    Randomly change energy traded
    Push output energy traded into meter_reading_deltas
    Returns: None
    -------
    random_variance: how much the energy delta is changed to create some energy balances, if 0, no changes
    are made at all
    """
    if db_obj is None:
        yaml_file = scenario_file_path
        # load configuration file
        with open(yaml_file) as config_file:
            config = yaml.load(config_file, Loader=yaml.FullLoader)
        # Create a db connection object
        db_obj = DatabaseConnection(db_dict=config['db_connections']['database_connection_admin'],
                                    lem_config=config['lem'])
        # Initialize database
        db_obj.init_db(clear_tables=False, reformat_tables=False)

    # for this to work, we need to have a full market cleared before, so execute the test if you havent
    market_results, _ = db_obj.get_results_market_ex_ante()
    assert not market_results.empty, "Error: The market results are empty"
    # retrieve list of users and initialize a mapping
    list_users_offers = list(set(market_results[db_obj.db_param.ID_USER_OFFER]))
    list_users_bids = list(set(market_results[db_obj.db_param.ID_USER_BID]))

    user_offers2ts_qty = dict([(user, {}) for user in list_users_offers])
    user_bids2ts_qty = dict([(user, {}) for user in list_users_bids])

    list_ts_delivery = []  # additionally we save all the timesteps registered
    # for each user we have a dictionary with each single timestep as key and the total energy traded in that
    # timestep as value
    for i, row in market_results.iterrows():
        # for the user offers
        if row[db_obj.db_param.TS_DELIVERY] in user_offers2ts_qty[row[db_obj.db_param.ID_USER_OFFER]]:
            user_offers2ts_qty[row[db_obj.db_param.ID_USER_OFFER]][row[db_obj.db_param.TS_DELIVERY]] += row[
                db_obj.db_param.QTY_ENERGY_TRADED]
        else:
            user_offers2ts_qty[row[db_obj.db_param.ID_USER_OFFER]][row[db_obj.db_param.TS_DELIVERY]] = row[
                db_obj.db_param.QTY_ENERGY_TRADED]

        # for the user bids
        if row[db_obj.db_param.TS_DELIVERY] in user_bids2ts_qty[row[db_obj.db_param.ID_USER_BID]]:
            user_bids2ts_qty[row[db_obj.db_param.ID_USER_BID]][row[db_obj.db_param.TS_DELIVERY]] -= row[
                db_obj.db_param.QTY_ENERGY_TRADED]
        else:
            user_bids2ts_qty[row[db_obj.db_param.ID_USER_BID]][row[db_obj.db_param.TS_DELIVERY]] = row[
                db_obj.db_param.QTY_ENERGY_TRADED]

        list_ts_delivery.append(row[db_obj.db_param.TS_DELIVERY])

    list_ts_delivery = sorted(list(set(list_ts_delivery)))  # eliminate duplicates and sort in ascending order
    # we know aggregate both the users who bid and the ones who offer
    list_users_offers.extend(list_users_bids)
    list_users_offers = list(set(list_users_offers))

    # we now map each user to its meter
    map_user2meter = db_obj.get_map_to_main_meter()
    # we filter the rest of the mappings from the dict and get only user 2 meter
    user2meter = dict([(user, map_user2meter[user]) for user in list_users_offers])

    assert list_users_offers == list(user2meter.keys()), "The list of users from the market result and the meters " \
                                                         "does not match "

    meter2ts_qty = dict([(user2meter[user], {}) for user in list_users_offers])

    for user, meter in user2meter.items():
        # we first extract the timesteps where the user had an interaction, offer or bid
        try:
            list_current_ts = list(user_offers2ts_qty[user].keys())
        except KeyError:
            list_current_ts = []
        try:
            list_current_ts.extend(list(user_bids2ts_qty[user].keys()))
        except KeyError:
            pass
        list_current_ts = list(set(list_current_ts))  # eliminate duplicates
        for ts in list_current_ts:
            # there may be an offer with such ts but not a bid or viceversa, so we initialize the other to 0
            try:
                offer = user_offers2ts_qty[user][ts]
            except KeyError:
                offer = 0
            try:
                bid = user_bids2ts_qty[user][ts]
            except KeyError:
                bid = 0
            meter2ts_qty[meter][ts] = offer - bid

    assert list(meter2ts_qty.keys()) == list(user2meter.values()), "Meters do not match in market and meter tables"

    # we create the dataframe for the delta readings and append the information
    simulated_meter_readings_delta = pd.DataFrame(columns=[db_obj.db_param.TS_DELIVERY, db_obj.db_param.ID_METER,
                                                           db_obj.db_param.ENERGY_IN, db_obj.db_param.ENERGY_OUT])
    for meter in meter2ts_qty:
        for ts in meter2ts_qty[meter]:
            if not rand_percent_var:  # not equal to 0
                rand_factor = random.randrange(-rand_percent_var, rand_percent_var) / 100.0 + 1.0
            else:
                rand_factor = 1
            if meter2ts_qty[meter][ts] > 0:
                simulated_meter_readings_delta = simulated_meter_readings_delta.append(
                    {db_obj.db_param.TS_DELIVERY: ts, db_obj.db_param.ID_METER: meter,
                     db_obj.db_param.ENERGY_IN: int(round(meter2ts_qty[meter][ts] * rand_factor)),
                     db_obj.db_param.ENERGY_OUT: 0}, ignore_index=True)
            else:
                simulated_meter_readings_delta = simulated_meter_readings_delta.append(
                    {db_obj.db_param.TS_DELIVERY: ts, db_obj.db_param.ID_METER: meter,
                     db_obj.db_param.ENERGY_IN: 0,
                     db_obj.db_param.ENERGY_OUT: -int(round(meter2ts_qty[meter][ts] * rand_factor))}, ignore_index=True)

    simulated_meter_readings_delta.sort_values(by=[db_obj.db_param.TS_DELIVERY, db_obj.db_param.ID_METER])
    simulated_meter_readings_delta = simulated_meter_readings_delta.reset_index(drop=True)

    return simulated_meter_readings_delta, list_ts_delivery


def insert_random_positions(db_obj, config, positions, n_positions, t_d_range, ids_user):
    positions.loc[:, db_obj.db_param.ID_USER] = random.choices(ids_user, k=n_positions)
    positions.loc[:, db_obj.db_param.T_SUBMISSION] = [round(time.time())] * n_positions
    positions.loc[:, db_obj.db_param.QTY_ENERGY] = random.choices(range(1, 1000, 1), k=n_positions)
    positions.loc[:, db_obj.db_param.TYPE_POSITION] = random.choices(list(config['lem']['types_position'].values()),
                                                                     k=n_positions)
    positions.loc[:, db_obj.db_param.QUALITY_ENERGY] = random.choices(list(config['lem']['types_quality'].values()),
                                                                      k=n_positions)
    positions.loc[:, db_obj.db_param.TS_DELIVERY] = random.choices(t_d_range, k=n_positions)
    positions.loc[:, db_obj.db_param.NUMBER_POSITION] = int(0)
    positions.loc[:, db_obj.db_param.STATUS_POSITION] = int(0)
    positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'offer', db_obj.db_param.PRICE_ENERGY] = [
        int(x * db_obj.db_param.EURO_TO_SIGMA / 1000)
        for x in random.choices(np.arange(config['retailer']['price_buy'],
                                          config['retailer']['price_sell'], 0.0001),
                                k=len(positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'offer', :]))]
    positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'offer',
                  db_obj.db_param.PREMIUM_PREFERENCE_QUALITY] = int(0)
    positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'bid', db_obj.db_param.PRICE_ENERGY] = [
        int(x * db_obj.db_param.EURO_TO_SIGMA / 1000)
        for x in random.choices(np.arange(config['retailer']['price_buy'],
                                          config['retailer']['price_sell'], 0.0001),
                                k=len(positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'bid', :]))]
    positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'bid',
                  db_obj.db_param.PREMIUM_PREFERENCE_QUALITY] = random.choices(range(0, 50, 1), k=len(
        positions.loc[positions[db_obj.db_param.TYPE_POSITION] == 'bid', :]))

    return positions


def create_random_positions(db_obj, config, ids_user, n_positions=None, verbose=False):
    if n_positions is None:
        n_positions = 100
    t_start = round(time.time()) - (
            round(time.time()) % config['lem']['interval_clearing']) + config['lem']['interval_clearing']
    t_end = t_start + config['lem']['horizon_clearing']
    # Range of time steps
    t_d_range = np.arange(t_start, t_end, config['lem']['interval_clearing'])
    # Create bid df
    positions = pd.DataFrame(columns=db_obj.get_table_columns(db_obj.db_param.NAME_TABLE_POSITIONS_MARKET_EX_ANTE))
    positions = insert_random_positions(db_obj, config, positions, n_positions, t_d_range, ids_user)

    # Drop duplicates
    positions = positions.drop_duplicates(
        subset=[db_obj.db_param.ID_USER, db_obj.db_param.NUMBER_POSITION, db_obj.db_param.TYPE_POSITION,
                db_obj.db_param.TS_DELIVERY])
    if verbose:
        print(pd.Timestamp.now(), 'Positions successfully written to DB')

    return positions


# Create random user ids
def create_user_ids(num=30):
    user_id_list = list()
    for i in range(num):
        # Create random user id in the form of 1234ABDS
        user_id_int = np.random.randint(1000, 10000)
        user_id_str = ''.join(random.sample(string.ascii_uppercase, 4))
        user_id_random = str(user_id_int) + user_id_str
        # Append user id to list
        user_id_list.append(user_id_random)

    return user_id_list


def _convert_qualities_to_int(db_obj, positions, dict_types):
    dict_types_inverted = {v: k for k, v in dict_types.items()}
    positions = positions.assign(**{db_obj.db_param.QUALITY_ENERGY: [dict_types_inverted[i] for i in
                                                                     positions[db_obj.db_param.QUALITY_ENERGY]]})

    return positions


if __name__ == '__main__':
    # setup_test_general(generate_random_test_data=True)
    # setup_clearing_ex_ante_test(generate_random_test_data=True)
    setup_settlement_test(generate_random_test_data=True)

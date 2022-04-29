import pytest
import test_utils
import pandas as pd
from lemlab.lem.settlement import determine_balancing_energy

config = None
db_obj = None
bc_obj_clearing_ex_ante = None
bc_obj_settlement = None


# this method is executed before all the others, to get useful global variables, needed for the tests
@pytest.fixture(scope="session", autouse=True)
def setup():
    global config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement
    config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = \
        test_utils.settlement_test(generate_random_test_data=True)


def test_clearing_results_ex_ante(db_obj=None, bc_obj_clearing_ex_ante=None, n_sample=None, n_positions=None,
                                  path_results=None):
    # Get market results from db and bc
    clearing_ex_ante_results_db, _ = db_obj.get_results_market_ex_ante()
    clearing_ex_ante_results_bc = bc_obj_clearing_ex_ante.get_market_results()
    # Reindex both df so column order is equal
    clearing_ex_ante_results_bc = clearing_ex_ante_results_bc.reindex(sorted(clearing_ex_ante_results_bc.columns),
                                                                      axis=1)
    clearing_ex_ante_results_db = clearing_ex_ante_results_db.reindex(sorted(clearing_ex_ante_results_db.columns),
                                                                      axis=1)
    # Create a unique string, then apply a stable sort, and drop index
    columns_to_sort_by = clearing_ex_ante_results_db.columns.to_list()
    columns_to_sort_by.remove("t_cleared")
    clearing_ex_ante_results_db = clearing_ex_ante_results_db.assign(sort_idx=[','.join(ele.split()) for ele in
                                                                               clearing_ex_ante_results_db.to_string(
                                                                                   header=False, index=False,
                                                                                   index_names=False,
                                                                                   columns=columns_to_sort_by).split(
                                                                                   '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)
    clearing_ex_ante_results_bc = clearing_ex_ante_results_bc.assign(sort_idx=[','.join(ele.split()) for ele in
                                                                               clearing_ex_ante_results_bc.to_string(
                                                                                   header=False,
                                                                                   index=False,
                                                                                   index_names=False,
                                                                                   columns=columns_to_sort_by).split(
                                                                                   '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)

    try:
        # Check whether market results are equal on db and bc
        pd.testing.assert_frame_equal(clearing_ex_ante_results_bc, clearing_ex_ante_results_db, check_dtype=False)
        diff = 0
    except AssertionError as e:
        # Check whether welfare is equal
        db_welfare = ((clearing_ex_ante_results_db["price_energy_bid"] - clearing_ex_ante_results_db["price_energy_offer"]) * clearing_ex_ante_results_db["qty_energy_traded"]).sum()
        bc_welfare = ((clearing_ex_ante_results_bc["price_energy_bid"] - clearing_ex_ante_results_bc["price_energy_offer"]) * clearing_ex_ante_results_bc["qty_energy_traded"]).sum()
        if db_welfare == bc_welfare:
            # if welfare is equal then results are equally acceptable, hence pass
            diff = 0
            pass
        else:
            if "DataFrame are different" in str(e.args):
                diff = 100
            else:
                diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
            if path_results:
                clearing_ex_ante_results_db.to_csv(
                    f"{path_results}/exception/{n_sample}_{n_positions}_unequal_clearing_ex_ante_results_db.csv")
                clearing_ex_ante_results_bc.to_csv(
                    f"{path_results}/exception/{n_sample}_{n_positions}_unequal_clearing_ex_ante_results_bc.csv")
    return diff


def test_meter_readings(db_obj=None, bc_obj_settlement=None, n_sample=None, n_positions=None, path_results=None):
    # Get meter readings delta
    meter_readings_delta_bc = bc_obj_settlement.get_meter_readings_delta().sort_values(
        by=[db_obj.db_param.TS_DELIVERY, db_obj.db_param.ID_METER])
    meter_readings_delta_bc = meter_readings_delta_bc.reset_index(drop=True)
    meter_readings_delta_db = db_obj.get_meter_readings_delta().sort_values(
        by=[db_obj.db_param.TS_DELIVERY, db_obj.db_param.ID_METER])
    meter_readings_delta_db = meter_readings_delta_db.reset_index(drop=True)

    # Check whether meter_readings_delta on bc and db are equal
    try:
        pd.testing.assert_frame_equal(meter_readings_delta_bc, meter_readings_delta_db, check_dtype=False)
        diff = 0
    except AssertionError as e:
        if "DataFrame are different" in str(e.args):
            diff = 100
        else:
            diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
    return diff


def test_balancing_energy(db_obj=None, bc_obj_settlement=None, bc_obj_clearing_ex_ante=None, n_sample=None, n_positions=None, path_results=None):
    # Get energy balances from db and bc
    balancing_energies_db = db_obj.get_energy_balancing()
    balancing_energies_bc = bc_obj_settlement.get_energy_balances()
    # Reindex column order
    balancing_energies_bc = balancing_energies_bc.reindex(sorted(balancing_energies_bc.columns), axis=1)
    balancing_energies_db = balancing_energies_db.reindex(sorted(balancing_energies_db.columns), axis=1)
    # Create a unique string, then apply a stable sort, and drop index
    balancing_energies_db = balancing_energies_db.assign(sort_idx=[','.join(ele.split()) for ele in
                                                                   balancing_energies_db.to_string(
                                                                       header=False, index=False,
                                                                       index_names=False).split(
                                                                       '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)
    balancing_energies_bc = balancing_energies_bc.assign(sort_idx=[','.join(ele.split()) for ele in
                                                                   balancing_energies_bc.to_string(
                                                                       header=False,
                                                                       index=False,
                                                                       index_names=False).split(
                                                                       '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(balancing_energies_db, balancing_energies_bc, check_dtype=False)
        diff = 0
    except AssertionError as e:
        # Check whether balancing energies are correctly computed by checking in python
        # Get market results from db and bc
        clearing_ex_ante_results_db, _ = db_obj.get_results_market_ex_ante()
        clearing_ex_ante_results_bc = bc_obj_clearing_ex_ante.get_market_results()
        # Get meter readings delta
        meter_readings_delta_bc = bc_obj_settlement.get_meter_readings_delta().sort_values(
            by=[db_obj.db_param.TS_DELIVERY, db_obj.db_param.ID_METER])
        meter_readings_delta_bc = meter_readings_delta_bc.reset_index(drop=True)
        meter_readings_delta_db = db_obj.get_meter_readings_delta().sort_values(
            by=[db_obj.db_param.TS_DELIVERY, db_obj.db_param.ID_METER])
        meter_readings_delta_db = meter_readings_delta_db.reset_index(drop=True)
        # Calculate balancing energies with python implementation
        balancing_energy_db_python = determine_balancing_energy(db_obj=db_obj, list_ts_delivery=clearing_ex_ante_results_db[db_obj.db_param.TS_DELIVERY].unique(), df_market_results=clearing_ex_ante_results_db, df_meter_readings=meter_readings_delta_db)
        balancing_energy_bc_python = determine_balancing_energy(db_obj=db_obj, list_ts_delivery=clearing_ex_ante_results_bc[db_obj.db_param.TS_DELIVERY].unique(), df_market_results=clearing_ex_ante_results_bc, df_meter_readings=meter_readings_delta_bc)
        # Reindex column order
        balancing_energy_bc_python = balancing_energy_bc_python.reindex(sorted(balancing_energy_bc_python.columns), axis=1)
        balancing_energy_db_python = balancing_energy_db_python.reindex(sorted(balancing_energy_db_python.columns), axis=1)
        # Create a unique string, then apply a stable sort, and drop index
        balancing_energy_db_python = balancing_energy_db_python.assign(sort_idx=[','.join(ele.split()) for ele in
                                                                       balancing_energy_db_python.to_string(
                                                                           header=False, index=False,
                                                                           index_names=False).split(
                                                                           '\n')]).sort_values(
            by="sort_idx", kind="stable").reset_index(drop=True)
        balancing_energy_bc_python = balancing_energy_bc_python.assign(sort_idx=[','.join(ele.split()) for ele in
                                                                       balancing_energy_bc_python.to_string(
                                                                           header=False,
                                                                           index=False,
                                                                           index_names=False).split(
                                                                           '\n')]).sort_values(
            by="sort_idx", kind="stable").reset_index(drop=True)

        if balancing_energy_bc_python.equals(balancing_energies_bc) and \
                balancing_energy_db_python.equals(balancing_energies_db):
            diff = 0
            pass
        else:
            if "DataFrame are different" in str(e.args):
                diff = 100
            else:
                diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
            if path_results:
                balancing_energies_db.to_csv(f"{path_results}/exception/{n_sample}_{n_positions}_unequal_balancing_energy_db.csv")
                balancing_energies_bc.to_csv(f"{path_results}/exception/{n_sample}_{n_positions}_unequal_balancing_energy_bc.csv")

    return diff


def test_prices_settlement(db_obj=None, bc_obj_settlement=None, n_sample=None, n_positions=None, path_results=None):
    # Get settlement prices from db and bc
    settlement_prices_db = db_obj.get_prices_settlement()
    settlement_prices_db = settlement_prices_db.sort_values(by=[db_obj.db_param.TS_DELIVERY,
                                                                db_obj.db_param.PRICE_ENERGY_BALANCING_POSITIVE])
    settlement_prices_db = settlement_prices_db.reset_index(drop=True)
    settlement_prices_bc = bc_obj_settlement.get_prices_settlement()
    settlement_prices_bc = settlement_prices_bc.sort_values(by=[db_obj.db_param.TS_DELIVERY,
                                                                db_obj.db_param.PRICE_ENERGY_BALANCING_POSITIVE])
    settlement_prices_bc = settlement_prices_bc.reset_index(drop=True)

    # Check whether settlement prices are equal on db and bc
    try:
        pd.testing.assert_frame_equal(settlement_prices_db, settlement_prices_bc)
        diff = 0
    except AssertionError as e:
        if "DataFrame are different" in str(e.args):
            diff = 100
        else:
            diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
    return diff


def test_transaction_logs(db_obj=None, bc_obj_settlement=None, n_sample=None, n_positions=None, path_results=None):
    # Get all logged transactions from db and bc
    log_transactions_db = db_obj.get_logs_transactions()
    log_transactions_db = log_transactions_db.loc[
        # log_transactions_db[db_obj.db_param.TYPE_TRANSACTION].str.contains('balancing|levies')]
        log_transactions_db[db_obj.db_param.TYPE_TRANSACTION].str.contains('balancing|levies')]
    log_transactions_bc = bc_obj_settlement.get_logs_transactions()
    # Reindex column order
    log_transactions_bc = log_transactions_bc.reindex(sorted(log_transactions_bc.columns), axis=1)
    log_transactions_db = log_transactions_db.reindex(sorted(log_transactions_db.columns), axis=1)

    # Create a unique string, then apply a stable sort, and drop index
    columns_to_sort_by = log_transactions_db.columns.to_list()
    columns_to_sort_by.remove("t_update_balance")
    log_transactions_db = log_transactions_db.assign(sort_idx=[','.join(ele.split()) for ele in
                                                               log_transactions_db.to_string(
                                                                   header=False, index=False,
                                                                   index_names=False, columns=columns_to_sort_by).split(
                                                                   '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)
    log_transactions_bc = log_transactions_bc.assign(sort_idx=[','.join(ele.split()) for ele in
                                                               log_transactions_bc.to_string(
                                                                   header=False,
                                                                   index=False,
                                                                   index_names=False, columns=columns_to_sort_by).split(
                                                                   '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)

    try:
        # Check whether all transactions on bc and db are equal
        pd.testing.assert_frame_equal(log_transactions_db, log_transactions_bc)
        diff = 0
    except AssertionError as e:
        if "DataFrame are different" in str(e.args):
            diff = 100
        else:
            diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
        if path_results:
            log_transactions_db.to_csv(f"{path_results}/exception/{n_sample}_{n_positions}_unequal_log_txs_db.csv")
            log_transactions_bc.to_csv(f"{path_results}/exception/{n_sample}_{n_positions}_unequal_log_txs_bc.csv")

    return diff


def test_user_info(db_obj=None, bc_obj_clearing_ex_ante=None, n_sample=None, n_positions=None, path_results=None):
    info_user_db = db_obj.get_info_user()
    info_user_bc = bc_obj_clearing_ex_ante.get_list_all_users()

    # Reindex column order
    info_user_bc = info_user_bc.reindex(sorted(info_user_bc.columns), axis=1)
    info_user_db = info_user_db.reindex(sorted(info_user_db.columns), axis=1)

    # Create a unique string, then apply a stable sort, and drop index
    columns_to_sort_by = info_user_db.columns.to_list()
    columns_to_sort_by.remove("t_update_balance")
    info_user_db = info_user_db.assign(sort_idx=[','.join(ele.split()) for ele in
                                                 info_user_db.to_string(
                                                     header=False, index=False,
                                                     index_names=False, columns=columns_to_sort_by).split(
                                                     '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)
    info_user_bc = info_user_bc.assign(sort_idx=[','.join(ele.split()) for ele in
                                                 info_user_bc.to_string(
                                                     header=False,
                                                     index=False,
                                                     index_names=False, columns=columns_to_sort_by).split(
                                                     '\n')]).sort_values(
        by="sort_idx", kind="stable").reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(info_user_db, info_user_bc)
        diff = 0
    except AssertionError as e:
        if "DataFrame are different" in str(e.args):
            diff = 100
        else:
            diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
        if path_results:
            info_user_db.to_csv(f"{path_results}/exception/{n_sample}_{n_positions}_unequal_user_infos_db.csv")
            info_user_bc.to_csv(f"{path_results}/exception/{n_sample}_{n_positions}_unequal_user_infos_bc.csv")
    return diff


def test_meter_info(db_obj=None, bc_obj_clearing_ex_ante=None, n_sample=None, n_positions=None, path_results=None):
    info_meter_db = db_obj.get_info_meter()
    info_meter_db = info_meter_db.sort_values(by=[db_obj.db_param.ID_METER])
    info_meter_db = info_meter_db.reset_index(drop=True)

    info_meter_bc = bc_obj_clearing_ex_ante.get_list_all_meters()
    info_meter_bc = info_meter_bc.sort_values(by=[db_obj.db_param.ID_METER])
    info_meter_bc = info_meter_bc.reset_index(drop=True)

    try:
        pd.testing.assert_frame_equal(info_meter_db, info_meter_bc)
        diff = 0
    except AssertionError as e:
        if "DataFrame are different" in str(e.args):
            diff = 100
        else:
            diff = float(str(e.args).split("(")[-1].split(")")[0][:-2])
    return diff

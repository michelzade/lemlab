import pytest
import test_utils
import pandas as pd

config = None
db_obj = None
bc_obj_clearing_ex_ante = None


@pytest.fixture(scope="session", autouse=True)
def setup():
    global config, db_obj, bc_obj_clearing_ex_ante
    config, db_obj, bc_obj_clearing_ex_ante, _ = test_utils.setup_test_general(generate_random_test_data=True)


def test_market_positions():
    # Get open market positions
    market_positions_open_bids_db, market_positions_open_offers_db = db_obj.get_open_positions()
    market_positions_open_db = pd.concat([market_positions_open_bids_db, market_positions_open_offers_db])
    market_positions_open_bc = bc_obj_clearing_ex_ante.get_open_positions(return_all=True)
    # Sort market results
    market_positions_open_bc = market_positions_open_bc.sort_values(by=[db_obj.db_param.TS_DELIVERY,
                                                                        db_obj.db_param.ID_USER,
                                                                        db_obj.db_param.QTY_ENERGY])
    market_positions_open_db = market_positions_open_db.sort_values(by=[db_obj.db_param.TS_DELIVERY,
                                                                        db_obj.db_param.ID_USER,
                                                                        db_obj.db_param.QTY_ENERGY])
    market_positions_open_db = market_positions_open_db.reset_index(drop=True)
    market_positions_open_bc = market_positions_open_bc.reset_index(drop=True)

    market_positions_open_db = test_utils._convert_qualities_to_int(db_obj,
                                                                    market_positions_open_db,
                                                                    config["lem"]['types_quality'])
    # Check whether positions are equal
    pd.testing.assert_frame_equal(market_positions_open_bc, market_positions_open_db, check_dtype=False)


def test_user_info():
    # Get user infos
    info_user_db = db_obj.get_info_user()
    info_user_bc = bc_obj_clearing_ex_ante.get_list_all_users()
    # Sort and reset indices
    info_user_db = info_user_db.sort_values(by=[db_obj.db_param.ID_USER])
    info_user_db = info_user_db.reset_index(drop=True)
    info_user_bc = info_user_bc.sort_values(by=[db_obj.db_param.ID_USER])
    info_user_bc = info_user_bc.reset_index(drop=True)
    # Check whether user infos are equal
    pd.testing.assert_frame_equal(info_user_db, info_user_bc, check_dtype=False)


def test_info_meters():
    # Get meter infos
    info_meter_db = db_obj.get_info_meter()
    info_meter_bc = bc_obj_clearing_ex_ante.get_info_meter()
    # Sort and reset indices
    info_meter_db = info_meter_db.sort_values(by=[db_obj.db_param.ID_USER])
    info_meter_db = info_meter_db.reset_index(drop=True)
    info_meter_bc = info_meter_bc.sort_values(by=[db_obj.db_param.ID_USER])
    info_meter_bc = info_meter_bc.reset_index(drop=True)
    # Check whether user infos are equal
    pd.testing.assert_frame_equal(info_meter_db, info_meter_bc, check_dtype=False)

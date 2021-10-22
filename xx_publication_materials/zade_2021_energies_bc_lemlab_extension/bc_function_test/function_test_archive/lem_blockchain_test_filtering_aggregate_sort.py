import pytest
import test_utils
from lemlab.lem.clearing_ex_ante import _convert_qualities_to_int, _aggregate_identical_positions

config = None
db_obj = None
bc_obj_clearing_ex_ante = None


@pytest.fixture(scope="session", autouse=True)
def setup():
    global config, db_obj, bc_obj_clearing_ex_ante
    config, db_obj, bc_obj_clearing_ex_ante, _ = test_utils.setup_test_general(generate_random_test_data=True)


def test_filtering_on_ts_delivery():
    open_bids_db, open_offers_db = db_obj.get_open_positions()

    market_horizon = config['lem']['horizon_clearing']
    interval_clearing = config['lem']['interval_clearing']
    # Calculate number of market clearings
    n_clearings = int(market_horizon / interval_clearing)
    print("n_clearings: " + str(n_clearings))
    t_clearing_first = min(min(open_bids_db[db_obj.db_param.TS_DELIVERY]),
                           min(open_bids_db[db_obj.db_param.TS_DELIVERY]))
    iterations = range(1, n_clearings + 1)

    for i in iterations:
        print("i: " + str(i))

        t_clearing_current = t_clearing_first + interval_clearing * i
        print("t_clearing_current: " + str(t_clearing_current))

        curr_clearing_offers_db = open_offers_db[open_offers_db[db_obj.db_param.TS_DELIVERY] == t_clearing_current]
        curr_clearing_bids_db = open_bids_db[open_bids_db[db_obj.db_param.TS_DELIVERY] == t_clearing_current]

        # Aggregate equal positions
        if not curr_clearing_bids_db.empty:
            curr_clearing_bids_db = _aggregate_identical_positions(db_obj=db_obj,
                                                                   positions=curr_clearing_bids_db,
                                                                   subset=[db_obj.db_param.PRICE_ENERGY,
                                                                           db_obj.db_param.QUALITY_ENERGY,
                                                                           db_obj.db_param.ID_USER])
        if not curr_clearing_offers_db.empty:
            curr_clearing_offers_db = _aggregate_identical_positions(db_obj=db_obj,
                                                                     positions=curr_clearing_offers_db,
                                                                     subset=[db_obj.db_param.PRICE_ENERGY,
                                                                             db_obj.db_param.QUALITY_ENERGY,
                                                                             db_obj.db_param.ID_USER])

        curr_clearing_offers_db = _convert_qualities_to_int(db_obj, curr_clearing_offers_db, config['lem']['types_quality'])
        curr_clearing_offers_db = curr_clearing_offers_db.sort_values(
            by=[db_obj.db_param.PRICE_ENERGY, db_obj.db_param.QUALITY_ENERGY, db_obj.db_param.QTY_ENERGY],
            ascending=[True, False, False])
        curr_clearing_bids_db = _convert_qualities_to_int(db_obj, curr_clearing_bids_db,
                                                            config['lem']['types_quality'])
        curr_clearing_bids_db = curr_clearing_bids_db.sort_values(by=[db_obj.db_param.PRICE_ENERGY, db_obj.db_param.QUALITY_ENERGY, db_obj.db_param.QTY_ENERGY],
                                                                  ascending=[False, False, False])

        curr_clearing_offers_db, curr_clearing_bids_db = [tuple(x) for x in curr_clearing_offers_db.values.tolist()], [tuple(x) for x in curr_clearing_bids_db.values.tolist()]

        print("len curr_clearing_offers_db: " + str(len(curr_clearing_offers_db)))
        print("len curr_clearing_bids_db: " + str(len(curr_clearing_bids_db)))
        curr_clearing_offers_blockchain, curr_clearing_bids_blockchain = \
            bc_obj_clearing_ex_ante.functions.filter_sort_aggregate_OffersBids_memory(t_clearing_current, True).call()

        assert curr_clearing_offers_blockchain == curr_clearing_offers_db
        assert curr_clearing_bids_blockchain == curr_clearing_bids_db

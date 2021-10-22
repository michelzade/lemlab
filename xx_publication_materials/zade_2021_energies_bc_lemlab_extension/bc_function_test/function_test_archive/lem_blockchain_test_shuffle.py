import pytest
import test_utils

config = None
db_obj = None
bc_obj_clearing_ex_ante = None


@pytest.fixture(scope="session", autouse=True)
def setup():
    global config, db_obj, bc_obj_clearing_ex_ante
    config, db_obj, bc_obj_clearing_ex_ante, _ = test_utils.setup_test_general(generate_random_test_data=True)


def test_shuffling():

    offers_bc = bc_obj_clearing_ex_ante.get_open_offers
    bids_bc = open_bids_blockchain

    offers_shuffled_bc = bc_obj_clearing_ex_ante.functions.shuffle_OfferBids(offers_bc).call()
    bids_shuffled_bc = bc_obj_clearing_ex_ante.functions.shuffle_OfferBids(bids_bc).call()

    assert len(offers_bc) == len(offers_shuffled_bc) and len(bids_bc) == len(
        bids_shuffled_bc)
    assert offers_bc != offers_shuffled_bc
    assert bids_bc != bids_shuffled_bc
    assert set([tuple(x) for x in offers_bc]) == set([tuple(x) for x in offers_shuffled_bc])
    assert set([tuple(x) for x in bids_bc]) == set([tuple(x) for x in bids_shuffled_bc])
    # test_utils.result_test(__file__, True)

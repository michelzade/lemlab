import os
import time
import subprocess
from xx_publication_materials.zade_2022_iewt_bc_extension.bc_evaluation.test_utils import setup_test_general


def kill_process_by_name(process_name):
    try:
        # Get PID of process
        pid_list = list(map(int, subprocess.check_output(["pidof", "openethereum"]).split()))
        for pid in pid_list:
            # Kill process
            print(f"Killing process pid {pid}")
            os.system(f"kill {pid}")
        # Wait for 10 seconds until process is cleanly killed
        time.sleep(30)
    except subprocess.CalledProcessError as cpe:
        print(f"Process {process_name} was not running.")
        print(cpe)
        raise cpe


def start_openethereum_node(path_toml_file):
    # Start openethereum node

    # # Option #1
    # subprocess.Popen(["nohup", "openethereum", "--config", path_toml_file], stdout=subprocess.DEVNULL,
    #                  stderr=subprocess.DEVNULL)
    # Option #2
    os.system(f"openethereum --config {path_toml_file} &")

    time.sleep(60)


def restart_openethereum_node(path_toml_file):
    process = "openethereum"
    try:
        kill_process_by_name(process_name=process)
    except subprocess.CalledProcessError as cpe:
        print(f"{process} was not running, trying to start node anyway.")
        time.sleep(10)

    start_openethereum_node(path_toml_file=path_toml_file)


if __name__ == '__main__':
    path_toml_file = "../../../oeEBL/oeEBL.toml"
    process = "openethereum"
    # Start openethereum node
    start_openethereum_node(path_toml_file=path_toml_file)
    # Restart openethereum node
    restart_openethereum_node(path_toml_file=path_toml_file)
    # Create connection objects
    config, db_obj, bc_obj_clearing_ex_ante, bc_obj_settlement = setup_test_general(generate_random_test_data=False,
                                                                                    test_config_path="xx_publication_materials/zade_2022_iewt_bc_extension/bc_evaluation/sim_test_config.yaml")
    # Trigger sample tx
    tx_hash = bc_obj_clearing_ex_ante.web3_eth.send_transaction({"from": bc_obj_clearing_ex_ante.web3_eth.coinbase,
                                                                 "to": bc_obj_clearing_ex_ante.web3_eth.coinbase,
                                                                 "value": 100})
    # Wait for tx receipt
    bc_obj_clearing_ex_ante.web3_eth.wait_for_transaction_receipt(tx_hash, timeout=10)
    # Kill openethereum
    kill_process_by_name(process_name=process)


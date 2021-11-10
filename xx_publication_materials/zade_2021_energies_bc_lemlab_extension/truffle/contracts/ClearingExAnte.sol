pragma solidity >=0.5.0 <0.7.5;
pragma experimental ABIEncoderV2;

import "./Param.sol";
import "./Sorting.sol";

contract ClearingExAnte {

	event logString(string arg); // produces a log at the end of a transaction and can be used for debugging (not live)
	Lb.LemLib.offer_bid[] offers; //list of total offers stored. they don't get deleted unless one reset the contract
	Lb.LemLib.offer_bid[] bids; //list of total bids stored. they don't get deleted unless one reset the contract
	Lb.LemLib.user[] user_infos;
	Lb.LemLib.meter[] meter_infos;
	mapping(string => string) map_to_user_id;
	mapping(string => string) map_to_main_meter_id;
	/*list of temporary offers stored.
	They are relative to each market clearing and for every market clearing they might be deleted.
	Now the deletion is performed via Web3.py in python before pushing new ones.*/
	Lb.LemLib.offer_bid[] temp_offers;
	/*list of temporary bids stored.
	They are relative to each market clearing and for every market clearing they might be deleted.
	Now the deletion is performed via Web3.py in python before pushing new ones.*/
	Lb.LemLib.offer_bid[] temp_bids;
	Lb.LemLib.market_result[] temp_market_results;//market results for each single clearing(for each specific t_clearing_current)
	Lb.LemLib.market_result_total[] market_results_total;//whole market results relative to the whole market clearing(96 loops)
	string string_to_log = "";//string used for the event logString
	Param p = new Param();//instance of the contract Param
	Lb.LemLib lib= new Lb.LemLib();//instance of the contract LemLib(general library with useful functionalities)
	Sorting srt = new Sorting();//instance of the contract Sorting(useful sorting functionalities)

	constructor() public{
		ClearingExAnte.clear_temp_data();	// clears all temp data
		}

	function get_own_address() public view returns(address){
		return address(this);
	}

	function clear_temp_data() public {//function that deletes objects from the contract storage
	    delete ClearingExAnte.temp_offers;
		delete ClearingExAnte.temp_bids;
		delete ClearingExAnte.temp_market_results;
		delete ClearingExAnte.market_results_total;
	}

	function clear_permanent_data() public {//function that deletes objects from the contract storage
		delete ClearingExAnte.user_infos;
		delete ClearingExAnte.meter_infos;
		delete ClearingExAnte.offers;
		delete ClearingExAnte.bids;
	}

	// Deletes objects from non static market tables
	function clear_dynamic_market_data_tables() public {
		delete ClearingExAnte.temp_offers;
		delete ClearingExAnte.temp_bids;
		delete ClearingExAnte.temp_market_results;
		delete ClearingExAnte.market_results_total;
		delete ClearingExAnte.offers;
		delete ClearingExAnte.bids;
	}

	/*
	same function as clearTempData(). It is used when clearTempData() exceeds the gas limit.
	In this case, variables have to be deleted by chunks. Reducing automatically the length deletes the last element and costs less gas
	*/
	function clear_temp_data_gas_limit(uint max_entries) public {
	    for(uint i = 0; i < max_entries; i++){
	    	if(ClearingExAnte.temp_offers.length > 0){
	    		ClearingExAnte.temp_offers.pop();
	    	}
	    	if(ClearingExAnte.temp_bids.length > 0){
	    		ClearingExAnte.temp_bids.pop();
	    	}
	    	if(ClearingExAnte.temp_market_results.length > 0){
	    		ClearingExAnte.temp_market_results.pop();
	    	}
	    }
	}

	/*
	similar function to clearTempData_gas_limit(). It is used when clearPermanentData() exceeds the gas limit.
	In this case, variables have to be deleted by chunks
	*/
	function clear_permanent_data_gas_limit(uint max_entries) public {
	    for(uint i = 0; i < max_entries; i++){
	    	if(ClearingExAnte.offers.length > 0){
	   // 		delete ClearingExAnte.offers[ClearingExAnte.offers.length-1];
	    		ClearingExAnte.offers.pop();
	    	}
	    	if(ClearingExAnte.bids.length > 0){
	   // 		delete ClearingExAnte.bids[ClearingExAnte.bids.length-1];
	    		ClearingExAnte.bids.pop();
	    	}
	    	if(ClearingExAnte.market_results_total.length > 0){
	   // 		delete ClearingExAnte.market_results_total[ClearingExAnte.market_results_total.length - 1];
	    		ClearingExAnte.market_results_total.pop();
	    	}
	    	if(ClearingExAnte.user_infos.length > 0){
	   // 		delete ClearingExAnte.user_infos[ClearingExAnte.user_infos.length - 1];
	    		ClearingExAnte.user_infos.pop();
	    	}
	    	if(ClearingExAnte.meter_infos.length > 0){
	   // 		delete ClearingExAnte.id_meters[ClearingExAnte.id_meters.length - 1];
	    		ClearingExAnte.meter_infos.pop();
	    	}
	    }
	}

	//add an offer or bid to the list of temporary and/or permanent offers in the storage of the contract
	function push_offer_bid(Lb.LemLib.offer_bid memory ob, bool offer, bool temp, bool permanent) public {
	    if(offer) push_offer(ob, temp, permanent);
		else push_bid(ob, temp, permanent);
	}

	//add an offer to the lists of temporary and/or permanent offers in the storage of the contract
	function push_offer(Lb.LemLib.offer_bid memory offer, bool temp, bool permanent) private {
		if(temp) ClearingExAnte.temp_offers.push(offer);
		if(permanent) ClearingExAnte.offers.push(offer);
	}

	//add an bid to the lists of temporary and/or permanent bids in the storage of the contract
	function push_bid(Lb.LemLib.offer_bid memory bid, bool temp, bool permanent) private {
		if(temp) ClearingExAnte.temp_bids.push(bid);
		if(permanent) ClearingExAnte.bids.push(bid);
	}

	//add a user info to the lists of user_infos in the storage of the contract
	function push_user_info(Lb.LemLib.user memory user_info) public {
	    ClearingExAnte.user_infos.push(user_info);
		map_to_user_id[user_info.id_market_agent] = user_info.id_user;
		map_to_user_id[user_info.id_user] = user_info.id_user;
	}

	//add a id_meter to the lists of id_meters in the storage of the contract
	function push_meter_info(Lb.LemLib.meter memory meter_info, bool main_meter) public {
		ClearingExAnte.meter_infos.push(meter_info);
		map_to_user_id[meter_info.id_meter] = meter_info.id_user;
		map_to_user_id[meter_info.id_user] = meter_info.id_user;
		if(main_meter) {
			map_to_main_meter_id[meter_info.id_meter] = meter_info.id_meter;
			map_to_main_meter_id[meter_info.id_user] = meter_info.id_meter;
		}
	}

	//gets the list of user_infos in the storage of the contract
	function get_user_infos() public view returns (Lb.LemLib.user[] memory) {
		return ClearingExAnte.user_infos;
	}

	//gets the list of id_meters in the storage of the contract
	function get_info_meters() public view returns (Lb.LemLib.meter[] memory) {
		return ClearingExAnte.meter_infos;
	}

	function get_main_meter_id(string memory id) public view returns (string memory main_meter_id) {
		return map_to_main_meter_id[id];
	}

	function map_id_to_main_meter_id(string memory id, string memory main_meter_id) public {
		map_to_main_meter_id[id] = main_meter_id;
	}

	function get_user_id(string memory id) public view returns(string memory){
		return map_to_user_id[id];
	}

	//function to delete a single user from the chain, for that, we push the last one into
	// that position and delete the last (decreasing the length auto deletes the last element at a cheaper gas cost), to not leave any empty spot
	// the function can additionally delete all the meters pointing to that user
	function delete_user(Lb.LemLib.user memory user, bool del_meters) public {
		for(uint i = 0; i < ClearingExAnte.user_infos.length; i++){
			if(lib.compare_strings(ClearingExAnte.user_infos[i].id_user, user.id_user)){
				// ClearingExAnte.user_infos[i]=ClearingExAnte.user_infos[ClearingExAnte.user_infos.length-1];
				ClearingExAnte.user_infos.pop();
				break;
			}
		}
		if (del_meters){
			// same as delete_meter, but in this case we do not break, as there may be many meters for one user
			for(uint j=0; j<ClearingExAnte.meter_infos.length; j++){
				if(lib.compare_strings(ClearingExAnte.meter_infos[j].id_user, user.id_user)){
				// 	ClearingExAnte.id_meters[j]=ClearingExAnte.id_meters[ClearingExAnte.id_meters.length-1];
					ClearingExAnte.meter_infos.pop();
				}
			}
		}
	}

	//function to delete a single meter from the chain
	function delete_meter(Lb.LemLib.meter memory meter) public{
		for(uint i=0; i<ClearingExAnte.meter_infos.length; i++){
			if(lib.compare_strings(ClearingExAnte.meter_infos[i].id_meter, meter.id_meter)){
				// ClearingExAnte.id_meters[i]=ClearingExAnte.id_meters[ClearingExAnte.id_meters.length-1];
				ClearingExAnte.meter_infos.pop();
				break;
			}
		}
	}

	//gets the list of temporary or permanent offers
	function get_offers(bool temp) public view returns(Lb.LemLib.offer_bid[] memory) {
	    if(temp) return ClearingExAnte.temp_offers;
		else return ClearingExAnte.offers;
	}

	//gets the list of temporary or permanent bids
    function get_bids(bool temp) public view returns(Lb.LemLib.offer_bid[] memory) {
		if(temp) return ClearingExAnte.temp_bids;
		else return ClearingExAnte.bids;
    }

    //gets the total market results
	function get_market_results_total() public view returns (Lb.LemLib.market_result_total[] memory) {
	    return ClearingExAnte.market_results_total;
	}

	//gets the temporary market results
	function get_temp_market_results() public view returns (Lb.LemLib.market_result[] memory) {
	    return ClearingExAnte.temp_market_results;
	}

	//returns a filtering of the temporary positions(offer/bids). The ones having the ts_delivery == t_clearing_current and t_clearing_current in the limits of the ts_delivery first and last of the user
	function filtered_offers_bids_ts_delivery_user(uint t_clearing_current) public view returns (Lb.LemLib.offer_bid[] memory, Lb.LemLib.offer_bid[] memory){
		uint len = 0;
		uint j = 0;
		for (uint i=0; i<ClearingExAnte.temp_offers.length; i++) {
			if ( ClearingExAnte.temp_offers[i].ts_delivery == t_clearing_current && lib.check_user_id_in_user_infos_interval(ClearingExAnte.temp_offers[i].id_user, t_clearing_current, ClearingExAnte.user_infos)) {
		    	len++;
			}
		}

		Lb.LemLib.offer_bid[] memory filtered_offers = new Lb.LemLib.offer_bid[](len);

		for (uint i=0; i<ClearingExAnte.temp_offers.length; i++) {
			if ( ClearingExAnte.temp_offers[i].ts_delivery == t_clearing_current && lib.check_user_id_in_user_infos_interval(ClearingExAnte.temp_offers[i].id_user, t_clearing_current, ClearingExAnte.user_infos)) {
		    	filtered_offers[j] = ClearingExAnte.temp_offers[i];
		    	j++;
			}
	    }

	    len = 0;
	    j=0;

		for (uint i=0; i<ClearingExAnte.temp_bids.length; i++) {
			if ( ClearingExAnte.temp_bids[i].ts_delivery == t_clearing_current && lib.check_user_id_in_user_infos_interval(ClearingExAnte.temp_bids[i].id_user, t_clearing_current, ClearingExAnte.user_infos)) {
		    	len++;
			}
		}

		Lb.LemLib.offer_bid[] memory filtered_bids = new Lb.LemLib.offer_bid[](len);

		for (uint i=0; i<ClearingExAnte.temp_bids.length; i++) {
			if ( ClearingExAnte.temp_bids[i].ts_delivery == t_clearing_current && lib.check_user_id_in_user_infos_interval(ClearingExAnte.temp_bids[i].id_user, t_clearing_current, ClearingExAnte.user_infos)) {
		    	filtered_bids[j] = ClearingExAnte.temp_bids[i];
		    	j++;
			}
		}
	    return (filtered_offers, filtered_bids);
	}

	//returns a filtering of the temporary positions(offer/bids). The ones having the ts_delivery == t_clearing_current and quantity of energy > 0
	function get_filtered_offers_bids_memory(uint t_clearing_current) public view returns (Lb.LemLib.offer_bid[] memory, Lb.LemLib.offer_bid[] memory){
	    uint len = 0;
		uint j = 0;
		for (uint i=0; i<ClearingExAnte.temp_offers.length; i++) {
			if ( ClearingExAnte.temp_offers[i].ts_delivery == t_clearing_current && ClearingExAnte.temp_offers[i].qty_energy > 0) {
		    	len++;
			}
		}

		Lb.LemLib.offer_bid[] memory filtered_offers = new Lb.LemLib.offer_bid[](len);

		for (uint i=0; i<ClearingExAnte.temp_offers.length; i++) {
			if ( ClearingExAnte.temp_offers[i].ts_delivery == t_clearing_current && ClearingExAnte.temp_offers[i].qty_energy > 0) {
		    	filtered_offers[j] = ClearingExAnte.temp_offers[i];
		    	j++;
			}
	    }

	    len = 0;
	    j=0;

		for (uint i=0; i<ClearingExAnte.temp_bids.length; i++) {
			if ( ClearingExAnte.temp_bids[i].ts_delivery == t_clearing_current && ClearingExAnte.temp_bids[i].qty_energy > 0) {
		    	len++;
			}
		}

		Lb.LemLib.offer_bid[] memory filtered_bids = new Lb.LemLib.offer_bid[](len);

		for (uint i=0; i<ClearingExAnte.temp_bids.length; i++) {
			if ( ClearingExAnte.temp_bids[i].ts_delivery == t_clearing_current && ClearingExAnte.temp_bids[i].qty_energy > 0) {
		    	filtered_bids[j] = ClearingExAnte.temp_bids[i];
		    	j++;
			}
		}
	    return (filtered_offers, filtered_bids);
	}

	//same as getFilteredOffersBids_memory(). takes more gas than that though
	function get_filtered_offers_bids_memory_two(uint t_clearing_current) public view returns (Lb.LemLib.offer_bid[] memory, Lb.LemLib.offer_bid[] memory){
		Lb.LemLib.offer_bid[] memory filtered_offers = new Lb.LemLib.offer_bid[](ClearingExAnte.temp_offers.length);
		uint j = 0;
		for (uint i=0; i<ClearingExAnte.temp_offers.length; i++) {
			if ( ClearingExAnte.temp_offers[i].ts_delivery == t_clearing_current ) {
		    	filtered_offers[j] = ClearingExAnte.temp_offers[i];
		    	j++;
			}
	    }
	    filtered_offers = lib.crop_bids(filtered_offers, 0, j-1);
		Lb.LemLib.offer_bid[] memory filtered_bids = new Lb.LemLib.offer_bid[](ClearingExAnte.temp_bids.length);
		j = 0;

		for (uint i=0; i<ClearingExAnte.temp_bids.length; i++) {
			if ( ClearingExAnte.temp_bids[i].ts_delivery == t_clearing_current ) {
		    	filtered_bids[j] = ClearingExAnte.temp_bids[i];
		    	j++;
			}
		}
		filtered_bids = lib.crop_bids(filtered_bids, 0, j-1);
	    return (filtered_offers, filtered_bids);
	}

	//same as getFilteredOffersBids_memory(). it also sorts the position by price
	function filter_sort_offers_bids_memory(uint t_clearing_current) public view returns(Lb.LemLib.offer_bid[] memory, Lb.LemLib.offer_bid[] memory) {
	    Lb.LemLib.offer_bid[] memory filtered_offers;
	    Lb.LemLib.offer_bid[] memory filtered_bids;
	    (filtered_offers, filtered_bids) = get_filtered_offers_bids_memory(t_clearing_current);
	    filtered_offers = srt.quick_sort_offers_bids_price(filtered_offers, true);
	    filtered_bids = srt.quick_sort_offers_bids_price(filtered_bids, false);
	    return (filtered_offers, filtered_bids);
	}

	//similar to filter_sort_OffersBids_memory(). It also sorts by quality(and optional quantity) and aggregate identical positions
	function filter_sort_aggregate_offers_bids_memory(uint t_clearing_current, bool simulation_test) public view returns(Lb.LemLib.offer_bid[] memory, Lb.LemLib.offer_bid[] memory) {
	    Lb.LemLib.offer_bid[] memory filtered_offers;
	    Lb.LemLib.offer_bid[] memory filtered_bids;
	    (filtered_offers, filtered_bids) = get_filtered_offers_bids_memory(t_clearing_current);

	    filtered_offers = srt.aggregate_identical_positions(filtered_offers, simulation_test);
	    filtered_bids = srt.aggregate_identical_positions(filtered_bids, simulation_test);

	    filtered_offers = srt.insertion_sort_offers_bids_price_quality(filtered_offers, true, false, simulation_test, false);
	    filtered_bids = srt.insertion_sort_offers_bids_price_quality(filtered_bids, false, false, simulation_test, false);
	    return (filtered_offers, filtered_bids);
	}

	//add one supplier bid and one supplier offer to the lists of offers and bids given in input. then it returns them.
	function add_retailer_memory(uint t_clearing_current, Lb.LemLib.offer_bid[] memory filtered_offers, Lb.LemLib.offer_bid[] memory filtered_bids) public view returns(Lb.LemLib.offer_bid[] memory, Lb.LemLib.offer_bid[] memory) {
	    Lb.LemLib.offer_bid memory supOfferBid = Lb.LemLib.offer_bid({id_user: p.get_id_retailer(), qty_energy: p.get_qty_offer_retailer(), price_energy: p.get_price_offer_retailer(), quality_energy:0, type_position:"0", premium_preference_quality: p.get_premium_preference_quality(), number_position:0, status_position:0, t_submission:t_clearing_current, ts_delivery:t_clearing_current});

		Lb.LemLib.offer_bid[] memory filtered_offers_sup = new Lb.LemLib.offer_bid[](filtered_offers.length+1);
		filtered_offers_sup[0] = supOfferBid;
		for(uint i=0; i<filtered_offers.length; i++) {
		    filtered_offers_sup[i+1] = filtered_offers[i];
		}

	    supOfferBid = Lb.LemLib.offer_bid({ts_delivery:t_clearing_current, price_energy: p.get_price_bid_retailer(), number_position:0, t_submission:t_clearing_current, id_user: p.get_id_retailer(), qty_energy: p.get_qty_bid_retailer(), status_position:0, type_position:"1", quality_energy:0, premium_preference_quality: p.get_premium_preference_quality()});

	    Lb.LemLib.offer_bid[] memory filtered_bids_sup = new Lb.LemLib.offer_bid[](filtered_bids.length+1);
		filtered_bids_sup[0] = supOfferBid;
		for(uint i=0; i<filtered_bids.length; i++) {
		    filtered_bids_sup[i+1] = filtered_bids[i];
		}
	    return (filtered_offers_sup, filtered_bids_sup);
	}

	//single clearing, relative for a specific t_clearing_current. At the end it can push the results temp_market_results and temp_market_result_total
	function single_clearing(uint t_clearing_current, bool add_retailer, bool uniform_pricing, bool discriminative_pricing, uint t_cleared, bool writeTempMarketResult, bool writeFinalMarketResult, bool verbose, bool shuffle, bool simulation_test) public {
		if(writeTempMarketResult) delete ClearingExAnte.temp_market_results;

		Lb.LemLib.offer_bid[] memory filtered_offers;
	    Lb.LemLib.offer_bid[] memory filtered_bids;
	    //filtering offers and bids by t_clearing_current
	    (filtered_offers, filtered_bids) = get_filtered_offers_bids_memory(t_clearing_current);
	    //aggregating offers and bids by with same price, user_id, quality
	    filtered_offers = srt.aggregate_identical_positions(filtered_offers, simulation_test);
	   	filtered_bids = srt.aggregate_identical_positions(filtered_bids, simulation_test);

	    //Check whether the flag retailer bids is true
    	//Insert retailer bids and offers
        if (add_retailer) {
        	(filtered_offers, filtered_bids) = ClearingExAnte.add_retailer_memory(t_clearing_current, filtered_offers, filtered_bids);
        }

        //Check whether offers or bids are empty
        if( filtered_offers.length == 0 || filtered_bids.length == 0 ) {
        	if(verbose) string_to_log = lib.concatenate_strings(string_to_log,"\tNo clearing - supply and/or bids are empty\n");
        }

        else { //Offers and bids are not empty

			if(verbose) string_to_log = lib.concatenate_strings(string_to_log,"\tLength of offers and bids > 0. Starting clearing\n");
            if(shuffle) {//shuffling positions
		    	filtered_offers = lib.shuffle_bids(filtered_offers);
		    	filtered_bids = lib.shuffle_bids(filtered_bids);
	    	}
	    	//sorting positions by price, quality, and possibly quantity
            filtered_offers = srt.insertion_sort_offers_bids_price_quality(filtered_offers, true, false, simulation_test, false);
	    	filtered_bids = srt.insertion_sort_offers_bids_price_quality(filtered_bids, false, false, simulation_test, false);

	    	if(verbose) {
	    		string_to_log = lib.concatenate_strings(lib.concatenate_strings(string_to_log,lib.concatenate_strings("\tOffers length: ",lib.uint_to_string(filtered_offers.length))),"\n");
				string_to_log = lib.concatenate_strings(lib.concatenate_strings(string_to_log,lib.concatenate_strings("\tBids length: ",lib.uint_to_string(filtered_bids.length))),"\n");
	    	}

            Lb.LemLib.market_result[] memory tmp_market_results;

            //merging filtered offers and bids
            tmp_market_results = merge_offers_bids_memory(filtered_offers, filtered_bids);
            if(verbose) string_to_log = lib.concatenate_strings(lib.concatenate_strings(string_to_log,lib.concatenate_strings("\tMerge offers/bid length: ",lib.uint_to_string(tmp_market_results.length))),"\n");

            //calculating uniform and discriminative pricing
            tmp_market_results = calc_market_clearing_prices(tmp_market_results, uniform_pricing, discriminative_pricing);

            if(writeTempMarketResult) {
            	for(uint i = 0; i < tmp_market_results.length; i++) {
                	ClearingExAnte.temp_market_results.push(tmp_market_results[i]);
            	}
            }
            if(verbose) string_to_log = lib.concatenate_strings(string_to_log,"\tCalculated clearing prices\n");

            //Check whether market has cleared a volume
            if(writeFinalMarketResult && tmp_market_results.length > 0) {
            	//time costly approach!
            	//Challenge storage c = challenges[challenges.length - 1];
            	Lb.LemLib.market_result_total memory temp_market_result_total;
            	for(uint i = 0; i<tmp_market_results.length; i++) {
            	    temp_market_result_total = Lb.LemLib.market_result_total(
            	        {
                        id_user_offer:tmp_market_results[i].id_user_offer,
                        price_energy_offer:tmp_market_results[i].price_energy_offer,
                        number_position_offer:tmp_market_results[i].number_position_offer,
                        ts_delivery:tmp_market_results[i].ts_delivery,
                        id_user_bid:tmp_market_results[i].id_user_bid,
                        price_energy_bid:tmp_market_results[i].price_energy_bid,
                        number_position_bid:tmp_market_results[i].number_position_bid,
                        price_energy_market_uniform:tmp_market_results[i].price_energy_market_uniform,
                        price_energy_market_discriminative:tmp_market_results[i].price_energy_market_discriminative,
                        qty_energy_traded:tmp_market_results[i].qty_energy_traded,
                        share_quality_NA:tmp_market_results[i].share_quality_NA,
        				share_quality_local:tmp_market_results[i].share_quality_local,
        				share_quality_green:tmp_market_results[i].share_quality_green,
        				share_quality_green_local:tmp_market_results[i].share_quality_green_local,
                        t_cleared:t_cleared
            	    });
            	    ClearingExAnte.market_results_total.push(temp_market_result_total);
            	}
            }
            else if (verbose) {
                string_to_log=lib.concatenate_strings(string_to_log,"\tMarket Volume == 0 or empty market results for this clearing\n");
                string_to_log=lib.concatenate_strings(lib.concatenate_strings(string_to_log,lib.concatenate_strings("\tMarket Results length:",lib.uint_to_string(ClearingExAnte.temp_market_results.length))),"\n");
            }
        }
	}

	//same as single_clearing(). It performs the operations in memory and return the temp market results(i.e. for this single clearing)
	function single_clearing_memory(uint t_clearing_current, bool add_retailer, bool uniform_pricing, bool discriminative_pricing, bool shuffle, bool simulation_test) public view returns(Lb.LemLib.market_result[] memory){
		Lb.LemLib.offer_bid[] memory filtered_offers;
	    Lb.LemLib.offer_bid[] memory filtered_bids;
	    (filtered_offers, filtered_bids) = get_filtered_offers_bids_memory(t_clearing_current);
	    //(filtered_offers, filtered_bids) = filteredOffersBids_ts_delivery_user(t_clearing_current);

	    filtered_offers = srt.aggregate_identical_positions(filtered_offers, simulation_test);
	   	filtered_bids = srt.aggregate_identical_positions(filtered_bids, simulation_test);

	    //Check whether this is the first clearing period and whether the flag retailer bids is true
    	//Insert retailer bids and offers
        if (add_retailer) {
        	(filtered_offers, filtered_bids) = ClearingExAnte.add_retailer_memory(t_clearing_current, filtered_offers, filtered_bids);
        }
        if(filtered_offers.length > 0 && filtered_bids.length > 0) { //Offers and bids are not empty and last update in offers/bids is newer than last clearing time
            if(shuffle) {
		    	filtered_offers = lib.shuffle_bids(filtered_offers);
		    	filtered_bids = lib.shuffle_bids(filtered_bids);
	    	}

            filtered_offers = srt.insertion_sort_offers_bids_price_quality(filtered_offers, true, false, simulation_test, false);
	    	filtered_bids = srt.insertion_sort_offers_bids_price_quality(filtered_bids, false, false, simulation_test, false);

            Lb.LemLib.market_result[] memory tmp_market_results = merge_offers_bids_memory(filtered_offers, filtered_bids);

            tmp_market_results = calc_market_clearing_prices(tmp_market_results, uniform_pricing, discriminative_pricing);

            return temp_market_results;
        }
        return new Lb.LemLib.market_result[](0);
	}

	//it performs the merge between a list of offers, and a list of bids. it produces an object of the type market_result.
	function merge_offers_bids_memory(Lb.LemLib.offer_bid[] memory filtered_offers, Lb.LemLib.offer_bid[] memory filtered_bids) public view returns(Lb.LemLib.market_result[] memory) {
	    //Insert cumulated bid energy into tables
	    uint[] memory energy_cumulated_offers = lib.get_energy_cumulated(filtered_offers);
	    uint[] memory energy_cumulated_bids = lib.get_energy_cumulated(filtered_bids);

	    //merge bids and offers

	    uint i = 0;
	    uint j = 0;
	    uint z = 0;
	    uint energy_cumulated;
	    Lb.LemLib.market_result memory merge;
	    //max length of results == filtered_offers.length + filtered_bids.length
	    Lb.LemLib.market_result[] memory temp_market_results_m = new Lb.LemLib.market_result[](filtered_offers.length + filtered_bids.length);
	    uint[] memory energy_cumulated_finals = new uint[](filtered_offers.length + filtered_bids.length);

	    while(i < energy_cumulated_offers.length && j < energy_cumulated_bids.length) {
	            if (energy_cumulated_offers[i] <= energy_cumulated_bids[j]) {
	                energy_cumulated = energy_cumulated_offers[i];
	            }
	            else {
	                energy_cumulated = energy_cumulated_bids[j];
	            }
	        merge = Lb.LemLib.market_result(
	                {
	                    id_user_offer:filtered_offers[i].id_user,
	                    qty_energy_offer:0,
	                    price_energy_offer:filtered_offers[i].price_energy,
	                    quality_energy_offer:filtered_offers[i].quality_energy,
	                    premium_preference_quality_offer:filtered_offers[i].premium_preference_quality,
	                    type_position_offer:filtered_offers[i].type_position,
	                    number_position_offer:filtered_offers[i].number_position,
	                    status_position_offer:filtered_offers[i].status_position,
	                    t_submission_offer:filtered_offers[i].t_submission,
	                    ts_delivery:filtered_offers[i].ts_delivery,
	                    id_user_bid:filtered_bids[j].id_user,
	                    qty_energy_bid:0,
	                    price_energy_bid:filtered_bids[j].price_energy,
	                    quality_energy_bid:filtered_bids[j].quality_energy,
	                    premium_preference_quality_bid:filtered_bids[j].premium_preference_quality,
	                    type_position_bid:filtered_bids[j].type_position,
	                    number_position_bid:filtered_bids[j].number_position,
	                    status_position_bid:filtered_bids[j].status_position,
	                    t_submission_bid:filtered_bids[j].t_submission,
	                    price_energy_market_uniform:0,
	                    price_energy_market_discriminative:0,
	                    qty_energy_traded:0,
	                    share_quality_NA:0,
	                    share_quality_local:0,
	                    share_quality_green:0,
	                    share_quality_green_local:0
	                });

	        if (energy_cumulated_offers[i] == energy_cumulated_bids[j]) {
	            i += 1;
	            j += 1;
	        }
	        else {
	                if (energy_cumulated_offers[i] < energy_cumulated_bids[j]) {
	                    i += 1;
	                }
	                else {
	                    j += 1;
	                }
	        }
	        //extract all merged bids and offers where offer price is lower or equal than the bid price
	        if( merge.price_energy_offer <= merge.price_energy_bid ) {
	            temp_market_results_m[z] = merge;
	            energy_cumulated_finals[z] = energy_cumulated;
	            z++;
	        }
	    }

	    //z basically equals the length

	   	if(z <= 0) {
	    	return new Lb.LemLib.market_result[](0);
	    }
	    //calculating the differences in the array of energy cumulated
	    uint[] memory qties_energy_traded = new uint[](z);
	    uint[] memory qtys_difference = lib.compute_differences(energy_cumulated_finals, 0, z - 1);
	    qties_energy_traded[0] = energy_cumulated_finals[0];
	    //assigning those differences to the array of energy traded
	    for(i = 1; i < qties_energy_traded.length; i++) {
	    	qties_energy_traded[i] = qtys_difference[i-1];
	    }

	    //modify temp_market_results_m length
        Lb.LemLib.market_result[] memory temp_market_results_final = new Lb.LemLib.market_result[](z);
        for(i = 0; i < z; i++) {
            temp_market_results_final[i] = temp_market_results_m[i];
            temp_market_results_final[i].qty_energy_offer = qties_energy_traded[i];
            temp_market_results_final[i].qty_energy_bid = qties_energy_traded[i];
            temp_market_results_final[i].qty_energy_traded = qties_energy_traded[i];
        }
        temp_market_results_final = add_quality_shares(temp_market_results_final);

	    return temp_market_results_final;
	}

	//it calculates the share of source of energy, for every single match offer/bid
	function add_quality_shares(Lb.LemLib.market_result[] memory temp_market_results_m) public pure returns(Lb.LemLib.market_result[] memory) {
		//{0: 'NA', 1: 'local', 2: 'green', 3: 'green_local'}
        uint qty_traded = 0;
        uint total_qty_traded = 0;
        uint total_qty_type_na = 0;
        uint total_qty_type_local = 0;
        uint total_qty_type_green = 0;
        uint total_qty_type_green_local = 0;
        for(uint i = 0; i < temp_market_results_m.length; i++) {
        	qty_traded = temp_market_results_m[i].qty_energy_traded;
        	total_qty_traded = total_qty_traded + qty_traded;
    		if(temp_market_results_m[i].quality_energy_offer == 0) total_qty_type_na = total_qty_type_na + qty_traded;
    		else if(temp_market_results_m[i].quality_energy_offer == 1) total_qty_type_local = total_qty_type_local + qty_traded;
    		else if(temp_market_results_m[i].quality_energy_offer == 2) total_qty_type_green = total_qty_type_green + qty_traded;
    		else if(temp_market_results_m[i].quality_energy_offer == 3) total_qty_type_green_local = total_qty_type_green_local + qty_traded;
    	}
		uint share_quality_local = (total_qty_type_local * 100) / total_qty_traded;
		uint share_quality_green = (total_qty_type_green * 100) / total_qty_traded;
		uint share_quality_green_local = (total_qty_type_green_local * 100) / total_qty_traded;
		uint share_quality_NA = 100 - (share_quality_local + share_quality_green + share_quality_green_local);
    	for(uint i = 0; i < temp_market_results_m.length; i++) {
    		temp_market_results_m[i].share_quality_NA = share_quality_NA;
    		temp_market_results_m[i].share_quality_local = share_quality_local;
    		temp_market_results_m[i].share_quality_green = share_quality_green;
    		temp_market_results_m[i].share_quality_green_local = share_quality_green_local;
    	}
    	return temp_market_results_m;
	}

	//calculates the uniform and discriminative pricing of market_result given in input
	function calc_market_clearing_prices(Lb.LemLib.market_result[] memory temp_market_results_m, bool uniform_pricing, bool discriminative_pricing) public pure returns(Lb.LemLib.market_result[] memory) {
	    //check whether merged bids and offers are empty
	    if (temp_market_results_m.length > 0) {
	        if(uniform_pricing) {
	            //Calculate market clearing price by taking average of last matching bids
	            //In Solidity, division rounds towards zero
	            uint price_cleared_uniform = (temp_market_results_m[temp_market_results_m.length - 1].price_energy_offer + temp_market_results_m[temp_market_results_m.length - 1].price_energy_bid)/2;
	            for(uint i = 0; i < temp_market_results_m.length; i++) {
	                temp_market_results_m[i].price_energy_market_uniform = price_cleared_uniform;
	            }
	        }
	        if(discriminative_pricing) {
	            //In Solidity, division rounds towards zero
	            for(uint i = 0; i < temp_market_results_m.length; i++) {
	                temp_market_results_m[i].price_energy_market_discriminative = (temp_market_results_m[i].price_energy_offer + temp_market_results_m[i].price_energy_bid)/2;
	            }
	        }
	    }
	    return temp_market_results_m;
	}

	//update the balances of the user infos in the storage, given the total market results on storage as well. It returns the user infos with the updated balances
	function update_balances_call() public view returns(Lb.LemLib.user[] memory) {
		Lb.LemLib.user[] memory temp_balance_update = lib.copy_array_user_info(ClearingExAnte.user_infos, 0, ClearingExAnte.user_infos.length - 1);
		for(uint i = 0; i < ClearingExAnte.market_results_total.length; i++) {
			int delta = int(ClearingExAnte.market_results_total[i].price_energy_market_uniform * ClearingExAnte.market_results_total[i].qty_energy_traded);//I don't divide by 1000, since there is no float
			for(uint j = 0; j < temp_balance_update.length; j++) {
				if(delta < 0) delta = (-1) * delta;
				string memory id_user = temp_balance_update[j].id_user;
				if(lib.compare_strings(ClearingExAnte.market_results_total[i].id_user_offer, id_user) || lib.compare_strings(ClearingExAnte.market_results_total[i].id_user_bid, id_user)) {
					if(!(lib.compare_strings(ClearingExAnte.market_results_total[i].id_user_offer,ClearingExAnte.market_results_total[i].id_user_bid))) {
						if(lib.compare_strings(ClearingExAnte.market_results_total[i].id_user_bid, id_user)) {
							delta = (-1) * delta;
						}
						temp_balance_update[j].balance_account = temp_balance_update[j].balance_account + delta;
					}
					temp_balance_update[j].t_update_balance = ClearingExAnte.market_results_total[i].t_cleared;
				}
			}
		}
		return temp_balance_update;
	}

	//update the balances of the user infos in the storage, given the total market results on storage as well. It writes the results on storage
	function update_balances_after_clearing_ex_ante() public {
		for(uint i = 0; i < ClearingExAnte.market_results_total.length; i++) {
			int delta = int(ClearingExAnte.market_results_total[i].price_energy_market_uniform * ClearingExAnte.market_results_total[i].qty_energy_traded);//I don't divide by 1000, since there is no float
			for(uint j = 0; j < ClearingExAnte.user_infos.length; j++) {
				if(delta < 0) delta = (-1) * delta;
				string memory id_user = ClearingExAnte.user_infos[j].id_user;
				if(lib.compare_strings(map_to_user_id[ClearingExAnte.market_results_total[i].id_user_offer], id_user) || lib.compare_strings(map_to_user_id[ClearingExAnte.market_results_total[i].id_user_bid], id_user)) {
					if(!(lib.compare_strings(map_to_user_id[ClearingExAnte.market_results_total[i].id_user_offer], map_to_user_id[ClearingExAnte.market_results_total[i].id_user_bid]))) {
						if(lib.compare_strings(map_to_user_id[ClearingExAnte.market_results_total[i].id_user_bid], id_user)) {
							delta = (-1) * delta;
						}
						ClearingExAnte.user_infos[j].balance_account = ClearingExAnte.user_infos[j].balance_account + delta;
					}
					ClearingExAnte.user_infos[j].t_update_balance = ClearingExAnte.market_results_total[i].t_cleared;
				}
			}
		}
	}

	//similar to the functions above, but it takes the parameters as input to update, only for 1 single user
	function update_user_balances(Lb.LemLib.log_transaction memory transaction_log) public{
		for(uint i = 0; i < user_infos.length; i++){
			if(lib.compare_strings(user_infos[i].id_user, map_to_user_id[transaction_log.id_user])){
				user_infos[i].balance_account += transaction_log.delta_balance;
				user_infos[i].t_update_balance = transaction_log.t_update_balance;
				break;
			}
		}
	}

	//it performs the full market clearing. The results are then stored in the variable market_results_total
	function market_clearing(uint n_clearings, uint t_clearing_first, bool supplier_bids, bool uniform_pricing, bool discriminative_pricing, uint clearing_interval, uint t_clearing_start, bool shuffle, bool verbose, bool update_balances, bool simulation_test) public {
	    if(verbose) {
	    	string_to_log = lib.concatenate_strings("Market clearing started on the blockchain\nNumber of clearings: ",lib.uint_to_string(n_clearings));
	    	string_to_log = lib.concatenate_strings(string_to_log,"\n");//two statements to reduce stack usage
	    }
	    for (uint i = 0; i < n_clearings; i++) {
            //Continuous clearing time, incrementing by market period
        	uint t_clearing_current = t_clearing_first + clearing_interval * i;
        	if(verbose) {
        		string_to_log = lib.concatenate_strings(string_to_log, lib.concatenate_strings("Clearing number: ",lib.uint_to_string(i)));
        		string_to_log = lib.concatenate_strings(string_to_log, lib.concatenate_strings(lib.concatenate_strings(". t_clearing_current = ", lib.uint_to_string(t_clearing_current)),".\n"));
        	}
        	bool add_supplier_bids;
        	if (i == 0 && supplier_bids) add_supplier_bids = supplier_bids;
        	else add_supplier_bids = false;
        	//performing a single clearing
        	single_clearing(t_clearing_current, add_supplier_bids, uniform_pricing, discriminative_pricing, t_clearing_start, false, true, verbose, shuffle, simulation_test);
    	}
    	if(update_balances) update_balances_after_clearing_ex_ante();
    	if(verbose) {
    		string_to_log = lib.concatenate_strings(string_to_log, "Updated balances of users");
    		emit logString(string_to_log);
    	}
	}
}
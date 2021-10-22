pragma solidity >=0.5.0 <0.7.5;
pragma experimental ABIEncoderV2;

//basic contract to get parametric data used in the algorithms
contract Param {
    
    string id_retailer = 'retailer01';        //id of energy retailer
    uint qty_energy_retailer_bid = 100000;    //quantity retailer bids on lem
    uint qty_energy_retailer_offer = 100000;  //quantity retailer offers on lem
    uint price_retailer_offer = 80000;         //offer price in sigma/Wh
    uint price_retailer_bid = 20000;     //bid price in sigma/Wh
    uint premium_preference_quality = 0;

	function get_id_retailer() public view returns(string memory) {
	    return id_retailer;
	}
	function get_price_offer_retailer() public view returns(uint) {
	    return price_retailer_offer;
	}
	function get_price_bid_retailer() public view returns(uint) {
	    return price_retailer_bid;
	}
	function get_qty_offer_retailer() public view returns(uint) {
	    return qty_energy_retailer_offer;
	}
	function get_qty_bid_retailer() public view returns(uint) {
	    return qty_energy_retailer_bid;
	}
	function get_premium_preference_quality() public view returns(uint) {
	    return premium_preference_quality;
	}
}
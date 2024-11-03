import numpy as np

class Agent(object):
    def __init__(self, agent_number, params={}):
        self.this_agent_number = agent_number  # index for this agent
        self.int = 42
        self.alpha = 4.0
        self.beta = -5.0
        self.gamma = 1.5
        self.project_part = params['project_part'] #useful to be able to use same competition code for each project part
        self.remaining_inventory = params['inventory_limit']
        self.inventory_replenish = params['inventory_replenish']
        self.low_earning_cutoff = 20
        self.dump_price = -10
        self.last_est = 0

    def _process_last_sale(
            self, 
            last_sale,
            state,
            inventories,
            time_until_replenish
        ):
        did_customer_buy_from_me = last_sale[0] == self.this_agent_number
        if did_customer_buy_from_me:  # can increase prices
            if last_sale[1][self.this_agent_number] == self.last_est + self.dump_price:
                self.dump_price += 0.5
            else:
                self.int += 1
                self.alpha *= 1.01
                self.beta *= 0.995
                self.gamma *= 1.005
        else:  # should decrease prices
            if last_sale[1][self.this_agent_number] == self.last_est + self.dump_price:
                self.dump_price -= 4
            else:
                self.int -= 1
                self.alpha *= 0.99
                self.beta *= 1.005
                self.gamma *= 0.995
        

    # Given an observation which is #info for new buyer, information for last iteration, and current profit and remaining inventory levels from each time
    # Covariates of the current buyer
    # Data from last iteration (who purchased from, prices for each agent)
    # Profits for each agent
    # Inventory levels for each agent
    # Returns an action: a non-negative number, indicating price this agent is posting for this new customer.

    def action(self, obs):
        new_buyer_covariates, last_sale, state, inventories, time_until_replenish = obs
        self.remaining_inventory = inventories[self.this_agent_number]
        self._process_last_sale(last_sale, state, inventories, time_until_replenish)

        if np.random.rand() < .005: #reset coef with some probability
            self.int = 42
            self.alpha = 4.0
            self.beta = -5.0
            self.gamma = 1.5

        est = np.dot(new_buyer_covariates, [self.alpha, self.beta, self.gamma]) + self.int
        self.last_est = est
        if self.remaining_inventory >= time_until_replenish:
            return est + self.dump_price
        else:
            if est < self.low_earning_cutoff:
                return 1000
            else:
                return est

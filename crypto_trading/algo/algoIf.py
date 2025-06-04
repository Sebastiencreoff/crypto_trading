#!/usr/bin/env python


class AlgoIf:
    """Super class to manage data."""

    def process(self, data_value, currency):
        """Process data, it returned 1 to buy and -1 to sell."""
        pass

    def update_config(self, config_section: dict):
        # Updates the algorithm's parameters from the given config section.
        # This method should be implemented by concrete algorithm classes.
        raise NotImplementedError

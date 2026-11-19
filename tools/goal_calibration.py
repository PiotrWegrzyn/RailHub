"""Measures what a stand-in player's company is worth each decade from the start of play to
END_YEAR, on one map and seed, to set the difficulty targets (railroads.sim.company.TARGETS).

    uv run python tools/goal_calibration.py Poland 1
"""

import sys
import time

import numpy as np

from railroads.newgame import MapChoice, Settings, make_game
from railroads.sim.autoplay import StandInPlayer
from railroads.sim.clock import END_YEAR
from railroads.sim.company import company_value
from railroads.sim.finance import dollars


class FrugalStandIn(StandInPlayer):
    """A stand-in player that buys a building only with the cash for it."""

    def _buy(self) -> None:
        growth = self.growth
        served = self._served()
        finances = self.railway.finances
        offers = []
        for kind in growth.rule.points:
            price = growth.owners.price_to_buy(kind)
            if not price:
                continue
            left = growth.count(kind) - growth.owners.owned(kind)
            for row, col in np.argwhere(left > 0).tolist():
                if (row, col) in growth.station_trade.in_reach and served[row, col] > 0:
                    worth = self._monthly_worth(kind) * served[row, col]
                    offers.append((worth / price, (row, col), kind, price))
        offers.sort(key=lambda offer: -offer[0])
        bought = 0
        for _, cell, kind, price in offers:
            if bought == self.rule.buys_per_year:
                break
            if finances.pay(price, "buildings bought"):
                growth.owners.own(cell, kind)
                bought += 1


def main(map_title: str, seed: int, frugal: bool = False) -> None:
    began = time.monotonic()
    saved = make_game(Settings(seed=seed, map=MapChoice(map_title)), report=lambda text: None)
    game = saved.game
    kind = FrugalStandIn if frugal else StandInPlayer
    player = kind(game.growth, game.railway, np.random.default_rng(seed + 2))
    game.stand_in = player
    print(f"{map_title} seed {seed}: set up in {time.monotonic() - began:.0f} s", flush=True)
    while game.growth.year < END_YEAR:
        player.month(np.random.default_rng(seed + player.months))
        if player.months % 120 == 0:
            value = company_value(game.railway)
            parts = ", ".join(f"{name} {dollars(amount)}" for name, amount in value.parts())
            print(
                f"{game.growth.year}: worth {dollars(value.total)} ({parts}), "
                f"{len(game.railway.stations())} stations, "
                f"{time.monotonic() - began:.0f} s",
                flush=True,
            )


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]), "frugal" in sys.argv[3:])

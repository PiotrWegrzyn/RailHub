"""A train calling at a station (railroads.sim.railway): it unloads every car, freight into the
station and passengers and mail paid for there, swaps them for the stop's cars and fills those, and
waits up to WAIT_MONTHS for them to fill if the stop says so. Freight from this station goes back
in its stockpile for nothing; the rest waits as goods brought, to be handed out."""

from typing import TYPE_CHECKING

from railroads.sim.constructions import Station
from railroads.sim.goods import TRAVELLERS, Good
from railroads.sim.grids import Cell
from railroads.sim.station_goods import StationGoods
from railroads.sim.tech import Tree
from railroads.sim.trains import Car, Parcels, Stop, Train

WAIT_MONTHS = 1.0

if TYPE_CHECKING:
    from railroads.sim.railway import Railway


class StationCalls:
    def __init__(self, railway: Railway) -> None:
        self._railway = railway

    def arrive(self, train: Train, stop: Stop) -> None:
        """Calls at a stop: unloads every car, takes on the stop's cars and fills them."""
        train.called = True
        if not any(isinstance(b, Station) for b in self._railway.constructions.at(stop.cell)):
            train.status = "Its stop has no station"
            return
        goods = self._railway.station_goods(stop.cell)
        if train.loaded_at == stop.cell:
            train.cars = self._swap_cars(train.cars, stop.cars, goods, stop.cell)
        else:
            demand = self._railway.station_demand(stop.cell)
            for car in train.cars:
                if car.load <= 0:
                    continue
                if car.good in TRAVELLERS:
                    earned = demand.deliver(car.good, car.load) * self._railway.price_factor(
                        car.good
                    )
                    self._railway.finances.earn(earned, car.good.value)
                    self._railway.research.earn(Tree.PASSENGERS, earned)
                    car.empty(car.load)
                    self._railway.history.deliver(self._railway.year, car.good)
                    continue
                unloaded = min(car.load, goods.room)
                if unloaded > 0:
                    self._store(goods, stop.cell, car.good, car.empty(unloaded))
                    self._railway.history.deliver(self._railway.year, car.good)
            train.cars = self._next_cars(train.cars, stop.cars)
        train.loaded_at = stop.cell
        self.load(train, stop)
        if stop.wait_for_load and train.cars and not train.full:
            train.waiting = WAIT_MONTHS

    def _next_cars(self, cars: list[Car], wanted: list[Good]) -> list[Car]:
        """The cars for the next leg: each keeps what the fullest car of its good still holds,
        freight that did not fit into the station; anything else left aboard is lost."""
        left = sorted(cars, key=lambda car: -car.load)
        kept = []
        for good in wanted:
            car = next((car for car in left if car.good == good), None)
            if car is None:
                kept.append(Car(good, capacity=self._railway.car_capacity(good)))
            else:
                left.remove(car)
                kept.append(car)
        return kept

    def _swap_cars(
        self, cars: list[Car], wanted: list[Good], goods: StationGoods, station: Cell
    ) -> list[Car]:
        """Swaps a train's cars for the wanted ones at the station that filled them: each wanted
        car keeps the fullest car of its good, and what dropped cars held goes back into the
        stockpile."""
        left = sorted(cars, key=lambda car: -car.load)
        kept = []
        for good in wanted:
            car = next((car for car in left if car.good == good), None)
            if car is None:
                kept.append(Car(good, capacity=self._railway.car_capacity(good)))
            else:
                left.remove(car)
                kept.append(car)
        for car in left:
            if car.load > 0:
                self._store(goods, station, car.good, car.empty(car.load))
        return kept

    @staticmethod
    def _store(goods: StationGoods, station: Cell, good: Good, parcels: Parcels) -> None:
        """Puts unloaded cargo in the station: what came from here goes back in its stockpile
        for nothing, the rest waits as goods brought."""
        home = parcels.pop(station, 0.0)
        if home > 0:
            goods.stockpile[good] = goods.stockpile.get(good, 0.0) + home
        if parcels:
            goods.bring(good, parcels)

    def load(self, train: Train, stop: Stop) -> None:
        """Tops each car up from the stop's stockpile, then with goods trains brought that the
        buildings around it have no use for this month."""
        goods = self._railway.station_goods(stop.cell)
        for car in train.cars:
            take = min(car.capacity - car.load, goods.stockpile.get(car.good, 0.0))
            if take > 0:
                car.fill({stop.cell: take})
                goods.stockpile[car.good] -= take
            spare = goods.brought.get(car.good, 0.0)
            spare -= self._railway.station_demand(stop.cell).monthly_want(car.good)
            transfer = min(car.capacity - car.load, spare)
            if transfer > 0:
                car.fill(goods.release(car.good, transfer))

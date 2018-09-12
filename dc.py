from dataclasses import dataclass

@dataclass
class InventoryItem:
    name: str
    unit_price: float
    quantity_on_hand: int = 0

    def total_cost(self) -> float:
        return self.unit_price * self.quantity_on_hand


if __name__ == '__main__':
    inv = InventoryItem('Oi', 10, 2)
    inv2 = InventoryItem('Oi', 10, 2)
    print(inv == inv2)
    print(
        f'Name: {inv.name}\n'
        f'Price: {inv.unit_price}\n'
        f'Qtt: {inv.quantity_on_hand}\n'
        f'Cost: {inv.total_cost()}'
    )

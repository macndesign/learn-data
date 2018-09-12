from dataclasses import dataclass, field, fields
from typing import Any, List
from math import asin, cos, radians, sin, sqrt
from cryptography.fernet import Fernet


@dataclass
class Message:
    desc: bytes

    def crip(self):
        key = Fernet.generate_key()
        f = Fernet(key)
        return f.encrypt(self.desc), key
    
    def decrip(self, token, key):
        f = Fernet(key)
        return f.decrypt(token)


@dataclass(frozen=True)
class Position:
    name: str
    lon: float = field(default=0.0, metadata={'unit': 'degrees'})
    lat: float = field(default=0.0, metadata={'unit': 'degrees'})

    def distance_to(self, other):
        r = 6371  # Earth radius in kilometers
        lam_1, lam_2 = radians(self.lon), radians(other.lon)
        phi_1, phi_2 = radians(self.lat), radians(other.lat)
        h = (sin((phi_2 - phi_1) / 2)**2
             + cos(phi_1) * cos(phi_2) * sin((lam_2 - lam_1) / 2)**2)
        return 2 * r * asin(sqrt(h))


@dataclass
class WithoutExplicitTypes:
    name: Any
    value: Any = 42


# Cards
RANKS = '2 3 4 5 6 7 8 9 10 J Q K A'.split()
SUITS = '♣ ♢ ♡ ♠'.split()

def make_french_deck():
    return [PlayingCard(r, s) for s in SUITS for r in RANKS]


@dataclass(order=True)
class PlayingCard:
    sort_index: int = field(init=False, repr=False)
    rank: str
    suit: str

    def __post_init__(self):
        self.sort_index = (RANKS.index(self.rank) * len(SUITS)
                           + SUITS.index(self.suit))

    def __str__(self):
        return f'{self.suit}{self.rank}'


@dataclass
class Deck:
    cards: List[PlayingCard] = field(default_factory=make_french_deck)

    def __repr__(self):
        cards = ', '.join(f'{c!s}' for c in self.cards)
        return f'{self.__class__.__name__}({cards})'


if __name__ == '__main__':
    print(fields(Position))
    lat_unit = fields(Position)[2].metadata['unit']
    print(lat_unit)
    p1 = Position('Oslo', 10.8, 59.9)
    print(f'{p1.name} is at {p1.lat}°N, {p1.lon}°E')
    p2 = Position('Null Island')
    print(f'{p2.name} is at {p2.lat}°N, {p2.lon}°E')
    p3 = Position('Greenwich', lat=51.8)
    print(f'{p3.name} is at {p3.lat}°N, {p3.lon}°E')
    p4 = Position('Vancouver', -123.1, 49.3)
    print(f'{p4.name} is at {p4.lat}°N, {p4.lon}°E')
    print(f'Oslo distance to Vancouver {p1.distance_to(p4)}')
    we = WithoutExplicitTypes('Oi')
    print(f'{we.name} - {we.value}')
    # Cards
    cards = Deck()
    print(cards)
    queen_of_hearts = PlayingCard('Q', '♡')
    ace_of_spades = PlayingCard('A', '♠')
    print('ace_of_spades > queen_of_hearts')
    print(ace_of_spades > queen_of_hearts)
    print(Deck(sorted(make_french_deck())))
    from random import sample
    print(Deck(sample(make_french_deck(), k=10)))
    # msg
    msg = Message(b'test enc/dec')
    print(msg)
    cr = msg.crip()
    print(cr)
    print(msg.decrip(*cr))

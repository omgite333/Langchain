from typing import TypeDict

class Person(TypeDict):
    name: str
    age: int
    email: str 

new_person: Person = {'name':''}

from pydantic import BaseModel , EmailStr , Field
from typing import Optional 

class Student(BaseModel):
    name: str
    age: Optional[int] = None
    email: EmailStr
    cgpa: float = Field(gt=0,lt =5)

new_student = {'name':'ntin'}

student = Student(**new_student)

print(student)
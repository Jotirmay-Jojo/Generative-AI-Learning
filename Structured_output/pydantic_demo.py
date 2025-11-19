from pydantic import BaseModel,EmailStr,Field
from typing import Optional

class Student(BaseModel):
    name:str = 'Joy'
    age:Optional[int] = None
    email:EmailStr
    cgpa: float =Field(gt=0,lt=10,description="Cgpa is the score of your achedemics", default=5)
    
new_student={'email':'joy@gmail.com','age':'22'} #type coercing(Implicit conversion by pydantic)

student=Student(**new_student)
print(student.model_dump_json)

stud_dict=dict(student)

print(stud_dict)
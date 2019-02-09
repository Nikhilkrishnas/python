class Employee():
    empCount = 0 #initialization of the data member
    empSal = [];
    empName=[];
    def __init__(self,name,family,salary,department): #default constructor
        self.empname = name
        self.empfamily = family
        self.empsalary = salary
        self.empdepartment = department
        Employee.empCount +=1  #counts the number of employees
        Employee.empSal.append(salary)  #appends salaray attribute
        Employee.empName.append(name)

    def avg_salary(self):
        print('the average salary is')
        sumSal = 0;
        for sal in Employee.empSal:
            sumSal = sumSal+ int(sal);
        return sumSal/len(Employee.empName)

class FulltimeEmployee(Employee):
    def __init__(self,name,family,salary,department):
        Employee.__init__(self,name,family,salary,department)


emp1 = FulltimeEmployee('apoorva','geetanjali','7000','CS');
emp2 = FulltimeEmployee('tejaswi','ayadapu','5000','CS');
emp3=FulltimeEmployee('BINDU','HIMA','7000','EE')
print(FulltimeEmployee.empCount)
print(FulltimeEmployee.empSal)
avgSal = FulltimeEmployee.avg_salary(Employee);
print(avgSal)


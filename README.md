# Customers' Purchasing Likelihood Prediction using Machine Learning 
## Introduction 
In 1998, the Adventure Works Cycles company collected a large volume of data about their existing customers, including demographic features and information about purchases they have made. The company is particularly interested in analyzing customer data to determine any apparent relationships between demographic features known about the customers and the likelihood of a customer purchasing a bike. Additionally, the analysis should endeavor to determine whether a customer's average monthly spend with the company can be predicted from known customer characteristics.

**Objectives** :
- Building a classification model **predicting** likelihood of a customer **to purchase a bike**
- Building a regression model **predicting** customer's **monthly spending** with the company
## Files description
### Data
The data consists of three files, containing data that was collected on January 1st 

#### AdvWorksCusts.csv
This files contains the Customer demographic data consisting of the following fields:
1. **CustomerID** (integer): A unique customer identifier.
Title (string): The customer's formal title (Mr, Mrs, Ms, Miss Dr, etc.)
2. **FirstName** (string): The customer's first name.
3. **MiddleName** (string): The customer's middle name.
4. **LastName** (string): The customer's last name.
5. **Suffix** (string): A suffix for the customer name (Jr, Sr, etc.)
6. **AddressLine1** (string): The first line of the customer's home address.
7. **AddressLine2** (string): The second line of the customer's home address.
8. **City** (string): The city where the customer lives.
9. **StateProvince** (string): The state or province where the customer lives.
10. **CountryRegion** (string): The country or region where the customer lives.
11. **PostalCode** (string): The postal code for the customer's address.
12. **PhoneNumber** (string): The customer's telephone number.
13. **BirthDate** (date): The customer's date of birth in the format YYYY-MM-DD.
14. **Education** (string): The maximum level of education achieved by the customer:
- *Partial High School*
- *High School*
- *Partial College*
- *Bachelors*
- *Graduate Degree*
15. **Occupation** (string): The type of job in which the customer is employed:
- *Manual*
- *Skilled Manual*
- *Clerical*
- *Management*
- *Professional*
16. **Gender** (string): The customer's gender (for example, M for male, F for female, etc.)
17. **MaritalStatus** (string): Whether the customer is married (M) or single (S).
18. **HomeOwnerFlag** (integer): A Boolean flag indicating whether the customer owns their own home (1) or not (0).
19. **NumberCarsOwned** (integer): The number of cars owned by the customer.
20. **NumberChildrenAtHome** (integer): The number of children the customer has who live at home.
21. **TotalChildren** (integer): The total number of children the customer has.
22. **YearlyIncome** (decimal): The annual income of the customer.

#### AW_AveMonthSpend.csv
This file contains the sales data for existing customers, consisting of the following fields:
1. **CustomerID** (integer): The unique identifier for the customer.
2. **AveMonthSpend** (decimal): The amount of money the customer spends with Adventure Works Cycles on average each month.

#### AW_BikeBuyer.csv
This file contains the sales data for existing customers, consisting of the following fields:
1. **CustomerID** (integer): The unique identifier for the customer.
2. **BikeBuyer** (integer): A Boolean flag indicating whether a customer has previously purchased a bike (1) or not (0).
#### AW_Test.csv
This file is similar to AdvWorksCuts.csv consist of data on which our predictive model will be tested
### Source code
#### Train_code.ipynb
This file contains the source code for analyzing customer's data, building predictive  models(regression and classification model) and evaluating accuracy of that models
#### Test_code.ipynb
This file contains the source code for predicting customer's bike purchasing likelihood and monthly spending with the company on testing data(AW_Test.csv) and storing it in a file(Result.csv)
### Models
#### class_model.sav
This file contains the newly build classification model for predicting bike purchasing likelihood of customers after executing the Train_code
#### reg_model.sav
This file contains the newly build regression model for predicting monthly spending of customers after executing the Train_code
### Result
#### Result.csv
This file contains the customers data along with the prediction on testing data(AW_Test.csv).Note that last two columns are predicted value in this file. 
## System Requirements
- **Language** : Python 3.7
- **Tools used** : Pandas, Numpy, Scikit learn, Seaborne, Matplotlib
- **Platform used** : Jupyter notebook
## Acknowledgement 
This project is the final exam of the course [Principles of Machine Learning: Python Edition](https://www.edx.org/course/principles-of-machine-learning-python-edition-4) and the datasets are downloaded from [here](https://github.com/MicrosoftLearning/Principles-of-Machine-Learning-Python/tree/master/Final%20Exam)

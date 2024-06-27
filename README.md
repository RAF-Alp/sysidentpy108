# Report for Assignment 1

## Project chosen

**Name:** Sysidentpy

**URL:** (https://github.com/wilsonrljr/sysidentpy)

**Project License:** BSD 3-Clause License

**Number of lines of code and the tool used to count it:** 16039

**Programming language:** Python

## Coverage measurement

### Existing tool

**Coverage Tool:** coverage.py

**How it was executed:**
 python -m coverage run -m sysidentpy.neural_network.tests.test_narxnn
 python -m coverage report -m

<Show the coverage results provided by the existing tool with a screenshot>
**Coverage Results (6%):**
 
![Screenshot from 2024-06-27 18-25-36](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/b3099ab2-f613-4a59-90f3-b49dcebb1e2c)


### Your own coverage tool
Set up a FLAG Dictionary, print_coverage() made to print FLAGs after tests have run:

![Screenshot from 2024-06-27 18-31-22](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/b1156fde-0a6e-411c-8b3f-277760d27b6f)

In Function Example (Set up and in first two branches):

![Screenshot from 2024-06-27 18-32-24](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/bb4738d2-186f-4ebb-a851-f3d7e8d2682d)

## Functions Tested

![Screenshot from 2024-06-27 19-20-49](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/23662f63-81c4-4f95-922d-d996c6127011)

## Mike Voeten
Functions:

_basis_function_n_steps_horizon: BFNSP: 8 branches

_basis_function_n_step_prediction: BFNSH: 5 branches

**Own Tool:**

https://github.com/RAF-Alp/sysidentpy108/commit/eb36a01962883f2b8eadbf4cbd0b92f5a6fe836d

![Screenshot from 2024-06-27 13-18-39](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/328a75f6-7520-400b-8efc-9adc75489046)



## Song
1. Function1: _model_prediction with 4 branches
A. Own Coverage measurement in a forked Repository

**Own Tool:**
https://github.com/wilsonrljr/sysidentpy/commit/f5f9fe35616f776d13df541e4d8cb59aba6627f1#diff-a65f03450fbba5edc554aa198ce7b72aa6c492226ff86455d1e39547761fa9e3
B. output

![Screenshot from 2024-06-27 18-35-45](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/96089d1f-971e-4074-af16-5ee08166d7b4)

(The coverage Report is generated by an HTML page to increase the readability of the user interface with the tool.)

2. Function2: narmax_predict with 6 branches
A. Own Coverage measurement in a forked Repository

**Own Tool:**
https://github.com/wilsonrljr/sysidentpy/commit/f5f9fe35616f776d13df541e4d8cb59aba6627f1#diff-a65f03450fbba5edc554aa198ce7b72aa6c492226ff86455d1e39547761fa9e3
B. output
It is shown above.

## Alpdeniz Sarici Hernandez

split_data

<Show a patch (diff) or a link to a commit made in your forked repository that shows the instrumented code to gather coverage measurements>

**Own Tool:**
https://github.com/RAF-Alp/sysidentpy108/commit/5f8533aca723990991b6c009238ce7f1f1f3400e 
Here is the print coverage function and the dictionary of flags used to save the branches being covered.

![Screenshot from 2024-06-27 18-36-30](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/50540d3c-559b-44de-88fe-dc07c5e3f375)

fit
<Provide the same kind of information provided for Function 1>
https://github.com/RAF-Alp/sysidentpy108/commit/5f8533aca723990991b6c009238ce7f1f1f3400e 
Here is the print coverage function and the dictionary of flags used to save the branches being covered.

![Screenshot from 2024-06-27 18-37-09](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/d5bfdbe1-f882-44ba-85d1-78a269ff1470)

## Luc Lacombe
fit
https://github.com/wilsonrljr/sysidentpy/commit/00916ed4e4331723b1efd6bec51eae6902da9e6b 

![Screenshot from 2024-06-27 18-38-20](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/9e5beb7b-3af8-441e-a101-0d5c499b7a72)

build_system_data

https://github.com/wilsonrljr/sysidentpy/commit/11904f42f6f44d271ebed752e40762a48df3eb34 

![Screenshot from 2024-06-27 18-39-50](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/e992c32e-7140-433b-9dc0-5abfe344dc53)

## Coverage improvement

### Individual tests

## Mike

<Test 1>

<Show a patch (diff) or a link to a commit made in your forked repository that shows the new/enhanced test>

<Provide a screenshot of the old coverage results (the same as you already showed above)>

![Screenshot from 2024-06-27 18-42-08](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/ccbd74bb-98c9-4770-8bec-6e7f6181dfb8)

The coverage has improved because every model type and empty input is tested.
<Provide a screenshot of the new coverage results>

## Song

# 1. Test for 'model_prediction’

![Screenshot from 2024-06-27 18-45-17](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/8e674490-a259-4d08-a73a-dc97c7cdbf80)

In the function, total of 4 branches are identified including 2 invisible else.

![Screenshot from 2024-06-27 18-45-58](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/fb1de5c8-62fa-4cc1-af79-7e6952545a76)

In the original test tool, only 1 branch has been reached, so three more tests are added to test three other conditions. 
+ 'test_additional_test_1_with_NFIR()','test_additional_test_2_with_NAR()','test_additional_test_5_with_Unknown()' 

![Screenshot from 2024-06-27 18-47-00](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/d035f1c7-5d8f-443d-a279-3d6749fbfed0)

As a result, it can fully cover all the branches.

# 2. Test for '_narmax_predict'

![Screenshot from 2024-06-27 18-49-29](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/37d5af54-114a-4680-8e38-78820286e226)
![Screenshot from 2024-06-27 18-49-59](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/a2b78ff2-b54d-46ec-acef-8e19a29baaa6)

In the function, a total of 6 branches are identified including 2 invisible else's.
In the original test tool, only 3 branches had been reached, so two more tests are added to test two other conditions. 

'test_additional_test_3_with_Insufficient_initial_conditions()',
'test_additional_test_4_with_X_none()'

**As a result, it can fully cover all the branches.**

![Screenshot from 2024-06-27 18-52-10](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/fd1dd177-3d42-4bbe-bf81-920d2a9684f8)


**Before coverage by lines:**

![Screenshot from 2024-06-27 18-53-12](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/8bed6b47-1a7c-47ac-b20c-a0c57a647c02)

**After coverage by lines:**

![Screenshot from 2024-06-27 18-53-30](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/b5968052-417a-4180-9766-9be603e1720d)

**Branch coverage before:**

![Screenshot from 2024-06-27 18-54-45](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/35ab5c26-9ba1-423c-adad-1a68a5f2550b)

**Branch coverage after:**

![Screenshot from 2024-06-27 18-55-13](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/01e77d25-be6a-46cd-9ba6-dfcda2d4c7b2)


## Alpdeniz Sarici Hernandez
test_fit_raise_y():
https://github.com/RAF-Alp/sysidentpy108/commit/7c8c931992d506e0c7bfa6aa8b02298b25d53cb1 

![Screenshot from 2024-06-27 18-56-08](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/2b3afa59-623b-4c6d-8766-597e7c20c949)

![Screenshot from 2024-06-27 18-56-36](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/991382aa-5677-483d-86cc-fa2c98ddb5e7)

Coverage is improved by 1 because I added an assert to cover X=None branch.  Test is modified.

test_fit_lag_nar()
https://github.com/RAF-Alp/sysidentpy108/commit/367f8081382b075056cfb4026ab7e12a49419455 

![Screenshot from 2024-06-27 18-57-06](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/db552b03-1927-4c34-b7d6-457021df3064)

![Screenshot from 2024-06-27 18-57-52](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/6315b51e-3901-4222-addc-8db49415e142)


Coverage is improved by 2 because it tests for verbose being true instead of false in the model. Test is modified

test_split_data_non_polynomial_no_ensemble()
https://github.com/RAF-Alp/sysidentpy108/commit/301319a0243b426de88748ac8fe475b41ad33588 

![Screenshot from 2024-06-27 18-58-19](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/c5d6a8cd-58d2-45c9-9a2f-5eeb7b9d7076)

![Screenshot from 2024-06-27 18-58-44](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/6dad21a5-b0ec-4fb5-a75a-8fb0e19ee6d0)

Improved the coverage by 2 because it tests split_data with a custom basis function where the ensemble is False and not Polynomial.

test_split_data_y_none()
https://github.com/RAF-Alp/sysidentpy108/commit/7c8c931992d506e0c7bfa6aa8b02298b25d53cb1 

![Screenshot from 2024-06-27 18-59-23](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/a099857e-7eda-4034-aa02-e16bbe78f3ec)

![Screenshot from 2024-06-27 18-59-50](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/5cbc0592-4035-42d4-a93e-1015c95df8a0)

New test function added improves branch coverage by 2 because it checks for the model without the Y parameter. That accesses one branch and then it checks for X being of polynomial type.


## Luc Lacombe

test_fit_degree_and_ensemble_branches() 

https://github.com/wilsonrljr/sysidentpy/commit/73c737bbbaa00bc9b27b0c8d1317fe7e1ddda114 

![Screenshot from 2024-06-27 19-02-13](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/3027baa7-968f-47f7-9b98-7e6479a864c4)

![Screenshot from 2024-06-27 19-02-32](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/fcb117dd-2f75-4998-bcfd-124a3b6e3f9a)

The coverage was improved from 5/7 branches reached (71.43%) to 7/7 branches (100%).
The first branch un-covered branch could only be reached if the fourier object of the fit method had a degree greater than one. The second un-covered branch could only be reached if the fourier object's boolean attribute ‘ensemble’ was True.
To reach these two branches and improve the coverage to 7/7 the “test_fit_degree_and_ensemble_branches” test was created which instantiates a fourier object with degree = 2 and ensemble = True and then the fit function is called with the object which enters both branches with the same test.
Finally three asserts are made to ensure the degree and ensemble were correctly set and that the final output of the fit method isnt None ensuring the function worked correctly with the given parameters.

test_build_system_data_static_gain_False_branch()
test_build_system_data_static_function_False_branch()

https://github.com/wilsonrljr/sysidentpy/commit/c2109493d7d025611f814f6f2fb0fc5771f07c7b 

![Screenshot from 2024-06-27 19-02-49](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/ab84d09b-69c9-4b97-9cb2-a0b372f9be76)

![Screenshot from 2024-06-27 19-03-04](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/c726bf99-b750-438d-865d-edbe37807003)

The coverage was improved from 3/5 branches reached (60%) to 5/5 branches (100%).
The first branch un-covered branch could only be reached if the AILS object’s ‘static_gain’ boolean attribute was False. The second un-covered branch could only be reached if the AILS object's boolean attribute ‘static_function’ was False.
To reach these two branches and improve the coverage to 7/7 the two test functions were created. The first one (test_build_system_data_static_gain_False_branch()) instantiates an AILS object with ‘static_gain’ set to False and then the build_system_data function is called with the object which enters the first branch. When the branch is entered it returns [y] + [static_function], therefore, to ensure the test is correct we assert that [y] + [static_function] was returned.
The second function one (test_build_system_data_static_function_False_branch()) instantiates an AILS object with ‘static_function’ set to False and then the build_system_data function is called with the object which enters the second branch. When the branch is entered it returns [y] + [static_gain], therefore, to ensure the test is correct we assert that [y] + [static_gain] was returned.
Although both these test functions are similar, since both the un-covered branches both lead to return statements two separate functions had to be made.

### Overall

![Screenshot from 2024-06-27 19-03-21](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/ee41da83-7a05-4c06-b0d5-65467e6ae4ef)

![Screenshot from 2024-06-27 19-03-40](https://github.com/RAF-Alp/sysidentpy108/assets/118909280/8e419a20-7fb5-4ac8-9b41-f66aa2efd6bd)

## Statement of individual contributions

# Mike Voeten:
I found several open-source Github repositories, counted the lines in them and put them together in a table for discussion. I created unit tests for two functions and a simple coverage tool. 


# Song:
I improved the print coverage function with HTML code to increase the readability of the test result. My contribution is that I set up the discussion on the meeting plan and the decision on the project schedules. Also, working on two functions to improve the coverage and sharing my approach with colleagues, which might be applicable to other methods. 


# Alpdeniz Sarici Hernandez:
I made the print coverage function to incorporate the dictionary of branches. With this, we could all output the branches that needed testing in our chosen functions. Once we knew which branches needed to be covered we created tests using arranges, acts, and asserts. Once I reached 80% I created my pull request and merged it with our code. Additionally, I was responsible for helping organize meetings and also clear any doubts we had.

# Luc Lacombe:
Besides working on my two functions and improving their coverage, my main contributions to the group were communication and help. I organized many of the online and in person meetings we had and I helped my groupmates with bugs and errors they were running into. For example I helped Alpdeniz fix an issue where many of his tests were failing due to the newest version of numpy not supporting int32 float type variables.


# Report for Assignment 2, Group 108

Programming language used: Python

## Workflow 1: Testing (`testing.yml`)

<Inform which tool is used to compile and test.>

<Provide the link to one log (from the "Actions" tab on GitHub) of an execution of this workflow>

## Workflow 2: Static analysis (`static_analysis_1.yml`)

<Inform which tool is used to perform code quality check with static analysis.>

<Provide the link to one log (from the "Actions" tab on GitHub) of an execution of this workflow>

## Workflow 3: Static analysis (`static_analysis_2.yml`)

Tool: pylint
Link: https://github.com/RAF-Alp/G108A2/actions/runs/9697490397 

## Workflow 4: Release (`release.yml`)

https://github.com/RAF-Alp/G108A2/actions/runs/9698887614/job/26766524998 

## Statement of individual contributions

<Write what each group member did. Use the following table for that and add additional text under it if you see fit.>

| Member          | Created workflows | Reviewed workflows | Merged pull requests' number | Issues Closed |
|-----------------|-------------------|--------------------|------------------------------|---------------|
| Mike            | 1                 | 1                  | 0                            | 1             |
| Song            | 8                 | 2                  | 1                            | 1             |
| Alpdeniz Sarici | 1                 | 4                  | 4                            | 1             |
| Luc Lacombe     | 1                 | 4                  | 0                            | 1             |




import graphlab 
import pandas as pd
import numpy as np

tmp = graphlab.SArray([1.,2.,3.])
tmp_cubed = tmp.apply(lambda x:x**3)
print tmp
print tmp_cubed

ex_sframe = graphlab.SFrame()
ex_sframe['power_1'] = tmp
print ex_sframe

def polynomial_sframe(feature,degree):
    poly_sframe = graphlab.SFrame()
    poly_sframe['power_1'] = feature
    if degree > 1:
        # range usually starts at 0 and stops at the endpoint-1. We want it to start at 2 and stop at degree
        for pow in range(2,degree+1): 
            name = 'power_' + str(pow)
            poly_sframe[name] = feature.apply(lambda x:x**pow)
    return poly_sframe

print polynomial_sframe(tmp,3)

sales = graphlab.SFrame('C:\\Machine_Learning\\kc_house_data.gl\\')  
sales = sales.sort(['sqft_living','price'])
poly1_data = polynomial_sframe(sales['sqft_living'], 1)
poly1_data['price'] = sales['price'] # add price to the data since it's the target
model1 = graphlab.linear_regression.create(poly1_data, target = 'price', features = ['power_1'], validation_set = None)
#let's take a look at the weights before we plot
model1.get("coefficients")
import matplotlib.pyplot as plt
plt.plot(poly1_data['power_1'],poly1_data['price'],'.',
        poly1_data['power_1'], model1.predict(poly1_data),'-')
plt.show()

poly2_data = polynomial_sframe(sales['sqft_living'], 2)
my_features = poly2_data.column_names() # get the name of the features
poly2_data['price'] = sales['price'] # add price to the data since it's the target
model2 = graphlab.linear_regression.create(poly2_data, target = 'price', features = my_features, validation_set = None)
model2.get("coefficients")
plt.plot(poly2_data['power_1'],poly2_data['price'],'.',
        poly2_data['power_1'], model2.predict(poly2_data),'-')
plt.show()


poly3_data = polynomial_sframe(sales['sqft_living'], 3)
my_features = poly3_data.column_names() # get the name of the features
poly3_data['price'] = sales['price'] # add price to the data since it's the target
model3 = graphlab.linear_regression.create(poly3_data, target = 'price', features = my_features, validation_set = None)
model3.get("coefficients")
plt.plot(poly3_data['power_1'],poly3_data['price'],'.',
        poly3_data['power_1'], model3.predict(poly3_data),'-')
plt.show()

poly15_data = polynomial_sframe(sales['sqft_living'], 15)
my_features = poly15_data.column_names() # get the name of the features
poly15_data['price'] = sales['price'] # add price to the data since it's the target
model15 = graphlab.linear_regression.create(poly15_data, target = 'price', features = my_features, validation_set = None)
model15.get("coefficients")
plt.plot(poly15_data['power_1'],poly15_data['price'],'.',
        poly15_data['power_1'], model15.predict(poly15_data),'-')
plt.show()

tmp1,tmp2 = sales.random_split(0.5,seed=0)
set_1,set_2 = tmp1.random_split(0.5,seed=0)
set_3,set_4 = tmp2.random_split(0.5,seed=0)


fig, ax = plt.subplots(nrows=2,ncols=2)
plt.subplot(2,2,1)
poly15_set1 = polynomial_sframe(set_1['sqft_living'], 15)
my_features = poly15_set1.column_names() # get the name of the features
poly15_set1['price'] = set_1['price'] # add price to the data since it's the target
model15_set1 = graphlab.linear_regression.create(poly15_set1, target = 'price', features = my_features, validation_set = None)
model15_set1.get("coefficients")
plt.plot(poly15_set1['power_1'],poly15_set1['price'],'.',
        poly15_set1['power_1'], model15_set1.predict(poly15_set1),'-')
#plt.show()

plt.subplot(2,2,2)
poly15_set2 = polynomial_sframe(set_2['sqft_living'], 15)
my_features = poly15_set2.column_names() # get the name of the features
poly15_set2['price'] = set_2['price'] # add price to the data since it's the target
model15_set2 = graphlab.linear_regression.create(poly15_set2, target = 'price', features = my_features, validation_set = None)
model15_set2.get("coefficients")
plt.plot(poly15_set2['power_1'],poly15_set2['price'],'.',
        poly15_set2['power_1'], model15_set2.predict(poly15_set2),'-')
#plt.show()

plt.subplot(2,2,3)
poly15_set3 = polynomial_sframe(set_3['sqft_living'], 15)
my_features = poly15_set3.column_names() # get the name of the features
poly15_set3['price'] = set_3['price'] # add price to the data since it's the target
model15_set3 = graphlab.linear_regression.create(poly15_set3, target = 'price', features = my_features, validation_set = None)
model15_set3.get("coefficients")
plt.plot(poly15_set3['power_1'],poly15_set3['price'],'.',
        poly15_set3['power_1'], model15_set3.predict(poly15_set3),'-')
#plt.show()

plt.subplot(2,2,4)
poly15_set4 = polynomial_sframe(set_4['sqft_living'], 15)
my_features = poly15_set4.column_names() # get the name of the features
poly15_set4['price'] = set_4['price'] # add price to the data since it's the target
model15_set4 = graphlab.linear_regression.create(poly15_set4, target = 'price', features = my_features, validation_set = None)
model15_set4.get("coefficients")
plt.plot(poly15_set4['power_1'],poly15_set4['price'],'.',
        poly15_set4['power_1'], model15_set4.predict(poly15_set4),'-')
plt.show()

training_and_validation, testing = sales.random_split(0.9,seed=1)
training, validation = training_and_validation.random_split(0.5,seed=1)

#decide degree based on validation set, 6
for i in range(1,16):
    poly_data_i = polynomial_sframe(training['sqft_living'], i)
    my_features = poly_data_i.column_names() # get the name of the features
    poly_data_i['price'] = training['price'] # add price to the data since it's the target
    model = graphlab.linear_regression.create(poly_data_i, target = 'price', features = my_features, validation_set = None, verbose=False)
    #model.get("coefficients")
    poly_val_i = polynomial_sframe(validation['sqft_living'], i)
    poly_val_i['price'] = validation['price']
    pred = model.predict(poly_val_i)
    error = pred - validation['price']
    error = error * error
    print "i = ",i," ",error.sum()

#test data
poly_train_6 = polynomial_sframe(training['sqft_living'], 6)
my_features = poly_train_6.column_names() # get the name of the features
poly_train_6['price'] = training['price'] # add price to the data since it's the target
model_6 = graphlab.linear_regression.create(poly_train_6, target = 'price', features = my_features, validation_set = None, verbose=False)
#model.get("coefficients")
poly_test = polynomial_sframe(testing['sqft_living'], i)
poly_test['price'] = testing['price']
pred = model_6.predict(poly_test)
error = pred - testing['price']
error = error * error
print "i = ",6," ",error.sum()

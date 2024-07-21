# Linear regression

### Linear regression is a statistic method use for modelisation between dependant variable and indepedant variable 
> In simple terms, it aims to establish a straight line that best fits the observed data points in a coordinate space.


## How its work 

### Linear regression is based on this formula
    > y^= theta0 + theta1 * x


## To perform a linear regression we have to use a Gradient descent
> Optimized algorithmn that learn by itself with a number of iterations and a learning rate
> It permit to find the minimum error of the linear regressions
 ### The formula
    t0 = learning * (1/ n_sample) * sum ( y^ - y)
    t1 = learning * (1/ n_sample) * sum ( y^ - y) * mileages ( or x)
    

To calculate the error between predictions and real prices we can use few formula to see if the performance of the linear regression is good and if we dont have to change the lr , n_iterate 

we can use
* R2 : 1 - (sum (prices - predict) ^ 2) / (sum (prices - mean_prices)^ 2) 
> it gave use a performance between [0;1] 
* MSE ( mean square error)  1/n_samples - (prices​−predict ​) ^ 2
* its give us a value and our goal is to reduce the mse ( more the mse is near 0 the better is)


# The importance of Normalisation or standardization
when we have different scale its crucial to normalize or standize our data , so the different scale can't ruin our model
#### Its permit to have our data on the same scale [0;1]

There two of transformation of the data

- We can use Z-scores ( standardization)
> x - mean / std

- We can use min max (normalisation)
- (x- min_val) / ( max_val - min_val)
- 
Dont forget to  change the theta0 and theta1 if you use normalisation or standardization
# Thompson Sampling

# Import dataset
dataset = read.csv('Ads_CTR_Optimisation.csv')

# Implement Thompson Sampling algorithm
N = 10000
d = 10
ads_selected = integer()
numbersOfRewards_1 = integer(d)
numbersOfRewards_0 = integer(d)
total_reward = 0
for (n in 1:N){
  ad = 0
  maxRandom = 0
  for (i in 1:d){
    randomBeta = rbeta(n=1, shape1=numbersOfRewards_1[i]+1, shape2=numbersOfRewards_0[i]+1)
    if (randomBeta > maxRandom){
      maxRandom = randomBeta
      ad = i
    }
  }
  ads_selected = append(ads_selected, ad)
  reward = dataset[n, ad]
  if (reward == 1){
    numbersOfRewards_1[ad] = numbersOfRewards_1[ad] + 1
  }
  else{
    numbersOfRewards_0[ad] = numbersOfRewards_0[ad] + 1
  }
  total_reward = total_reward + reward
}

# Visualize results in histogram
hist(ads_selected, 
     col = 'blue',
     main = 'Histogram of ads selections',
     xlab = 'Ad #',
     ylab = 'Number of times each ad was selected')
import pandas as pd
import numpy as np
import time
'''
pricelist = pd.DataFrame(columns=['Name','Price','GPU'])
    
def x(GPU):
    from selectorlib import Extractor
    import requests 
    import json 
    from time import sleep


    # Create an Extractor by reading from the YAML file
    e = Extractor.from_yaml_file('search_results.yml')

    def scrape(url):  

        headers = {
            'dnt': '1',
            'upgrade-insecure-requests': '1',
            'user-agent': "Mozilla/5.0 (Windows NT 6.3; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/38.0.2125.111 Safari/537.36",
            'accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,image/apng,*/*;q=0.8,application/signed-exchange;v=b3;q=0.9',
            'sec-fetch-site': 'same-origin',
            'sec-fetch-mode': 'navigate',
            'sec-fetch-user': '?1',
            'sec-fetch-dest': 'document',
            'referer': 'https://www.amazon.com/',
            'accept-language': 'en-US;q=0.9,en;q=0.8',
        }
        # Download the page using requests
        print("Downloading %s"%url)
        r = requests.get(url, headers=headers)
        # Simple check to check if page was blocked (Usually 503)
        if r.status_code > 500:
            if "To discuss automated access to Amazon data please contact" in r.text:
                print("Page %s was blocked by Amazon. Please try using better proxies\n"%url)
            else:
                print("Page %s must have been blocked by Amazon as the status code was %d"%(url,r.status_code))
            return None
        # Pass the HTML of the page and create 
        return e.extract(r.text)

    # product_data = []
    
    with open("search_results_urls.txt",'r') as urllist, open('search_results_output.jsonl','w') as outfile:        
        for url in urllist.read().splitlines():
            data = scrape(url) 
                    
            if data:
                for product in data['products']:
                    leng = len(GPU.split(" "))
                    count=0
                    for gpu in GPU.split(" "):
                        
                        if gpu in product['title']:
                            count+=1
                        else:
                            pass 
                        try:
                            if len(GPU.split(" "))==count:
                                pricelist.loc[len(pricelist)] = [product['title'],float(product['price'].replace(",",'').replace("$",'')),GPU]
                        except:
                            pass    

                    product['search_url'] = url
                    print("Saving Product: %s"%product['title'])
                    json.dump(product,outfile)
                    outfile.write("\n")
                    # sleep(5)
        
miner = pd.read_csv("miner.csv")
gpu = pd.read_csv("GPU.csv")

gpu['profit'] = gpu['Net Profit'].str.split(" ").str[0]
miner['profit'] = miner['Net Profit'].str.split(" ").str[0]
gpu['profit'] =gpu['profit'].astype(float)
miner['profit'] =miner['profit'].astype(float)

for gp in gpu['Model'].values:
    f = open("search_results_urls.txt","w")
    f.write("https://www.amazon.com/s?k="+str(gp))
    f.close()
    x(gp)

print(pricelist.head())
pricelist.to_csv("prices_miner.csv")

'''
def knapsack_unbounded_dp(items, C):
    # order by max value per item size
    items = sorted(items, key=lambda item: item[VALUE]/float(item[SIZE]), reverse=True)
 
    # Sack keeps track of max value so far as well as the count of each item in the sack
    sack = [(0, [0 for i in items]) for i in range(0, C+1)]   # value, [item counts]
 
    for i,item in enumerate(items):
        name, size, value = item
        for c in range(size, C+1):
            sackwithout = sack[c-size]  # previous max sack to try adding this item to
            trial = sackwithout[0] + value
            used = sackwithout[1][i]
            if sack[c][0] < trial:
                # old max sack with this added item is better
                sack[c] = (trial, sackwithout[1][:])
                sack[c][1][i] +=1   # use one more
 
    value, bagged = sack[C]
    numbagged = sum(bagged)
    size = sum(items[i][1]*n for i,n in enumerate(bagged))
    # convert to (iten, count) pairs) in name order
    bagged = sorted((items[i][NAME], n) for i,n in enumerate(bagged) if n)
 
    return value, size, numbagged, bagged
gpu = pd.read_csv("GPU.csv")
prices  = pd.read_csv("prices.csv",index_col=0)

gpu['profit'] = gpu['Net Profit'].str.split(" ").str[0]

gpu['profit'] =gpu['profit'].astype(float)

asdprices = prices.sort_values(by = 'Price',ascending = True)
decprices = prices.sort_values(by = 'Price',ascending = False)
low = asdprices.drop_duplicates(subset ="GPU",keep = 'first')
high = decprices.drop_duplicates(subset ="GPU",keep = 'first')

gpu = gpu[gpu['Model'].isin(list(low['GPU'].values))]
gpu = gpu[['Model','profit']]

gpu = gpu.sort_values(by = 'Model',ascending = True)
low = low.sort_values(by = 'GPU',ascending = True)
high = high.sort_values(by = 'GPU',ascending = True)
low['profit'] = gpu['profit'].values
high['profit'] = gpu['profit'].values
low = low[['GPU','Price','profit']]
high = high[['GPU','Price','profit']]
low['Recoverytimeindays'] = low['Price']/low['profit']
high['Recoverytimeindays'] = high['Price']/high['profit']

low = low.sort_values(by = 'Recoverytimeindays',ascending = True)
high = high.sort_values(by = 'Recoverytimeindays',ascending = True)



from operator import itemgetter as iget
from itertools import product
from random import shuffle
low = low[['GPU','Price','profit']]
high = high[['GPU','Price','profit']]
low['Price'] = low['Price'].astype(int)
high['Price'] = high['Price'].astype(int)

NAME, SIZE, VALUE = range(3)
itemslow = low.values
itemshigh = high.values

capacity = 5000


# Simple Case
print("Money: ",capacity," :", knapsack_unbounded_dp(high.values,capacity)) 


# Variation in Money
for loop in range(1000,11000,1000):
    print("Money: ",loop,",","Recovery time: ",int(loop/knapsack_unbounded_dp(high.values, loop)[0]),knapsack_unbounded_dp(high.values, loop)) 



# Variation in time period
# We considered a constant decrease of 0.1 in profit.
possiblesolutions = {}
timesteps = 10
temphigh = high.copy()
for loop in range(0,timesteps):
    print("Decreasing profit values for all GPUs: ",loop)
    print(knapsack_unbounded_dp(itemshigh,capacity))
    possiblesolutions[loop]= knapsack_unbounded_dp(itemshigh,capacity)
    temphigh['profit']= temphigh['profit'] - 0.1
    itemshigh = temphigh.values
tableoftimestep = pd.DataFrame()
print(high)
for key in possiblesolutions:
    print(possiblesolutions[key])
    tempprofit = 0
    tempprofitlist = []
    for time in range(0,timesteps): 
        temp = 0
        for GPU in possiblesolutions[key][3]:
            temp = temp+ high[high['GPU']==GPU[0]]['profit'].values[0]*GPU[1]-0.1*time*GPU[1]
        tempprofitlist.append(temp)    
    tableoftimestep["time_step"+str(key)] = tempprofitlist
print(tableoftimestep)
sum_column = tableoftimestep.sum(axis=0)
print(sum_column)
exit(0)


'''
values = [
    360, 83, 59, 130, 431, 67, 230, 52, 93, 125, 670, 892, 600, 38, 48, 147,
    78, 256, 63, 17, 120, 164, 432, 35, 92, 110, 22, 42, 50, 323, 514, 28,
    87, 73, 78, 15, 26, 78, 210, 36, 85, 189, 274, 43, 33, 10, 19, 389, 276,
    312
]
weights = [[
    7, 0, 30, 22, 80, 94, 11, 81, 70, 64, 59, 18, 0, 36, 3, 8, 15, 42, 9, 0,
    42, 47, 52, 32, 26, 48, 55, 6, 29, 84, 2, 4, 18, 56, 7, 29, 93, 44, 71,
    3, 86, 66, 31, 65, 0, 79, 20, 65, 52, 13
]]
capacities = [850]
solver = pywrapknapsack_solver.KnapsackSolver(
    pywrapknapsack_solver.KnapsackSolver.
    KNAPSACK_MULTIDIMENSION_BRANCH_AND_BOUND_SOLVER, 'KnapsackExample')

solver.Init(values, weights, capacities)
computed_value = solver.Solve()
packed_items = []
packed_weights = []
total_weight = 0
print('Total value =', computed_value)
for i in range(len(values)):
    if solver.BestSolutionContains(i):
        packed_items.append(i)
        packed_weights.append(weights[0][i])
        total_weight += weights[0][i]
print('Total weight:', total_weight)
print('Packed items:', packed_items)
print('Packed_weights:', packed_weights)

packed_items = [x for x in range(0, len(weights[0]))
                  if solver.BestSolutionContains(x)]
'''

# Python3 program to find maximum
# achievable value with a knapsack
# of weight W and multiple instances allowed.

# Returns the maximum value
# with knapsack of W capacity
def unboundedKnapsack(W, n, val, wt):

	# dp[i] is going to store maximum
	# value with knapsack capacity i.
	dp = [0 for i in range(W + 1)]

	ans = 0

	# Fill dp[] using above recursive formula
	for i in range(W + 1):
		for j in range(n):
			if (wt[j] <= i):
				dp[i] = max(dp[i], dp[i - wt[j]] + val[j])

	return dp[W]

# Driver program
W = 10000
val = [1.44, 1.89, 2.10]
wt = [2200, 3899, 7000]
n = len(val)

print(unboundedKnapsack(W, n, val, wt))

# This code is contributed by Anant Agarwal.


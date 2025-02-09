from support import *
import matplotlib.pyplot as plt

# schema is (Date, SeaLevel)
# data = pd.read_csv('data/dead_sea.csv')
# data['Date'] = pd.to_datetime(data['Date'], dayfirst=True)

# Plot the data
# dates = pd.to_datetime(data['Date'], dayfirst=True)
# sea_levels = data['SeaLevel']
# plt.plot(dates, sea_levels)
# plt.xlabel('Date')
# plt.ylabel('Sea Level')
# plt.title('Dead Sea Level Over Time')
# plt.show()


# left = rectangular_region(data, {'Date': ('"01/01/1992"', '"01/01/2025"')}) # <--- Not correct!!!
# right = rectangular_region(data, {'Date': ('"01/01/1976"', '"31/12/1992"')}) # <--- This too!!!
# func = lambda x: x.SeaLevel
# statement = (0, np.inf)

# print(baseline_unconstrained(left, right, func, statement))


# schema is (Country, Year, GDP)
data = pd.read_csv('data/africa_gdp_new.csv')

# Plot the data
# countries = data['Country'].unique()
# for country in countries:
#     country_data = data[data['Country'] == country]
#     years = country_data['Year']
#     gdps = country_data['GDP']
#     plt.plot(years, gdps, label=country)
# plt.xlabel('Year')
# plt.ylabel('GDP')
# plt.title(f'{country} GDP Over Time')
# plt.legend()
# plt.show()


left = rectangular_region(data, {'Year': (2009, 2009)})
right = rectangular_region(data, {'Year': (2008, 2008)})
func = lambda x: x.GDP
statement = (-100_000_000, np.inf)
constraints = lambda x1, x2: x1.Country == x2.Country

# print(baseline_constrained(left, right, func, statement, constraints))
# print(exact_constrained(left, right, func, statement, constraint))
# print(pair_sampling(example, left, right, func, statement, constraints, 0.95))

# print(tightest_statement(left, right, func, 0.99))
# print(most_supported_statement(left, right, func, 1_000_000_000, constraints))



####################################################

# Example usage

# tuples = [
#     ('NewYork', 0, 20), ('NewYork', 1, 19), ('NewYork', 2, 18), ('NewYork', 3, 17), ('NewYork', 4, 16), ('NewYork', 5, 15),
#     ('NewYork', 6, 14), ('NewYork', 7, 15), ('NewYork', 8, 16), ('NewYork', 9, 17), ('NewYork', 10, 18), ('NewYork', 11, 19),
#     ('NewYork', 12, 20), ('NewYork', 13, 21), ('NewYork', 14, 22), ('NewYork', 15, 23), ('NewYork', 16, 24), ('NewYork', 17, 25),
#     ('NewYork', 18, 26), ('NewYork', 19, 27), ('NewYork', 20, 28), ('NewYork', 21, 29), ('NewYork', 22, 30), ('NewYork', 23, 31),
#     ('LosAngeles', 0, 30), ('LosAngeles', 1, 29), ('LosAngeles', 2, 28), ('LosAngeles', 3, 27), ('LosAngeles', 4, 26), ('LosAngeles', 5, 25),
#     ('LosAngeles', 6, 24), ('LosAngeles', 7, 25), ('LosAngeles', 8, 26), ('LosAngeles', 9, 27), ('LosAngeles', 10, 28), ('LosAngeles', 11, 29),
#     ('LosAngeles', 12, 30), ('LosAngeles', 13, 31), ('LosAngeles', 14, 32), ('LosAngeles', 15, 33), ('LosAngeles', 16, 34), ('LosAngeles', 17, 35),
#     ('LosAngeles', 18, 36), ('LosAngeles', 19, 37), ('LosAngeles', 20, 38), ('LosAngeles', 21, 39), ('LosAngeles', 22, 40), ('LosAngeles', 23, 41),
#     ('Chicago', 0, 10), ('Chicago', 1, 9), ('Chicago', 2, 8), ('Chicago', 3, 7), ('Chicago', 4, 6), ('Chicago', 5, 5),
#     ('Chicago', 6, 4), ('Chicago', 7, 5), ('Chicago', 8, 6), ('Chicago', 9, 7), ('Chicago', 10, 8), ('Chicago', 11, 9),
#     ('Chicago', 12, 10), ('Chicago', 13, 11), ('Chicago', 14, 12), ('Chicago', 15, 13), ('Chicago', 16, 14), ('Chicago', 17, 15),
#     ('Chicago', 18, 16), ('Chicago', 19, 17), ('Chicago', 20, 18), ('Chicago', 21, 19), ('Chicago', 22, 20), ('Chicago', 23, 21),
# ]

# example = pd.DataFrame(tuples, columns=['city', 'hour', 'temperature'])


# left = rectangular_region(example, {'city': ('"NewYork"', '"NewYork"')})
# right = rectangular_region(example, {'city': ('"LosAngeles"', '"LosAngeles"')})
# func = lambda x: x.temperature
# statement = (0, np.inf)#, 0)
# constraints = lambda x1, x2: x1.hour == x2.hour
# confidence = 0.95
# range_width = 10

# print(baseline_unconstrained(left, right, func, statement))
# print(exact_unconstrained(left, right, func, statement))
# print(baseline_constrained(left, right, func, statement, constraints))
# print(exact_constrained(left, right, func, statement, constraints))
# print(pair_sampling(example, left, right, func, statement, constraints, 0.95))
# print(point_sampling(example, left, right, func, statement))
# print(tightest_statement(left, right, func, confidence)#, constraints))
# print(most_supported_statement(left, right, func, range_width)#, constraints))
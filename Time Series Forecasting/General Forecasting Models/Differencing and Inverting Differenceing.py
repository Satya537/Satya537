# -*- coding: utf-8 -*-
"""
Created on Sun May  9 11:23:43 2021

@author: satyajit.mahapatra
"""
# difference dataset
def difference(data, interval):
	return [data[i] - data[i - interval] for i in range(interval, len(data))]

# invert difference
def invert_difference(orig_data, diff_data, interval):
	return [diff_data[i-interval] + orig_data[i-interval] for i in range(interval, len(orig_data))]
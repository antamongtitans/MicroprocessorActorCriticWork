#import tools used for data creation
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import operator
import math

#needed for storage
import pickle
import sys
sys.setrecursionlimit(10000)

from dataStructures import *


def is_over_lap_or_out_bounds(rectList, newRect, exteriorRect):
	intersection = False
	buffer = 20 #amount around border
	#check intersection between macros
	for item in rectList:
		intersection = item.is_intersect(newRect, buffer)
		if (intersection):
			#print ("There is an intersection")
			return True
	#check if inside bounding box
	#If any of the sides from A are outside of B
	#return true when there is overlap
	if( newRect.get_minx() <= exteriorRect.get_minx() ):
		return True;

	if( newRect.get_miny() <= exteriorRect.get_miny() ):
		return True;

	if( newRect.get_maxx() >= exteriorRect.get_maxx() ):
		return True;

	if( newRect.get_maxy() >= exteriorRect.get_maxy() ):
		return True;
		
	return intersection
	
def add_random_blocks(iterations, nameCount):
	for iter in range(iterations):
		min_x = np.random.randint(low=0, high=5000)
		min_y = np.random.randint(low=0, high=5000)
		width = np.random.randint(low=200, high=1000)
		height = np.random.randint(low=200, high=1000)
		max_x = min_x + width
		max_y = min_y + height
		tmp_rect = Rectangle_struct(str(nameCount), min_x, max_x, min_y, max_y )
		nameCount = nameCount + 1
		if (not is_over_lap_or_out_bounds(rect_obj, tmp_rect, extRect)):
			rect_obj.append(tmp_rect)
#create recantangle

def create_new_rectangle(chipName):
	# build a rectangle in axes coords units in nanometers
	left, width = 0, np.random.normal(5000, 50)
	bottom, height = 0, np.random.normal(5000, 50)
	right = left + width
	top = bottom + height
	extRect = Rectangle_struct("ExteriorRect", left, width, bottom, height)

	buffer_macros = 30

	#Create List of rectangle objects
	rect_obj = []

	#Randomly shrink item either by subtraction of division
	ops = {'-':operator.sub, '/':operator.truediv}
	#randomly choose if square of rectangular
	rec_squ = ['square', 'rectangle']

		#iterate 100 times
	for iter in range(100):
		#other VARIABLES
		nameCount = 0

		#Intitial startup to make smaller
		div_width = np.random.normal(0, 3)
		div_width = abs(div_width)
		if (div_width < 2):
			div_width = 2
		macro_width = width / div_width
		rec_squ_choice = np.random.choice(rec_squ)
		if (rec_squ_choice == 'square'):
			macro_height = macro_width
		else:
			selector = np.random.normal(0, 0.5)
			#select whether it should be larger or smaller based
			#on positive or negative number then divide according 
			div = False
			if (selector < 0):
				div = True
			selector = abs(selector)
			if (selector < 1):
				macro_height = macro_width
			else:
				if (div):
					macro_height = macro_width / selector
				else:
					macro_height = macro_width * selector

		currentX = (np.random.randint(low=1, high=int((abs(width - macro_width) + 10) / 10)) * 10)
		currentY = (np.random.randint(low=1, high=int((abs(height - macro_height) + 10) / 10)) * 10)

		topy = currentY + macro_height
		topx = currentX + macro_width

		tmp_rect = Rectangle_struct(str(nameCount), currentX, topx, currentY, topy)
		nameCount = nameCount + 1
		if (not is_over_lap_or_out_bounds(rect_obj, tmp_rect, extRect)):
			rect_obj.append(tmp_rect)

		nameCount = nameCount + 1

		currentX = (np.random.randint(low=0, high=500) * 10)
		currentY = (np.random.randint(low=0, high=500) * 10)

		#REMOVE this later
		DEBUG = True

		#lets loop this sucker
		while (macro_width > 100 or macro_height > 100):
			#print("Macro Width: " + str(macro_width))
			#Division is half as likely
			opChoice = np.random.choice(list(ops.keys()))
			#print(opChoice)
			#Subtraction by gausian distribution of half width
			if (opChoice == '-'):
				sub_width = np.random.normal(50, 10)
				if (sub_width < 0):
					sub_width = 0
				#print("Sub width: " + str(sub_width))
				macro_width = macro_width - sub_width
				rec_squ_choice = np.random.choice(rec_squ)
				if (rec_squ_choice == 'square'):
					macro_height = macro_width
				else:
					selector = np.random.normal(0, 0.5)
					#select whether it should be larger or smaller based
					#on positive or negative number then divide according 
					div = False
					if (selector < 0):
						div = True
						#print("Divide ")
					selector = abs(selector)
					#print("Selector multi: " + str(selector))
					if (selector < 1):
						macro_height = macro_width
					else:
						if (div):
							macro_height = macro_width / selector
						else:
							macro_height = macro_width * selector
			else:
				div_width = np.random.normal(0, 2)
				div_width = abs(div_width)
				if (div_width < 2):
					div_width = 2
				#print("Divide width: " + str(div_width))
				macro_width = macro_width / div_width
				rec_squ_choice = np.random.choice(rec_squ)
				if (rec_squ_choice == 'square'):
					macro_height = macro_width
				else:
					selector = np.random.normal(0, 0.5)
					#select whether it should be larger or smaller based
					#on positive or negative number then divide according 
					div = False
					if (selector < 0):
						div = True
					selector = abs(selector)
					if (selector < 1):
						macro_height = macro_width
					else:
						if (div):
							macro_height = macro_width / selector
						else:
							macro_height = macro_width * selector
			
			#Random number created
			shape_multiplier = np.random.normal(0, 2)
			shape_multiplier = abs(shape_multiplier)
			#Round to nearest integer
			shape_multiplier = round(shape_multiplier)
			#make even by multiplying by 2
			shape_multiplier = 2 * int(shape_multiplier)
			
			#Test Squares so set shape multiplier to 4
			shape_multiplier = 20

			#if its less than 0.5 then make it 1
			#check if perfect sqaure
			if (shape_multiplier <= 1):
				shape_multiplier = 1
				is_square = False
			elif (int(shape_multiplier**0.5)**2 == int(shape_multiplier)):
				is_square = True
			else:
				is_square = False
			
			#print("shape_multiplier: " + str(shape_multiplier))
			#Intial locations for squre of macros
			squareCounter = 0
			square_x_og_location = currentX
			square_y_og_location = currentY
			first_time_y_9 = True
			first_time_y_16 = True
			first_time = True
			
			#REMOVE THIS AFTER TESTING
			if (DEBUG):
				macro_height = 300
				macro_width = 300
				DEBUG = False
			
			#Decide number of columns
			num_columns = np.random.randint(low=1, high=8)
			#num_rows round up of total / columns
			num_rows = math.ceil(shape_multiplier / num_columns)
			
			#starts at one since updated after placement
			row_count = 1
			column_count = 0
			
			
			for x in range(shape_multiplier):
				if (is_square):
					#Create Sub Rectangles
					topy = currentY + macro_height
					topx = currentX + macro_width
					tmp_rect = Rectangle_struct(str(nameCount), currentX, topx, currentY, topy)
					nameCount = nameCount + 1
					if (not is_over_lap_or_out_bounds(rect_obj, tmp_rect, extRect)):
						rect_obj.append(tmp_rect)
					if (squareCounter == 0):
						currentX = currentX + macro_width + buffer_macros
					elif (squareCounter == 1):
						currentY = currentY + macro_height + buffer_macros
						currentX = currentX - macro_width - buffer_macros
					elif (squareCounter == 2):
						currentX = currentX + macro_width + buffer_macros
					elif (squareCounter == 3):
						currentX = square_x_og_location
						currentY = currentY + macro_height + buffer_macros
						#need to iterate + 1 to make my life easier later so its not leading
						squareCounter = squareCounter + 1
					if ((squareCounter**0.5) > 2 and (squareCounter**0.5) < 3):
						if ((squareCounter**0.5) < 2.6):
							currentX = currentX + macro_width + buffer_macros
						else:
							if (first_time_y_9):
								currentY = square_y_og_location
								first_time_y_9 = False
							else:
								currentY = currentY + macro_height + buffer_macros
					elif ((squareCounter**0.5) >= 3 and (squareCounter**0.5) <= 4):
						if (first_time):
							currentX = square_x_og_location
							currentY = square_y_og_location + 3 * ( macro_height + buffer_macros )
							first_time = False
						elif ((squareCounter**0.5) < 3.6):
							currentX = currentX + macro_width + buffer_macros
						else:
							if (first_time_y_16):
								currentY = square_y_og_location
								first_time_y_16 = False
							else:
								currentY = currentY + macro_height + buffer_macros
				else:
					#Create Sub Rectangles
					topy = currentY + macro_height
					topx = currentX + macro_width
					tmp_rect = Rectangle_struct(str(nameCount), currentX, topx, currentY, topy)
					nameCount = nameCount + 1
					if (not is_over_lap_or_out_bounds(rect_obj, tmp_rect, extRect)):
						rect_obj.append(tmp_rect)
					
					if ( row_count < num_rows):
						currentX = currentX + macro_width + buffer_macros
						row_count = row_count + 1
					else:
						if ( column_count < num_columns):
							currentY = currentY + macro_height + buffer_macros
							column_count = column_count + 1
							#reset rows for next column
							currentX = square_x_og_location
							row_count = 1
				squareCounter = squareCounter + 1
			currentX = (np.random.randint(low=0, high=500) * 10)
			currentY = (np.random.randint(low=0, high=500) * 10)

	#get number of objects in list
	numRectObj = len(rect_obj)

	#Create Random Connections
	for item in rect_obj:
		#Generate random number of connections
		num_connections = np.random.randint(low=1, high=(numRectObj/10))
		#Generate random sample of equal distance
		tmp_sample = random.sample(rect_obj, num_connections)
		finale_sample = tmp_sample
		#See Distance and determine if its a good one to use
		for connection in tmp_sample:
			#Calulate distance between item and connection
			distance = ( (connection.get_middle_x() - item.get_middle_x())**2 
				+ (connection.get_middle_y() - item.get_middle_y())**2 )**0.5
			#Determine how likely it would be connected
			max_distance = abs(np.random.normal((5000/1.5), 2000))
			if (max_distance < distance):
				finale_sample.remove(connection)
		item.set_connect(tmp_sample)
	#return the chip to be stored
	print("Made chip: " + chipName)
	return Chip(chipName, height, width, rect_obj)
	
#Create array to store in pickle
chip_array = []
for i in range (1000):
	chip_name = "chip_" + str(i)
	tmpChip = create_new_rectangle(chip_name)
	chip_array.append(tmpChip)
#store object in pickle Array of rectangles
pickle.dump( chip_array, open( "save_chips_1000.p", "wb" ) )


